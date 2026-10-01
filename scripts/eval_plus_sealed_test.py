"""Held-out Plus test eval: retrieval + disease + ITM + missing-modality.

Loads PEFT_001 dual encoder and disease/ITM heads. Test split only; no training.
 CUDA_VISIBLE_DEVICES=1; write under RADPAIR_ROOT (repo root).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (
    assert_under_repo,
    chexpert_plus_root,
    default_chexbert_impression,
    default_plus_csv,
    repo_root,
    resolve_repo_path,
)

from src.data.dataset import CHEXBERT_LABEL_KEYS, CheXpertPlusPairDataset, load_plus_pair_rows
from src.eval.retrieval import recall_at_k
from src.losses.disease_bce import masked_bce_with_logits, multilabel_auroc_f1
from src.models.disease_head import DiseaseHead
from src.models.dual_encoder import DualEncoder
from src.models.itm_head import ITMHead


def resolve_path(base: Path, maybe: str) -> Path:
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.float32),
        "label_mask": torch.tensor([b["label_mask"] for b in batch], dtype=torch.float32),
        "patient_id": [b["patient_id"] for b in batch],
    }


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


def load_peft_encoder(cfg: dict, text_encoder: str, device: torch.device, ckpt_path: Path) -> DualEncoder:
    model = DualEncoder(
        text_encoder_name=text_encoder,
        embed_dim=int(cfg["embed_dim"]),
        freeze_image=bool(cfg["freeze_image"]),
        freeze_text=bool(cfg["freeze_text"]),
        tau=0.07,
        learnable_tau=False,
    )
    model.enable_text_lora(
        r=int(cfg["lora_rank"]),
        alpha=int(cfg["lora_alpha"]),
        dropout=float(cfg.get("lora_dropout", 0.1)),
        target_modules=list(cfg.get("lora_modules", ["query", "value"])),
    )
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    if not isinstance(blob, dict) or "model" not in blob:
        raise SystemExit(f"unexpected PEFT checkpoint format: {ckpt_path}")
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys loading PEFT ckpt: {unexpected[:8]}")
    if missing:
        print(f"warn missing keys: {missing[:8]}", flush=True)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_all(encoder: DualEncoder, loader: DataLoader, device: torch.device, use_amp: bool):
    encoder.eval()
    images, texts = [], []
    for batch in loader:
        with _autocast(device, use_amp):
            ie = encoder.encode_image(batch["pixel_values"].to(device, non_blocking=True))
            te = encoder.encode_text(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
        images.append(ie.float())
        texts.append(te.float())
    return torch.cat(images, dim=0), torch.cat(texts, dim=0)


@torch.no_grad()
def shuffled_recall_mean(image_emb: torch.Tensor, text_emb: torch.Tensor, n_trials: int, seed: int) -> dict:
    g = torch.Generator(device=text_emb.device)
    g.manual_seed(seed)
    acc = []
    for _ in range(n_trials):
        perm = torch.randperm(text_emb.size(0), generator=g, device=text_emb.device)
        acc.append(recall_at_k(image_emb, text_emb[perm]))
    keys = acc[0].keys()
    return {k: float(np.mean([m[k] for m in acc])) for k in keys}


def itm_pair_logits(head: ITMHead, image_emb: torch.Tensor, text_emb: torch.Tensor):
    b = image_emb.size(0)
    if b < 2:
        raise RuntimeError("ITM batch size must be >= 2")
    pos = head(image_emb, text_emb)
    neg = head(image_emb, torch.roll(text_emb, shifts=1, dims=0))
    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat(
        [
            torch.ones(b, device=logits.device),
            torch.zeros(b, device=logits.device),
        ],
        dim=0,
    )
    return logits, labels


@torch.no_grad()
def eval_disease_condition(
    encoder: DualEncoder,
    head: DiseaseHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    condition: str,
) -> dict:
    encoder.eval()
    head.eval()
    all_probs, all_labels, all_masks = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["label_mask"].to(device, non_blocking=True)
        with _autocast(device, use_amp):
            if condition == "text_only":
                emb = encoder.encode_text(ids, attn)
            else:
                emb = encoder.encode_image(pixels)
            logits = head(emb.float())
            loss = masked_bce_with_logits(logits.float(), labels, mask)
        total_loss += float(loss.item())
        n_batches += 1
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    metrics = multilabel_auroc_f1(
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_masks),
        threshold=threshold,
    )
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["condition"] = condition
    return metrics


@torch.no_grad()
def eval_itm(
    encoder: DualEncoder,
    head: ITMHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> dict:
    encoder.eval()
    head.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        with _autocast(device, use_amp):
            img = encoder.encode_image(batch["pixel_values"].to(device, non_blocking=True))
            txt = encoder.encode_text(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
            logits, labels = itm_pair_logits(head, img.float(), txt.float())
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    y = np.concatenate(all_labels)
    p = np.concatenate(all_probs)
    return {
        "loss": total_loss / max(n_batches, 1),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, (p >= threshold).astype(np.float32))),
        "n_pairs": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "plus_sealed_test.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(cfg.get("n_test_loaded", 0)) <= 0:
        raise SystemExit("held-out eval requires n_test_loaded > 0")
    if str(cfg.get("u_policy", "")).strip().lower() != "ignore":
        raise SystemExit("u_policy must be ignore")
    set_seed(int(cfg["seed"]))

    test_manifest = resolve_path(ROOT, str(cfg["test_manifest"]))
    test_rows = load_plus_pair_rows(
        test_manifest, allowed_splits={"test"}, allow_test=True
    )
    n_exp = int(cfg.get("expected_n_test", 0) or 0)
    if n_exp and len(test_rows) != n_exp:
        raise SystemExit(f"test has {len(test_rows)} rows, expected {n_exp}")
    if len(test_rows) != int(cfg["n_test_loaded"]):
        raise SystemExit(
            f"n_test_loaded={cfg['n_test_loaded']} but loaded {len(test_rows)} rows"
        )

    cache_root = resolve_path(ROOT, str(cfg["image_cache_root"]))
    assert_under_repo(cache_root, "image_cache_root")
    peft_ckpt = resolve_path(ROOT, str(cfg["peft_ckpt"]))
    disease_ckpt = resolve_path(ROOT, str(cfg["disease_ckpt"]))
    itm_ckpt = resolve_path(ROOT, str(cfg["itm_ckpt"]))
    for pth, name in (
        (peft_ckpt, "peft_ckpt"),
        (disease_ckpt, "disease_ckpt"),
        (itm_ckpt, "itm_ckpt"),
    ):
        if not pth.is_file():
            raise SystemExit(f"{name} missing: {pth}")

    exp = resolve_path(ROOT, str(cfg["experiment_dir"]))
    assert_under_repo(exp, "experiment_dir")
    exp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, exp / "config.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(
        f"device={device} amp={use_amp} n_test={len(test_rows)} cache={cache_root} "
        f"peft={peft_ckpt} python={sys.executable}",
        flush=True,
    )
    print(
        "Held-out test eval: F+I + PEFT_001; report metrics as-is",
        flush=True,
    )

    text_encoder = str(resolve_path(ROOT, str(cfg["text_encoder"])))
    tok_kw = {}
    if Path(text_encoder).is_dir():
        tok_kw["local_files_only"] = True
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder, **tok_kw)

    ds = CheXpertPlusPairDataset(
        test_rows,
        tokenizer,
        int(cfg["max_text_length"]),
        image_cache_root=cache_root,
        require_cache=True,
    )
    loader = make_loader(ds, int(cfg["batch_size"]), int(cfg.get("num_workers", 4)))

    encoder = load_peft_encoder(cfg, text_encoder, device, peft_ckpt)
    thr = float(cfg.get("decision_threshold", 0.5))

    print("retrieval embed…", flush=True)
    ie, te = embed_all(encoder, loader, device, use_amp)
    retrieval = recall_at_k(ie, te)
    shuffle = shuffled_recall_mean(ie, te, int(cfg.get("n_shuffle", 5)), int(cfg["seed"]))
    print(
        f"  retrieval i2t_R@1={retrieval['i2t_R@1']:.4f} t2i_R@1={retrieval['t2i_R@1']:.4f} "
        f"chance={retrieval['chance_R@1']:.6f} shuffle_i2t={shuffle['i2t_R@1']:.4f}",
        flush=True,
    )

    print("disease + missing-modality…", flush=True)
    try:
        dblob = torch.load(disease_ckpt, map_location=device, weights_only=False)
    except TypeError:
        dblob = torch.load(disease_ckpt, map_location=device)
    dhead = DiseaseHead(embed_dim=int(cfg["embed_dim"]), n_labels=int(cfg["n_labels"])).to(device)
    dhead.load_state_dict(dblob["disease_head"])
    dhead.eval()
    disease = {}
    for cond in ("image_only", "text_only"):
        m = eval_disease_condition(encoder, dhead, loader, device, use_amp, thr, cond)
        disease[cond] = {
            "macro_auroc": m["macro_auroc"],
            "macro_f1_per_label": m["macro_f1_per_label"],
            "loss": m["loss"],
            "n_labels_with_auroc": m.get("n_labels_with_auroc"),
        }
        print(
            f"  {cond} macro_auroc={m['macro_auroc']:.4f} macro_f1={m['macro_f1_per_label']:.4f}",
            flush=True,
        )

    print("ITM…", flush=True)
    try:
        iblob = torch.load(itm_ckpt, map_location=device, weights_only=False)
    except TypeError:
        iblob = torch.load(itm_ckpt, map_location=device)
    ihead = ITMHead(embed_dim=int(cfg["embed_dim"])).to(device)
    ihead.load_state_dict(iblob["itm_head"])
    ihead.eval()
    itm = eval_itm(encoder, ihead, loader, device, use_amp, thr)
    print(f"  itm auroc={itm['auroc']:.4f} acc={itm['accuracy']:.4f}", flush=True)

    payload = {
        "device": str(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "n_test": len(test_rows),
        "n_test_loaded": len(test_rows),
        "u_policy": "ignore",
        "dicom_preprocess": str(cfg.get("dicom_preprocess", "chambon_appendix_a")),
        "peft_ckpt": str(peft_ckpt),
        "disease_ckpt": str(disease_ckpt),
        "itm_ckpt": str(itm_ckpt),
        "decision_threshold": thr,
        "label_keys": list(CHEXBERT_LABEL_KEYS),
        "retrieval": {k: float(v) for k, v in retrieval.items()},
        "retrieval_shuffle_mean": {k: float(v) for k, v in shuffle.items()},
        "disease_missing_modality": disease,
        "itm": itm,
        "note": "Held-out test. PEFT_001 + disease/ITM heads.",
    }
    write_json(exp / "metrics.json", payload)
    print(f"wrote {exp / 'metrics.json'}", flush=True)
    print("SEALED_TEST_EXIT:0", flush=True)


if __name__ == "__main__":
    main()
