"""Plus missing-modality val eval: same DiseaseHead; image-only vs text-only.

Playbook: keep the disease head trained on image embeds. Image-only / image_baseline
use encode_image; text-only feeds encode_text into the same head (honest transfer).
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
import yaml
from torch.cuda.amp import autocast
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
from src.losses.disease_bce import masked_bce_with_logits, multilabel_auroc_f1
from src.models.disease_head import DiseaseHead
from src.models.dual_encoder import DualEncoder


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
        print(f"warn missing keys (ok if none critical): {missing[:8]}", flush=True)
    if bool(cfg.get("freeze_encoder", True)):
        for p in model.parameters():
            p.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


def load_disease_head(cfg: dict, device: torch.device, ckpt_path: Path) -> DiseaseHead:
    head = DiseaseHead(embed_dim=int(cfg["embed_dim"]), n_labels=int(cfg["n_labels"]))
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    if not isinstance(blob, dict) or "disease_head" not in blob:
        raise SystemExit(f"unexpected disease checkpoint format: {ckpt_path}")
    head.load_state_dict(blob["disease_head"])
    head = head.to(device)
    head.eval()
    return head


@torch.no_grad()
def eval_condition(
    encoder: DualEncoder,
    head: DiseaseHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    condition: str,
) -> dict:
    if condition not in {"image_only", "text_only", "image_baseline"}:
        raise ValueError(f"unknown condition: {condition}")
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
        with autocast(enabled=use_amp):
            if condition in {"image_only", "image_baseline"}:
                emb = encoder.encode_image(pixels)
            else:
                emb = encoder.encode_text(ids, attn)
            logits = head(emb.float() if not use_amp else emb)
            loss = masked_bce_with_logits(logits.float(), labels, mask)
        total_loss += float(loss.item())
        n_batches += 1
        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    metrics = multilabel_auroc_f1(probs, labels, masks, threshold=threshold)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["condition"] = condition
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "plus_missing_modality.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if str(cfg.get("u_policy", "")).strip().lower() != "ignore":
        raise SystemExit("u_policy must be ignore for this eval")
    set_seed(int(cfg["seed"]))

    val_manifest = resolve_path(ROOT, str(cfg["val_manifest"]))
    val_rows = load_plus_pair_rows(val_manifest, allowed_splits={"val"})
    n_va_exp = int(cfg.get("expected_n_val", 0) or 0)
    if n_va_exp and len(val_rows) != n_va_exp:
        raise SystemExit(f"val has {len(val_rows)} rows, expected {n_va_exp}")

    cache_root = resolve_path(ROOT, str(cfg["image_cache_root"]))
    assert_under_repo(cache_root, "image_cache_root")
    peft_ckpt = resolve_path(ROOT, str(cfg["peft_ckpt"]))
    disease_ckpt = resolve_path(ROOT, str(cfg["disease_ckpt"]))
    if not peft_ckpt.is_file():
        raise SystemExit(f"peft_ckpt missing: {peft_ckpt}")
    if not disease_ckpt.is_file():
        raise SystemExit(f"disease_ckpt missing: {disease_ckpt}")

    exp = resolve_path(ROOT, str(cfg["experiment_dir"]))
    assert_under_repo(exp, "experiment_dir")
    exp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, exp / "config.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(
        f"device={device} amp={use_amp} n_val={len(val_rows)} cache={cache_root} "
        f"peft={peft_ckpt} disease={disease_ckpt} python={sys.executable}",
        flush=True,
    )
    print(
        "missing-modality: image_only vs text_only with shared DiseaseHead",
        flush=True,
    )

    text_encoder = str(resolve_path(ROOT, str(cfg["text_encoder"])))
    tok_kw = {}
    if Path(text_encoder).is_dir():
        tok_kw["local_files_only"] = True
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder, **tok_kw)

    max_len = int(cfg["max_text_length"])
    val_ds = CheXpertPlusPairDataset(
        val_rows, tokenizer, max_len, image_cache_root=cache_root, require_cache=True
    )
    val_loader = make_loader(val_ds, int(cfg["batch_size"]), int(cfg.get("num_workers", 4)))

    encoder = load_peft_encoder(cfg, text_encoder, device, peft_ckpt)
    head = load_disease_head(cfg, device, disease_ckpt)
    thr = float(cfg.get("decision_threshold", 0.5))

    conditions = ["image_baseline", "image_only", "text_only"]
    results: dict[str, dict] = {}
    for cond in conditions:
        print(f"eval condition={cond}…", flush=True)
        m = eval_condition(encoder, head, val_loader, device, use_amp, thr, cond)
        results[cond] = {
            "loss": m["loss"],
            "macro_auroc": m["macro_auroc"],
            "macro_f1_per_label": m["macro_f1_per_label"],
            "macro_f1_thresholded_pooled": m.get("macro_f1_thresholded_pooled"),
            "n_labels_with_auroc": m.get("n_labels_with_auroc"),
            "per_label": m.get("per_label"),
        }
        print(
            f"  {cond}  loss={m['loss']:.4f}  macro_auroc={m['macro_auroc']:.4f}  "
            f"macro_f1={m['macro_f1_per_label']:.4f}",
            flush=True,
        )

    payload = {
        "device": str(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "n_val": len(val_rows),
        "u_policy": "ignore",
        "dicom_preprocess": str(cfg.get("dicom_preprocess", "chambon_appendix_a")),
        "peft_ckpt": str(peft_ckpt),
        "disease_ckpt": str(disease_ckpt),
        "decision_threshold": thr,
        "label_keys": list(CHEXBERT_LABEL_KEYS),
        "conditions": results,
        "table": {
            "image_baseline_macro_auroc": results["image_baseline"]["macro_auroc"],
            "image_only_macro_auroc": results["image_only"]["macro_auroc"],
            "text_only_macro_auroc": results["text_only"]["macro_auroc"],
            "image_baseline_macro_f1": results["image_baseline"]["macro_f1_per_label"],
            "image_only_macro_f1": results["image_only"]["macro_f1_per_label"],
            "text_only_macro_f1": results["text_only"]["macro_f1_per_label"],
        },
        "note": (
            "Missing-modality val eval: same DiseaseHead (trained on image emb). "
            "text_only feeds text emb into that head. "
        ),
    }
    write_json(exp / "metrics.json", payload)
    print(f"wrote {exp / 'metrics.json'}", flush=True)
    print("MISSING_MODALITY_EXIT:0", flush=True)


if __name__ == "__main__":
    main()
