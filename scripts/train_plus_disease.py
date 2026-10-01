"""Plus disease auxiliary: BCE + U-Ignore on frozen PEFT dual encoder.

Trains DiseaseHead on image embeddings only. Does not retune InfoNCE.
 CUDA_VISIBLE_DEVICES=1; tmux; write under RADPAIR_ROOT (repo root).
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
from torch.cuda.amp import GradScaler, autocast
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


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate,
        "drop_last": shuffle,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        kwargs["generator"] = g
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
        raise SystemExit(f"unexpected checkpoint format: {ckpt_path}")
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


@torch.no_grad()
def eval_disease(
    encoder: DualEncoder,
    head: DiseaseHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> dict:
    encoder.eval()
    head.eval()
    all_probs, all_labels, all_masks = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["label_mask"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            img = encoder.encode_image(pixels)
            logits = head(img.float() if not use_amp else img)
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
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "plus_disease.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if str(cfg.get("u_policy", "")).strip().lower() != "ignore":
        raise SystemExit("u_policy must be ignore for this trainer")
    set_seed(int(cfg["seed"]))

    train_manifest = resolve_path(ROOT, str(cfg["train_manifest"]))
    val_manifest = resolve_path(ROOT, str(cfg["val_manifest"]))
    train_rows = load_plus_pair_rows(train_manifest, allowed_splits={"train"})
    val_rows = load_plus_pair_rows(val_manifest, allowed_splits={"val"})
    n_tr_exp = int(cfg.get("expected_n_train", 0) or 0)
    n_va_exp = int(cfg.get("expected_n_val", 0) or 0)
    if n_tr_exp and len(train_rows) != n_tr_exp:
        raise SystemExit(f"train has {len(train_rows)} rows, expected {n_tr_exp}")
    if n_va_exp and len(val_rows) != n_va_exp:
        raise SystemExit(f"val has {len(val_rows)} rows, expected {n_va_exp}")

    cache_root = resolve_path(ROOT, str(cfg["image_cache_root"]))
    assert_under_repo(cache_root, "image_cache_root")
    ckpt_path = resolve_path(ROOT, str(cfg["init_ckpt"]))
    if not ckpt_path.is_file():
        raise SystemExit(f"init_ckpt missing: {ckpt_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(
        f"device={device} amp={use_amp} n_train={len(train_rows)} n_val={len(val_rows)} "
        f"cache={cache_root} ckpt={ckpt_path} python={sys.executable}",
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
    train_ds = CheXpertPlusPairDataset(
        train_rows, tokenizer, max_len, image_cache_root=cache_root, require_cache=True
    )
    val_ds = CheXpertPlusPairDataset(
        val_rows, tokenizer, max_len, image_cache_root=cache_root, require_cache=True
    )
    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    seed = int(cfg["seed"])
    train_loader = make_loader(train_ds, bs, nw, True, seed)
    val_loader = make_loader(val_ds, bs, nw, False, seed)

    encoder = load_peft_encoder(cfg, text_encoder, device, ckpt_path)
    head = DiseaseHead(embed_dim=int(cfg["embed_dim"]), n_labels=int(cfg["n_labels"])).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"disease_head_params={n_head} encoder_frozen={cfg.get('freeze_encoder', True)}", flush=True)

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scaler = GradScaler(enabled=use_amp)
    thr = float(cfg.get("decision_threshold", 0.5))

    exp = resolve_path(ROOT, str(cfg["experiment_dir"]))
    assert_under_repo(exp, "experiment_dir")
    exp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, exp / "config.yaml")
    ckpt_out = exp / "checkpoint.pt"
    metrics_path = exp / "metrics.json"

    print("eval untrained disease head…", flush=True)
    untrained = eval_disease(encoder, head, val_loader, device, use_amp, thr)
    print(
        f"untrained_val loss={untrained['loss']:.4f} "
        f"macro_auroc={untrained['macro_auroc']} "
        f"macro_f1_per_label={untrained['macro_f1_per_label']}",
        flush=True,
    )

    history = []
    best_auroc = -1.0
    best_epoch = 0
    stale = 0
    patience = int(cfg["early_stop_patience"])
    stopped_at = None

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        encoder.eval()
        head.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["label_mask"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                with torch.no_grad():
                    img = encoder.encode_image(pixels)
                logits = head(img.detach())
                loss = masked_bce_with_logits(logits.float(), labels, mask)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)

        val_m = eval_disease(encoder, head, val_loader, device, use_amp, thr)
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_loss": val_m["loss"],
            "val_macro_auroc": val_m["macro_auroc"],
            "val_macro_f1_per_label": val_m["macro_f1_per_label"],
            "val_n_labels_with_auroc": val_m["n_labels_with_auroc"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  train_loss {mean_loss:.4f}  val_loss {val_m['loss']:.4f}  "
            f"val_macro_auroc {val_m['macro_auroc']:.4f}  "
            f"val_macro_f1 {val_m['macro_f1_per_label']:.4f}",
            flush=True,
        )

        score = float(val_m["macro_auroc"]) if np.isfinite(val_m["macro_auroc"]) else -1.0
        if score > best_auroc:
            best_auroc = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "disease_head": head.state_dict(),
                    "init_ckpt": str(ckpt_path),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_macro_auroc": score,
                    "label_keys": list(CHEXBERT_LABEL_KEYS),
                },
                ckpt_out,
            )
        else:
            stale += 1

        payload = {
            "device": str(device),
            "python": sys.executable,
            "torch": torch.__version__,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "u_policy": "ignore",
            "init_ckpt": str(ckpt_path),
            "disease_head_params": n_head,
            "label_keys": list(CHEXBERT_LABEL_KEYS),
            "untrained_val": {
                "loss": untrained["loss"],
                "macro_auroc": untrained["macro_auroc"],
                "macro_f1_per_label": untrained["macro_f1_per_label"],
                "n_labels_with_auroc": untrained["n_labels_with_auroc"],
            },
            "best_epoch": best_epoch,
            "best_val_macro_auroc": best_auroc,
            "history": history,
            "early_stopped": False,
            "note": (
                "Disease BCE + U-Ignore on frozen PEFT image embeddings. "
                "Empty labels→0; -1 masked. No test. LibAUC/DeepAUC not used."
            ),
        }
        write_json(metrics_path, payload)

        if stale >= patience:
            stopped_at = epoch
            break

    try:
        blob = torch.load(ckpt_out, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_out, map_location=device)
    head.load_state_dict(blob["disease_head"])
    best_val = eval_disease(encoder, head, val_loader, device, use_amp, thr)
    # stringify per_label for JSON
    per = {CHEXBERT_LABEL_KEYS[int(k)]: v for k, v in best_val["per_label"].items()}

    payload = {
        "device": str(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "u_policy": "ignore",
        "dicom_preprocess": cfg.get("dicom_preprocess", "chambon_appendix_a"),
        "init_ckpt": str(ckpt_path),
        "disease_head_params": n_head,
        "label_keys": list(CHEXBERT_LABEL_KEYS),
        "decision_threshold": thr,
        "untrained_val": {
            "loss": untrained["loss"],
            "macro_auroc": untrained["macro_auroc"],
            "macro_f1_per_label": untrained["macro_f1_per_label"],
            "n_labels_with_auroc": untrained["n_labels_with_auroc"],
        },
        "best_epoch": best_epoch,
        "best_val": {
            "loss": best_val["loss"],
            "macro_auroc": best_val["macro_auroc"],
            "macro_f1_per_label": best_val["macro_f1_per_label"],
            "macro_f1_thresholded_pooled": best_val["macro_f1_thresholded_pooled"],
            "n_labels_with_auroc": best_val["n_labels_with_auroc"],
            "per_label": per,
        },
        "early_stopped": stopped_at is not None,
        "stopped_at_epoch": stopped_at,
        "history": history,
        "note": "Plus disease BCE (U-Ignore) on frozen PEFT_001 image embeddings.",
    }
    write_json(metrics_path, payload)
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_val["macro_auroc"],
        "best_val_macro_f1_per_label": best_val["macro_f1_per_label"],
        "early_stopped": stopped_at is not None,
        "stopped_at_epoch": stopped_at,
        "disease_head_params": n_head,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {metrics_path}")
    print("DISEASE_EXIT:0", flush=True)


if __name__ == "__main__":
    main()
