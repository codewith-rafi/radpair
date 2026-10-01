"""Plus G_PEFT: LoRA on BioClinicalBERT (query+value) + train projections; ResNet frozen.

Same recipe as G1_003 for fair freeze vs PEFT comparison. Early stop on val. No test.
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
from torch.utils.data import DataLoader, Subset
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

from src.data.dataset import CheXpertPlusPairDataset, load_plus_pair_rows
from src.eval.retrieval import recall_at_k
from src.losses.infonce import symmetric_infonce
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
        "patient_id": [b["patient_id"] for b in batch],
    }


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


@torch.no_grad()
def embed_all(model: DualEncoder, loader: DataLoader, device: torch.device, use_amp: bool):
    model.eval()
    images, texts = [], []
    for batch in loader:
        with autocast(enabled=use_amp):
            ie, te, _ = model(
                batch["pixel_values"].to(device, non_blocking=True),
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
        images.append(ie.float())
        texts.append(te.float())
    return torch.cat(images, dim=0), torch.cat(texts, dim=0)


@torch.no_grad()
def shuffled_recall_mean(
    image_emb: torch.Tensor, text_emb: torch.Tensor, n_trials: int, seed: int
) -> dict:
    g = torch.Generator(device=text_emb.device)
    g.manual_seed(seed)
    trials = []
    for _ in range(n_trials):
        perm = torch.randperm(text_emb.size(0), generator=g, device=text_emb.device)
        trials.append(recall_at_k(image_emb, text_emb[perm]))
    keys = trials[0].keys()
    return {k: float(np.mean([t[k] for t in trials])) for k in keys}


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "plus_g_peft.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if str(cfg.get("lora_targets", "")).strip().lower() != "text_only":
        raise SystemExit("this trainer only supports lora_targets: text_only")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(
        f"device={device} amp={use_amp} n_train={len(train_rows)} n_val={len(val_rows)} "
        f"cache={cache_root} python={sys.executable}",
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

    train_eval_n = min(int(cfg.get("train_eval_n", 512)), len(train_ds))
    rng = np.random.default_rng(seed)
    train_eval_idx = rng.choice(len(train_ds), size=train_eval_n, replace=False).tolist()
    train_eval_ds = Subset(train_ds, train_eval_idx)
    train_eval_loader = make_loader(train_eval_ds, bs, nw, False, seed)
    train_monitor_chance = 1.0 / train_eval_n

    model = DualEncoder(
        text_encoder_name=text_encoder,
        embed_dim=int(cfg["embed_dim"]),
        freeze_image=bool(cfg["freeze_image"]),
        freeze_text=bool(cfg["freeze_text"]),
        tau=float(cfg["tau"]),
        learnable_tau=bool(cfg["learnable_tau"]),
    )
    model.enable_text_lora(
        r=int(cfg["lora_rank"]),
        alpha=int(cfg["lora_alpha"]),
        dropout=float(cfg.get("lora_dropout", 0.1)),
        target_modules=list(cfg.get("lora_modules", ["query", "value"])),
    )
    model = model.to(device)

    proj_params = model.projection_parameters()
    lora_params = model.lora_parameters()
    if not proj_params or not lora_params:
        raise SystemExit(
            f"expected proj+lora trainable, got n_proj={len(proj_params)} n_lora={len(lora_params)}"
        )
    n_proj = sum(p.numel() for p in proj_params)
    n_lora = sum(p.numel() for p in lora_params)
    n_trainable = n_proj + n_lora
    print(
        f"lora_rank={cfg['lora_rank']} alpha={cfg['lora_alpha']} "
        f"n_proj={n_proj} n_lora={n_lora} n_trainable={n_trainable}",
        flush=True,
    )
    opt = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": float(cfg["lr"])},
            {"params": lora_params, "lr": float(cfg["lora_lr"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    scaler = GradScaler(enabled=use_amp)

    exp = resolve_path(ROOT, str(cfg["experiment_dir"]))
    assert_under_repo(exp, "experiment_dir")
    exp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, exp / "config.yaml")
    ckpt_path = exp / "checkpoint.pt"
    metrics_path = exp / "metrics.json"

    print("embedding untrained val…", flush=True)
    ie_v, te_v = embed_all(model, val_loader, device, use_amp)
    untrained_val = recall_at_k(ie_v, te_v)
    print("untrained_val", {k: round(float(v), 6) for k, v in untrained_val.items()}, flush=True)

    history = []
    best_val_i2t = -1.0
    best_epoch = 0
    stale = 0
    patience = int(cfg["early_stop_patience"])
    stopped_at = None
    chance = 1.0 / len(val_rows)
    n_shuffle = int(cfg.get("n_shuffle", 5))

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                image_emb, text_emb, scale = model(
                    batch["pixel_values"].to(device, non_blocking=True),
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
                loss = symmetric_infonce(image_emb, text_emb, scale)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)

        ie_tr, te_tr = embed_all(model, train_eval_loader, device, use_amp)
        ie_va, te_va = embed_all(model, val_loader, device, use_amp)
        train_m = recall_at_k(ie_tr, te_tr)
        val_m = recall_at_k(ie_va, te_va)
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "train_eval_n": train_eval_n,
            **prefix_metrics(train_m, "train"),
            **prefix_metrics(val_m, "val"),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  loss {mean_loss:.4f}  "
            f"train{train_eval_n}_i2t_R@1 {train_m['i2t_R@1']:.4f}  "
            f"val_i2t_R@1 {val_m['i2t_R@1']:.4f}  "
            f"val_t2i_R@1 {val_m['t2i_R@1']:.4f}",
            flush=True,
        )

        val_i2t = float(val_m["i2t_R@1"])
        if val_i2t > best_val_i2t:
            best_val_i2t = val_i2t
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_i2t_R@1": val_i2t,
                },
                ckpt_path,
            )
        else:
            stale += 1

        payload = {
            "device": str(device),
            "python": sys.executable,
            "torch": torch.__version__,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "u_policy": cfg.get("u_policy", "ignore"),
            "dicom_preprocess": cfg.get("dicom_preprocess", "chambon_appendix_a"),
            "lora_rank": int(cfg["lora_rank"]),
            "lora_alpha": int(cfg["lora_alpha"]),
            "lora_targets": cfg.get("lora_targets"),
            "train_eval_n": train_eval_n,
            "train_monitor_chance_R@1": train_monitor_chance,
            "chance_val_R@1": chance,
            "trainable_params": n_trainable,
            "trainable_proj_params": n_proj,
            "trainable_lora_params": n_lora,
            "untrained_val": {k: float(v) for k, v in untrained_val.items()},
            "best_epoch": best_epoch,
            "best_val_i2t_R@1": best_val_i2t,
            "history": history,
            "early_stopped": False,
            "note": (
                "Plus G_PEFT LoRA text_only + Chambon cache. No test. "
                f"Pass: val i2t R@1 > 5*chance (chance={chance:.6f}). "
                f"Train monitor is seeded random {train_eval_n} pairs "
                f"(chance={train_monitor_chance:.6f}); not comparable to val gallery."
            ),
        }
        write_json(metrics_path, payload)

        if stale >= patience:
            stopped_at = epoch
            break

    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(blob["model"])
    ie_va, te_va = embed_all(model, val_loader, device, use_amp)
    ie_tr, te_tr = embed_all(model, train_eval_loader, device, use_amp)
    best_val = recall_at_k(ie_va, te_va)
    best_train = recall_at_k(ie_tr, te_tr)
    shuffle_best = shuffled_recall_mean(ie_va, te_va, n_shuffle, seed)
    gap = float(best_train["i2t_R@1"] - best_val["i2t_R@1"])
    beats_chance = float(best_val["i2t_R@1"]) > 5.0 * chance

    payload = {
        "device": str(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "u_policy": cfg.get("u_policy", "ignore"),
        "dicom_preprocess": cfg.get("dicom_preprocess", "chambon_appendix_a"),
        "lora_rank": int(cfg["lora_rank"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "lora_targets": cfg.get("lora_targets"),
        "train_eval_n": train_eval_n,
        "train_monitor_chance_R@1": train_monitor_chance,
        "chance_val_R@1": chance,
        "trainable_params": n_trainable,
        "trainable_proj_params": n_proj,
        "trainable_lora_params": n_lora,
        "untrained_val": {k: float(v) for k, v in untrained_val.items()},
        "best_epoch": best_epoch,
        "best_val": {k: float(v) for k, v in best_val.items()},
        "best_train_monitor_at_best_val": {k: float(v) for k, v in best_train.items()},
        "shuffle_best_val_mean": shuffle_best,
        "train_val_i2t_gap_monitor": gap,
        "beats_5x_chance": beats_chance,
        "early_stopped": stopped_at is not None,
        "stopped_at_epoch": stopped_at,
        "history": history,
        "note": (
            "Plus G_PEFT LoRA text_only + Chambon Appendix A JPG cache. No test. "
            f"beats_5x_chance: val i2t R@1 > 5*chance with chance={chance:.6f}. "
            f"Train monitor: seeded random {train_eval_n} (not full train gallery). "
            "Do not compare to IU Recall@K."
        ),
    }
    write_json(metrics_path, payload)
    summary = {
        "best_epoch": best_epoch,
        "best_val_i2t_R@1": float(best_val["i2t_R@1"]),
        "best_val_t2i_R@1": float(best_val["t2i_R@1"]),
        "shuffle_mean_i2t_R@1": float(shuffle_best["i2t_R@1"]),
        "beats_5x_chance": beats_chance,
        "early_stopped": stopped_at is not None,
        "stopped_at_epoch": stopped_at,
        "trainable_params": n_trainable,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
