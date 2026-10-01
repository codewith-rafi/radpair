"""embed_image / embed_text for downstream projects.

Wraps DualEncoder.encode_* (already L2-normalised). Default checkpoint is
Plus PEFT_001 (LoRA on BioClinicalBERT query+value). Does not read any
manifest or split — callers supply tensors only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dual_encoder import DualEncoder, parse_freeze_image

# Default: F+I PEFT LoRA checkpoint. Path relative to repo root.
DEFAULT_CKPT = ROOT / "experiments" / "plus_g_peft_001" / "checkpoint.pt"


def _resolve_path(maybe: str) -> str:
    p = Path(maybe)
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def _looks_like_peft(state: dict, cfg: dict) -> bool:
    if any("lora_" in k for k in state):
        return True
    return bool(cfg.get("use_text_lora") or cfg.get("lora_rank"))


@torch.no_grad()
def embed_image(model: DualEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    """Image tensor (B, 3, 224, 224) → L2-normalised (B, d)."""
    model.eval()
    return model.encode_image(pixel_values)


@torch.no_grad()
def embed_text(
    model: DualEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Token ids + mask → L2-normalised (B, d)."""
    model.eval()
    return model.encode_text(input_ids, attention_mask)


def load_dual_encoder(
    ckpt_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> DualEncoder:
    """Load DualEncoder. Default: plus_g_peft_001 (PEFT LoRA). Returns eval mode.

    PEFT checkpoints require enable_text_lora before load_state_dict.
    Non-PEFT (G_frozen / e2e) load without LoRA wrappers.
    """
    path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint missing: {path}. "
            "Pass experiments/plus_g_peft_001/checkpoint.pt (default PEFT share)."
        )
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    if not isinstance(blob, dict) or "model" not in blob:
        raise SystemExit(f"unexpected checkpoint format: {path}")
    cfg = blob.get("cfg") or {}
    state = blob["model"]
    text_encoder = str(cfg.get("text_encoder") or "")
    if not text_encoder:
        raise SystemExit("checkpoint cfg missing text_encoder")
    text_encoder = _resolve_path(text_encoder)
    tok_dir = Path(text_encoder)
    if tok_dir.is_dir():
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = DualEncoder(
        text_encoder_name=text_encoder,
        embed_dim=int(cfg.get("embed_dim", 256)),
        freeze_image=parse_freeze_image(cfg.get("freeze_image", True)),
        freeze_text=bool(cfg.get("freeze_text", True)),
        tau=float(cfg.get("tau", 0.07)),
        learnable_tau=bool(cfg.get("learnable_tau", False)),
    )
    if _looks_like_peft(state, cfg):
        model.enable_text_lora(
            r=int(cfg.get("lora_rank", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.1)),
            target_modules=list(cfg.get("lora_modules", ["query", "value"])),
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys loading ckpt: {list(unexpected)[:8]}")
    if missing:
        print(f"warn missing keys: {list(missing)[:8]}", flush=True)
    model = model.to(device)
    model.eval()
    return model
