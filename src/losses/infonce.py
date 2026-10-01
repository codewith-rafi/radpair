"""Symmetric InfoNCE (CLIP / Dasgupta Algorithm 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * image_emb @ text_emb.t()
    labels = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)
