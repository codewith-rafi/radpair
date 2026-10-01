"""Image↔text Recall@K on a gallery of L2-normalised embeddings."""

from __future__ import annotations

import torch


@torch.no_grad()
def recall_at_k(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    sim = image_emb @ text_emb.t()
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)
    out: dict[str, float] = {}
    for name, matrix in (("i2t", sim), ("t2i", sim.t())):
        ranking = matrix.argsort(dim=1, descending=True)
        ranks = (ranking == targets[:, None]).int().argmax(dim=1)
        for k in ks:
            k_use = min(k, n)
            out[f"{name}_R@{k}"] = float((ranks < k_use).float().mean().item())
    out["chance_R@1"] = 1.0 / n
    return out
