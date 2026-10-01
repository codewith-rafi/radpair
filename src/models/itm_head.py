"""Binary image–text matching head on concatenated L2 embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class ITMHead(nn.Module):
    """MLP: [img_emb; text_emb] → logit (match vs non-match)."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_emb, text_emb], dim=-1)
        return self.net(x).squeeze(-1)
