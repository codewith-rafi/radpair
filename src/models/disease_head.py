"""Linear multi-label disease head on L2 image embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiseaseHead(nn.Module):
    def __init__(self, embed_dim: int = 256, n_labels: int = 14) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_labels)

    def forward(self, image_emb: torch.Tensor) -> torch.Tensor:
        return self.fc(image_emb)
