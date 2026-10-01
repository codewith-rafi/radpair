"""CLIP-style dual encoder: ResNet-50 + BioClinicalBERT + projections.

freeze_image: True = all frozen; False = all trainable; "last_block" = train ResNet layer4 only.
freeze_text: True = all frozen; False = all trainable; "last_6_layers" = freeze embeddings + layers 0-5.
Frozen towers stay in eval() during train() so BN/dropout do not update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel


def parse_freeze_image(value) -> bool | str:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    if s == "last_block":
        return "last_block"
    raise ValueError(f"freeze_image must be true/false/last_block, got {value!r}")


def parse_freeze_text(value) -> bool | str:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    if s == "last_6_layers":
        return "last_6_layers"
    raise ValueError(f"freeze_text must be true/false/last_6_layers, got {value!r}")


def _text_last6_trainable(name: str) -> bool:
    """True if BioClinicalBERT param should train under ConVIRT-style last-6 FT."""
    if name.startswith("encoder.layer."):
        try:
            idx = int(name.split(".")[2])
        except (IndexError, ValueError):
            return False
        return idx >= 6
    if name.startswith("pooler."):
        return True
    return False


class DualEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_name: str,
        embed_dim: int = 256,
        freeze_image: bool | str = True,
        freeze_text: bool | str = True,
        tau: float = 0.07,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_proj = nn.Linear(feat_dim, embed_dim)

        text = AutoModel.from_pretrained(text_encoder_name)
        self.text_encoder = text
        self.text_proj = nn.Linear(text.config.hidden_size, embed_dim)

        self.freeze_image = parse_freeze_image(freeze_image)
        self.freeze_text = parse_freeze_text(freeze_text)
        self.use_text_lora = False
        if self.freeze_image is True:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        elif self.freeze_image == "last_block":
            for name, p in self.image_encoder.named_parameters():
                p.requires_grad = name.startswith("layer4.")
        if self.freeze_text is True:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        elif self.freeze_text == "last_6_layers":
            for name, p in self.text_encoder.named_parameters():
                p.requires_grad = _text_last6_trainable(name)

        if learnable_tau:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / tau)))
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / tau)))

        self.train(False)

    def enable_text_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: list[str] | tuple[str, ...] = ("query", "value"),
    ) -> None:
        """Wrap BioClinicalBERT with LoRA; base BERT stays frozen, adapters + projs train."""
        from peft import LoraConfig, get_peft_model

        if self.use_text_lora:
            raise RuntimeError("text LoRA already enabled")
        if self.freeze_text == "last_6_layers":
            raise RuntimeError("cannot combine text LoRA with freeze_text=last_6_layers")
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=list(target_modules),
            bias="none",
        )
        self.text_encoder = get_peft_model(self.text_encoder, cfg)
        self.use_text_lora = True
        self.freeze_text = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(pixel_values)
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return F.normalize(self.text_proj(cls), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_emb = self.encode_image(pixel_values)
        text_emb = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return image_emb, text_emb, scale

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_image is True:
            self.image_encoder.eval()
        elif self.freeze_image == "last_block":
            self.image_encoder.eval()
            self.image_encoder.layer4.train(mode)
        if self.freeze_text is True and not self.use_text_lora:
            self.text_encoder.eval()
        elif self.freeze_text == "last_6_layers":
            self.text_encoder.eval()
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i >= 6:
                    layer.train(mode)
            if hasattr(self.text_encoder, "pooler") and self.text_encoder.pooler is not None:
                self.text_encoder.pooler.train(mode)
        return self

    def projection_parameters(self):
        params = list(self.image_proj.parameters()) + list(self.text_proj.parameters())
        if isinstance(self.logit_scale, nn.Parameter):
            params.append(self.logit_scale)
        return params

    def lora_parameters(self):
        if not self.use_text_lora:
            return []
        return [p for p in self.text_encoder.parameters() if p.requires_grad]

    def backbone_trainable_parameters(self):
        """Image + text encoder params with requires_grad (excludes projections)."""
        params = [p for p in self.image_encoder.parameters() if p.requires_grad]
        if self.use_text_lora:
            params.extend(self.lora_parameters())
        else:
            params.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
        return params

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
