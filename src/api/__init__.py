"""Public embed API. Callers supply tensors only (no split loading here)."""

from src.api.embed import embed_image, embed_text, load_dual_encoder

__all__ = ["embed_image", "embed_text", "load_dual_encoder"]
