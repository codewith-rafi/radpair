# Embed API

Thin wrappers around the dual encoder. Embeddings are **L2-normalised**, `d=256`.

## Default checkpoint

`experiments/plus_g_peft_001/checkpoint.pt` (PEFT LoRA r=8 on BioClinicalBERT query+value; ResNet frozen). Override with any DualEncoder checkpoint path.

## Usage

```python
from transformers import AutoTokenizer
from src.api import load_dual_encoder, embed_image, embed_text

device = "cuda:0"  # after setting CUDA_VISIBLE_DEVICES if needed
model = load_dual_encoder(device=device)  # default ckpt = plus_g_peft_001
tok = AutoTokenizer.from_pretrained("models/Bio_ClinicalBERT", local_files_only=True)

# image: (B, 3, 224, 224) ImageNet-normalised float tensor on same device
img_emb = embed_image(model, pixel_values)   # (B, 256), L2

# text: tokenized Findings+Impression (or any report string)
batch = tok(["Findings ...\nImpression ..."], padding=True, truncation=True, max_length=256, return_tensors="pt")
txt_emb = embed_text(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
```

## Notes

- Does **not** load manifests or splits; callers own preprocessing (Chambon JPG → 224 + ImageNet norm for images).
- PEFT LoRA is enabled automatically when the checkpoint contains `lora_*` keys.
- Returns `DualEncoder` in `eval()` mode; no gradients by default (`@torch.no_grad` on embed_*).
