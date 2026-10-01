# Bio_ClinicalBERT (download locally)

Do not commit model weights to git.

```bash
# from repo root
huggingface-cli download emilyalsentzer/Bio_ClinicalBERT --local-dir models/Bio_ClinicalBERT
```

Configs use relative path: `text_encoder: models/Bio_ClinicalBERT`.
