# Processed manifests

Manifests contain radiology report text and must not be pushed to a public repository.

```bash
export RADPAIR_ROOT="$(pwd)"
export CHEXPERT_PLUS_ROOT=/path/to/chexpert_plus

python scripts/write_plus_split_manifest.py --split train --out data/processed/plus_train_manifest.csv
python scripts/write_plus_split_manifest.py --split val   --out data/processed/plus_val_manifest.csv
python scripts/join_plus_chexbert_and_stats.py
python scripts/build_plus_test_manifest.py
```

Expected sizes (15k subset, seed 42): train 10293 / val 1518 / test 3189.
