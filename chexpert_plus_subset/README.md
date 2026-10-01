# CheXpert Plus controlled subset — metadata only

This package does **not** ship patient images or report-text CSVs.

- **Licensed source:** [CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1); set `CHEXPERT_PLUS_ROOT` to your local dump
  (`df_chexpert_plus_240401.csv`, DICOM via `path_to_dcm`, CheXbert labels).
- **subset_n:** 15000 | **seed:** 42 | **split:** patient-level 70/10/20
- **Counts:** train 10293 / val 1518 / test 3189 pairs
- **View:** one frontal per study
- **Text:** Findings + Impression (newline join)
- **u_policy:** `ignore` (Irvin U-Ignore)
- **DICOM preprocess:** Chambon Appendix A (`cv2`) → JPG cache at
  `chexpert_plus_subset/images_chambon/`

Files here: `README.md`, `subset_summary.json`, `preprocess_stats.json`, `cache_report.json`.

Rebuild under the repo root (`RADPAIR_ROOT`):

1. Manifests via `scripts/write_plus_split_manifest.py` and `scripts/join_plus_chexbert_and_stats.py`
2. `python scripts/cache_plus_chambon_jpg.py` → `chexpert_plus_subset/images_chambon/`
