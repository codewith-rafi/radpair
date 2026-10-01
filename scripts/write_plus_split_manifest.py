"""Build a Plus split manifest with Findings+Impression joined (train or val only).

Writes under RADPAIR_ROOT. Train/val only.

Example:
  python scripts/write_plus_split_manifest.py --split val --out data/processed/plus_val_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import (
    assert_under_repo,
    chexpert_plus_root,
    default_chexbert_impression,
    default_plus_csv,
    repo_root,
    resolve_repo_path,
)
DEFAULT_SUBSET = resolve_repo_path("chexpert_plus_subset/manifest.csv")
DEFAULT_CSV = None  # resolved via CHEXPERT_PLUS_ROOT in main


def report_text(row: dict) -> str:
    findings = (row.get("section_findings") or "").strip()
    impression = (row.get("section_impression") or "").strip()
    return (findings + "\n" + impression).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset_manifest", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.csv is None:
        args.csv = default_plus_csv()

    out = args.out.resolve()
    assert_under_repo(out, "out")
    if args.split == "test":
        raise SystemExit("refusing test split; use build_plus_test_manifest.py + eval_plus_sealed_test.py for held-out eval")

    chosen: list[dict] = []
    with args.subset_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("our_split") == args.split:
                chosen.append(row)
    if not chosen:
        raise SystemExit(f"no rows for split={args.split}")

    need = {(r["path_to_dcm"].strip().lstrip("/")) for r in chosen}
    texts: dict[str, str] = {}
    with args.csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rel = (row.get("path_to_dcm") or "").strip().lstrip("/")
            if rel in need and rel not in texts:
                texts[rel] = report_text(row)
                if len(texts) == len(need):
                    break
    missing = need - set(texts)
    if missing:
        raise SystemExit(f"missing report text for {len(missing)} paths, e.g. {next(iter(missing))}")

    fields = [
        "pair_id",
        "deid_patient_id",
        "patient_report_date_order",
        "path_to_dcm",
        "path_to_image",
        "image_path",
        "report_text",
        "our_split",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(chosen):
            rel = r["path_to_dcm"].strip().lstrip("/")
            text = texts[rel]
            if not text:
                raise SystemExit(f"empty report_text for {rel}")
            w.writerow(
                {
                    "pair_id": f"{args.split}_{i:05d}",
                    "deid_patient_id": r["deid_patient_id"],
                    "patient_report_date_order": r["patient_report_date_order"],
                    "path_to_dcm": rel,
                    "path_to_image": r.get("path_to_image", ""),
                    "image_path": r["image_path"],
                    "report_text": text,
                    "our_split": args.split,
                }
            )
    print(f"wrote {len(chosen)} {args.split} pairs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
