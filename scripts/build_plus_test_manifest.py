"""Build sealed Plus test manifest (Findings+Impression) from subset pairs CSV.

Filters our_split=test from chexpert_plus_subset/pairs_with_text_labels.csv.
Writes under RADPAIR_ROOT/data/processed/. 
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_PAIRS = resolve_repo_path("chexpert_plus_subset/pairs_with_text_labels.csv")
OUT_TEST = ROOT / "data" / "processed" / "plus_test_manifest.csv"
LABEL_KEYS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def main() -> None:

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--out", type=Path, default=OUT_TEST)
    p.add_argument("--expected_n_test", type=int, default=3189)
    args = p.parse_args()

    out = assert_under_repo(args.out.resolve(), "out")

    fieldnames = [
        "pair_id",
        "deid_patient_id",
        "patient_report_date_order",
        "path_to_dcm",
        "path_to_image",
        "image_path",
        "report_text",
        "our_split",
        *LABEL_KEYS,
    ]
    rows_out: list[dict] = []
    n_empty = 0
    with args.pairs.open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if (r.get("our_split") or "").strip() != "test":
                continue
            text = (r.get("report_text") or "").strip()
            if not text:
                n_empty += 1
                continue
            pid = (r.get("deid_patient_id") or "").strip()
            order = (r.get("patient_report_date_order") or "").strip()
            rows_out.append(
                {
                    "pair_id": f"test_{len(rows_out):05d}",
                    "deid_patient_id": pid,
                    "patient_report_date_order": order,
                    "path_to_dcm": (r.get("path_to_dcm") or "").strip(),
                    "path_to_image": (r.get("path_to_image") or "").strip(),
                    "image_path": (r.get("image_path") or "").strip(),
                    "report_text": text,
                    "our_split": "test",
                    **{k: (r.get(k) or "") for k in LABEL_KEYS},
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "pairs_csv": str(args.pairs),
        "out": str(out),
        "n_test": len(rows_out),
        "n_empty_report_dropped": n_empty,
        "n_unique_patients": len({r["deid_patient_id"] for r in rows_out}),
        "expected_n_test": args.expected_n_test,
        "text": "Findings+Impression (from pairs_with_text_labels)",
        "note": "Sealed Step 9 only. Do not use for tuning.",
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.expected_n_test and len(rows_out) != args.expected_n_test:
        print(
            f"WARN n_test={len(rows_out)} expected={args.expected_n_test}",
            flush=True,
        )
    print("PLUS_TEST_MANIFEST_EXIT:0", flush=True)


if __name__ == "__main__":
    main()
