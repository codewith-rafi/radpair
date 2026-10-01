#!/usr/bin/env python3
"""Join CheXbert impression labels + persist report_text for Plus subset / manifests.

Writes only under RADPAIR_ROOT.  Does not load test for training.
u_policy=ignore is recorded in preprocess_stats (Irvin U-Ignore); G1 does not use labels.

Example:
  python scripts/join_plus_chexbert_and_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (
    assert_under_repo,
    chexpert_plus_root,
    default_chexbert_impression,
    default_plus_csv,
    repo_root,
    resolve_repo_path,
)

from src.data.dataset import CHEXBERT_LABEL_KEYS  # noqa: E402

DEFAULT_SUBSET = resolve_repo_path("chexpert_plus_subset/manifest.csv")
DEFAULT_CSV = None  # resolved via CHEXPERT_PLUS_ROOT in main
DEFAULT_CHEXBERT = None  # resolved via CHEXPERT_PLUS_ROOT in main
DEFAULT_TRAIN = ROOT / "data" / "processed" / "plus_train_manifest.csv"
DEFAULT_VAL = ROOT / "data" / "processed" / "plus_val_manifest.csv"
OUT_DIR = resolve_repo_path("chexpert_plus_subset")


def report_text_from_csv_row(row: dict) -> str:
    findings = (row.get("section_findings") or "").strip()
    impression = (row.get("section_impression") or "").strip()
    return (findings + "\n" + impression).strip()


def load_chexbert_index(path: Path) -> dict[str, dict]:
    """JSONL keyed by path_to_image (relative .jpg)."""
    idx: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj.get("path_to_image") or "").strip().lstrip("/")
            if key:
                idx[key] = obj
    return idx


def label_fields(obj: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in CHEXBERT_LABEL_KEYS:
        if obj is None or k not in obj or obj[k] is None:
            out[k] = ""
        else:
            out[k] = str(obj[k])
    return out


def main() -> int:

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset_manifest", type=Path, default=DEFAULT_SUBSET)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--chexbert", type=Path, default=DEFAULT_CHEXBERT)
    p.add_argument("--train_manifest", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val_manifest", type=Path, default=DEFAULT_VAL)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    if args.csv is None:
        args.csv = default_plus_csv()
    if args.chexbert is None:
        args.chexbert = default_chexbert_impression()

    out_dir = args.out_dir.resolve()
    assert_under_repo(out_dir, "out_dir")

    print("loading CheXbert impression index…", flush=True)
    chex = load_chexbert_index(args.chexbert)
    print(f"chexbert rows={len(chex)}", flush=True)

    subset_rows = list(csv.DictReader(args.subset_manifest.open(newline="", encoding="utf-8")))
    need_dcm = {(r["path_to_dcm"].strip().lstrip("/")) for r in subset_rows}
    texts: dict[str, str] = {}
    print("joining report_text from Plus CSV…", flush=True)
    with args.csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rel = (row.get("path_to_dcm") or "").strip().lstrip("/")
            if rel in need_dcm and rel not in texts:
                texts[rel] = report_text_from_csv_row(row)
                if len(texts) == len(need_dcm):
                    break
    missing_text = need_dcm - set(texts)
    if missing_text:
        raise SystemExit(f"missing report_text for {len(missing_text)} dcm paths")

    n_label_miss = 0
    enriched: list[dict] = []
    for r in subset_rows:
        rel_dcm = r["path_to_dcm"].strip().lstrip("/")
        rel_img = (r.get("path_to_image") or "").strip().lstrip("/")
        obj = chex.get(rel_img)
        if obj is None:
            n_label_miss += 1
            labs = label_fields(None)
        else:
            labs = label_fields(obj)
        enriched.append(
            {
                **{k: r.get(k, "") for k in r},
                "path_to_dcm": rel_dcm,
                "path_to_image": rel_img,
                "report_text": texts[rel_dcm],
                **labs,
            }
        )

    base_fields = [
        "deid_patient_id",
        "patient_report_date_order",
        "path_to_dcm",
        "path_to_image",
        "image_path",
        "frontal_lateral",
        "ap_pa",
        "official_split",
        "our_split",
        "report_text",
    ]
    fields = base_fields + list(CHEXBERT_LABEL_KEYS)
    pairs_path = out_dir / "pairs_with_text_labels.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in enriched:
            w.writerow(row)
    print(f"wrote {pairs_path} n={len(enriched)} label_miss={n_label_miss}", flush=True)

    # Rewrite train/val processed manifests with labels + report_text.
    by_split = {"train": [], "val": []}
    for row in enriched:
        sp = row.get("our_split")
        if sp in by_split:
            by_split[sp].append(row)

    def write_split(split: str, out: Path) -> None:
        rows = by_split[split]
        out_fields = [
            "pair_id",
            "deid_patient_id",
            "patient_report_date_order",
            "path_to_dcm",
            "path_to_image",
            "image_path",
            "report_text",
            "our_split",
        ] + list(CHEXBERT_LABEL_KEYS)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(rows):
                w.writerow(
                    {
                        **r,
                        "pair_id": f"{split}_{i:05d}",
                        "our_split": split,
                    }
                )
        print(f"wrote {out} n={len(rows)}")

    write_split("train", args.train_manifest.resolve())
    write_split("val", args.val_manifest.resolve())

    # Duplicate / short-text stats (train/val only).
    def dup_stats(rows: list[dict]) -> dict:
        texts_list = [r["report_text"] for r in rows]
        c = Counter(texts_list)
        n_dup_extra = sum(v - 1 for v in c.values() if v > 1)
        n_short = sum(1 for t in texts_list if len(t) < 50)
        return {
            "n": len(rows),
            "n_unique_report_text": len(c),
            "n_dup_report_text_extra": n_dup_extra,
            "n_report_text_lt_50_chars": n_short,
        }

    stats = {
        "u_policy": "ignore",
        "u_policy_note": "Irvin U-Ignore: mask uncertain labels in disease loss/metrics when used",
        "chexbert_source": str(args.chexbert),
        "chexbert_section": "impression_fixed",
        "n_subset": len(enriched),
        "n_chexbert_join_miss": n_label_miss,
        "train": dup_stats(by_split["train"]),
        "val": dup_stats(by_split["val"]),
        "dicom_preprocess": "chambon_appendix_a",
        "text_field": "findings_plus_impression",
        "text_format": "raw findings\\nimpression (no IU Findings:/Impression: prefixes)",
    }
    stats_path = out_dir / "preprocess_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
