"""Build Plus train/val manifests with report_text = section_findings only (ablation A1).

Reads existing processed manifests (same pairs/labels/paths). Joins findings from
CheXpert Plus CSV via CHEXPERT_PLUS_ROOT (read-only). Writes under
RADPAIR_ROOT/data/processed/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
DEFAULT_CSV = None  # resolved via CHEXPERT_PLUS_ROOT in main
DEFAULT_TRAIN = ROOT / "data" / "processed" / "plus_train_manifest.csv"
DEFAULT_VAL = ROOT / "data" / "processed" / "plus_val_manifest.csv"
OUT_TRAIN = ROOT / "data" / "processed" / "plus_train_findings_only_manifest.csv"
OUT_VAL = ROOT / "data" / "processed" / "plus_val_findings_only_manifest.csv"


def load_findings_index(csv_path: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if "path_to_dcm" not in (reader.fieldnames or []):
            raise SystemExit(f"Plus CSV missing path_to_dcm: {csv_path}")
        if "section_findings" not in (reader.fieldnames or []):
            raise SystemExit(f"Plus CSV missing section_findings: {csv_path}")
        for row in reader:
            key = (row.get("path_to_dcm") or "").strip()
            if not key:
                continue
            idx[key] = (row.get("section_findings") or "").strip()
    return idx


def rewrite_manifest(
    src: Path,
    dst: Path,
    findings: dict[str, str],
) -> dict:
    with src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty manifest: {src}")
    fieldnames = list(rows[0].keys())
    if "report_text" not in fieldnames or "path_to_dcm" not in fieldnames:
        raise SystemExit(f"manifest needs path_to_dcm + report_text: {src}")

    out_rows: list[dict] = []
    n_empty = 0
    n_missing_key = 0
    for r in rows:
        key = (r.get("path_to_dcm") or "").strip()
        text = findings.get(key)
        if text is None:
            n_missing_key += 1
            continue
        if not text:
            n_empty += 1
            continue
        nr = dict(r)
        nr["report_text"] = text
        out_rows.append(nr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    return {
        "src": str(src),
        "dst": str(dst),
        "n_src": len(rows),
        "n_out": len(out_rows),
        "n_dropped_empty_findings": n_empty,
        "n_dropped_missing_csv_key": n_missing_key,
    }


def main() -> None:

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--train_manifest", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val_manifest", type=Path, default=DEFAULT_VAL)
    p.add_argument("--out_train", type=Path, default=OUT_TRAIN)
    p.add_argument("--out_val", type=Path, default=OUT_VAL)
    p.add_argument("--expected_n_train", type=int, default=10293)
    p.add_argument("--expected_n_val", type=int, default=1518)
    args = p.parse_args()
    if args.csv is None:
        args.csv = default_plus_csv()

    for path in (args.out_train, args.out_val):
        assert_under_repo(path, "output")

    print(f"indexing findings from {args.csv} …", flush=True)
    findings = load_findings_index(args.csv)
    print(f"findings_keys={len(findings)}", flush=True)

    tr = rewrite_manifest(args.train_manifest, args.out_train, findings)
    va = rewrite_manifest(args.val_manifest, args.out_val, findings)
    print(json.dumps({"train": tr, "val": va}, indent=2), flush=True)

    if tr["n_out"] != args.expected_n_train:
        print(
            f"WARN train n_out={tr['n_out']} expected={args.expected_n_train} "
            f"(dropped empty/missing findings)",
            flush=True,
        )
    if va["n_out"] != args.expected_n_val:
        print(
            f"WARN val n_out={va['n_out']} expected={args.expected_n_val} "
            f"(dropped empty/missing findings)",
            flush=True,
        )
    print("FINDINGS_ONLY_MANIFEST_EXIT:0", flush=True)


if __name__ == "__main__":
    main()
