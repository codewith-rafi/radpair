#!/usr/bin/env python3
"""Build Chambon Appendix A JPG cache for Plus train+val (CPU; write under RADPAIR_ROOT).

Uses cv2 (opencv-python-headless). Never write into CHEXPERT_PLUS_ROOT; use tmux for long builds.

Example:
  # activate your conda/venv
  cd $RADPAIR_ROOT  # repo root
  python scripts/cache_plus_chambon_jpg.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

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

from src.data.dataset import chambon_cache_jpg_path, load_dicom_rgb  # noqa: E402

DEFAULT_TRAIN = ROOT / "data" / "processed" / "plus_train_manifest.csv"
DEFAULT_VAL = ROOT / "data" / "processed" / "plus_val_manifest.csv"
DEFAULT_CACHE = resolve_repo_path("chexpert_plus_subset/images_chambon")


def _convert_one(args: tuple[str, str, str, bool]) -> tuple[str, str | None]:
    """Worker: (dcm_abs, jpg_abs, path_to_dcm, force) -> (path_to_dcm, error_or_None)."""
    dcm_abs, jpg_abs, rel, force = args
    try:
        jpg = Path(jpg_abs)
        if (not force) and jpg.is_file() and jpg.stat().st_size > 0:
            return rel, None
        img = load_dicom_rgb(Path(dcm_abs))
        arr = np.asarray(img.convert("L"))
        jpg.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(jpg), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return rel, f"cv2.imwrite failed: {jpg}"
        return rel, None
    except Exception as e:  # noqa: BLE001 — collect all decode failures
        return rel, f"{type(e).__name__}: {e}"


def main() -> int:

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_manifest", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val_manifest", type=Path, default=DEFAULT_VAL)
    p.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JPGs (required after switching PIL→cv2 convert).",
    )
    args = p.parse_args()

    cache_root = args.cache_root.resolve()
    assert_under_repo(cache_root, "cache_root")
    cache_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str, bool]] = []
    seen: set[str] = set()
    for man in (args.train_manifest, args.val_manifest):
        with man.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = row["path_to_dcm"].strip().lstrip("/")
                if rel in seen:
                    continue
                seen.add(rel)
                dcm = Path(row["image_path"])
                jpg = chambon_cache_jpg_path(cache_root, rel)
                jobs.append((str(dcm), str(jpg), rel, bool(args.force)))

    fails: list[dict[str, str]] = []
    n_ok = 0
    n_done = 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_convert_one, j) for j in jobs]
        for fut in as_completed(futs):
            rel, err = fut.result()
            n_done += 1
            if err:
                fails.append({"path_to_dcm": rel, "error": err})
            else:
                n_ok += 1
            if n_done % 500 == 0 or n_done == len(jobs):
                print(f"cache {n_done}/{len(jobs)} ok={n_ok} fail={len(fails)}", flush=True)

    report = {
        "cache_root": str(cache_root),
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_fail": len(fails),
        "jpeg_quality": 95,
        "force": bool(args.force),
        "dicom_preprocess": "chambon_appendix_a",
        "backend": "cv2",
        "fails": fails[:50],
    }
    report_path = cache_root / "cache_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_jobs", "n_ok", "n_fail", "force", "cache_root")}, indent=2))
    print(f"wrote {report_path}")
    if fails:
        raise SystemExit(f"cache failed for {len(fails)} files; see {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
