"""Portable path helpers for this repository export.

RADPAIR_ROOT: repo root (directory containing scripts/, configs/, src/).
CHEXPERT_PLUS_ROOT: licensed CheXpert Plus dump (DICOM + source CSV); required
only for data-build scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("RADPAIR_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(maybe: str | Path) -> Path:
    p = Path(maybe)
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()


def assert_under_repo(path: Path | str, label: str = "path") -> Path:
    """Require writes/caches to live under the export repo."""
    root = repo_root().resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            f"{label} must be under RADPAIR_ROOT ({root}), got {resolved}"
        ) from exc
    return resolved


def chexpert_plus_root() -> Path:
    env = os.environ.get("CHEXPERT_PLUS_ROOT", "").strip()
    if not env:
        raise SystemExit(
            "Set CHEXPERT_PLUS_ROOT to your licensed CheXpert Plus directory "
            "(contains df_chexpert_plus_*.csv and DICOM paths)."
        )
    root = Path(env).resolve()
    if not root.is_dir():
        raise SystemExit(f"CHEXPERT_PLUS_ROOT is not a directory: {root}")
    return root


def default_plus_csv() -> Path:
    return chexpert_plus_root() / "df_chexpert_plus_240401.csv"


def default_chexbert_impression() -> Path:
    return chexpert_plus_root() / "chexbert_labels" / "impression_fixed.json"
