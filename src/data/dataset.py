"""Manifest-backed CXR–report pairs (DICOM / Chambon JPG cache)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CheXbert / CheXpert 14 observations (join columns).
CHEXBERT_LABEL_KEYS = (
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
)


def parse_chexbert_labels_u_ignore(row: dict[str, str]) -> tuple[list[float], list[float]]:
    """Irvin U-Ignore: keep 0/1; mask uncertain (-1). Empty → 0 (not mentioned).

    Returns (labels[14], mask[14]) with mask 1 = contribute to loss/metrics.
    """
    labels: list[float] = []
    mask: list[float] = []
    for key in CHEXBERT_LABEL_KEYS:
        raw = (row.get(key) or "").strip()
        if raw == "" or raw.lower() in {"nan", "none"}:
            labels.append(0.0)
            mask.append(1.0)
            continue
        try:
            v = float(raw)
        except ValueError as e:
            raise ValueError(f"bad label {key}={raw!r}") from e
        if abs(v - (-1.0)) < 1e-6:
            labels.append(0.0)
            mask.append(0.0)
        elif abs(v - 1.0) < 1e-6:
            labels.append(1.0)
            mask.append(1.0)
        elif abs(v - 0.0) < 1e-6:
            labels.append(0.0)
            mask.append(1.0)
        else:
            raise ValueError(f"unexpected label {key}={raw!r}")
    return labels, mask


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_dicom_rgb(path: Path) -> Image.Image:
    """CheXpert Plus DICOM → RGB PIL via Chambon Appendix A (Plus paper).

    cv2.convertScaleAbs(255/max) → MONOCHROME1 invert → equalizeHist → RGB.
    """
    import cv2
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx <= 0:
        rescaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        rescaled = cv2.convertScaleAbs(arr, alpha=(255.0 / mx))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        rescaled = cv2.bitwise_not(rescaled)
    if rescaled.ndim == 3:
        rescaled = cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY) if rescaled.shape[-1] == 3 else rescaled[..., 0]
    adjusted = cv2.equalizeHist(rescaled)
    return Image.fromarray(adjusted, mode="L").convert("RGB")


def chambon_cache_jpg_path(cache_root: Path, path_to_dcm: str) -> Path:
    rel = path_to_dcm.strip().lstrip("/")
    if rel.lower().endswith(".dcm"):
        rel = rel[:-4] + ".jpg"
    return Path(cache_root) / rel


def load_manifest_rows(manifest_path: Path, patient_ids: set[str] | None = None) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if patient_ids is not None:
        rows = [r for r in rows if r["patient_id"] in patient_ids]
    rows.sort(key=lambda r: int(r["patient_id"]))
    return rows


def load_plus_tiny_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if any(r.get("our_split") != "train" for r in rows):
        raise ValueError("plus tiny manifest must be train-only")
    return rows


def load_plus_pair_rows(
    manifest_path: Path,
    allowed_splits: set[str] | None = None,
    *,
    allow_test: bool = False,
) -> list[dict[str, str]]:
    """Load Plus rows that already include report_text + absolute image_path.

    Refuses our_split=test unless allow_test=True (playbook Step 9 sealed eval).
    """
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if allowed_splits is None:
        allowed_splits = {"train", "val"}
    if "test" in allowed_splits and not allow_test:
        raise ValueError(
            "refusing to load test unless allow_test=True (playbook Step 9 sealed eval)"
        )
    bad = [r.get("our_split") for r in rows if r.get("our_split") not in allowed_splits]
    if bad:
        raise ValueError(f"unexpected our_split values (first): {bad[0]!r}; allowed={sorted(allowed_splits)}")
    for r in rows:
        if not (r.get("report_text") or "").strip():
            raise ValueError(f"empty report_text for {r.get('path_to_dcm')}")
        if not (r.get("image_path") or "").strip():
            raise ValueError(f"empty image_path for {r.get('path_to_dcm')}")
    return rows


def read_id_list(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids


class IUXrayPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: Path,
        tokenizer,
        max_text_length: int = 256,
    ) -> None:
        self.rows = rows
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.tfm = image_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        path = self.image_root / row["image_path"]
        image = Image.open(path).convert("RGB")
        pixel = self.tfm(image)
        encoded = self.tokenizer(
            row["report_text"],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "patient_id": row["patient_id"],
        }


class CheXpertPlusPairDataset(Dataset):
    """Plus pairs: Chambon JPG cache (required) or live Chambon DICOM convert."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer,
        max_text_length: int = 256,
        image_cache_root: Path | str | None = None,
        require_cache: bool = True,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.tfm = image_transform()
        self.image_cache_root = Path(image_cache_root) if image_cache_root else None
        self.require_cache = bool(require_cache)
        if self.require_cache and self.image_cache_root is None:
            raise ValueError("require_cache=True needs image_cache_root")
        if self.require_cache and self.image_cache_root is not None:
            missing = []
            for r in rows:
                jp = chambon_cache_jpg_path(self.image_cache_root, r["path_to_dcm"])
                if not jp.is_file():
                    missing.append(str(jp))
                    if len(missing) >= 5:
                        break
            if missing:
                raise FileNotFoundError(
                    f"Chambon JPG cache miss (showing up to 5): {missing}. "
                    "Run scripts/cache_plus_chambon_jpg.py first."
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        if self.image_cache_root is not None:
            jpg = chambon_cache_jpg_path(self.image_cache_root, row["path_to_dcm"])
            if jpg.is_file():
                image = Image.open(jpg).convert("RGB")
            elif self.require_cache:
                raise FileNotFoundError(jpg)
            else:
                path = Path(row["image_path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                image = load_dicom_rgb(path)
        else:
            path = Path(row["image_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            image = load_dicom_rgb(path)
        pixel = self.tfm(image)
        encoded = self.tokenizer(
            row["report_text"],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels, label_mask = parse_chexbert_labels_u_ignore(row)
        return {
            "pixel_values": pixel,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "patient_id": row.get("deid_patient_id") or row.get("pair_id") or "",
            "labels": labels,
            "label_mask": label_mask,
        }
