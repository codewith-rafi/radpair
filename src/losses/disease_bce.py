"""Masked multi-label BCE and AUROC / macro-F1 under U-Ignore."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """logits/labels/mask: (B, C). Mean over masked elements; 0 if none."""
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (loss * m).sum() / denom


@torch.no_grad()
def multilabel_auroc_f1(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Per-label AUROC (skip if <2 classes in masked rows) + macro-F1.

    probs/labels/mask: (N, C) numpy.
    """
    n_labels = probs.shape[1]
    aucs = []
    per_label = {}
    y_true_flat = []
    y_pred_flat = []
    for c in range(n_labels):
        m = mask[:, c] > 0.5
        if m.sum() < 2:
            per_label[c] = {"auroc": None, "n": int(m.sum())}
            continue
        y = labels[m, c]
        p = probs[m, c]
        if len(np.unique(y)) < 2:
            per_label[c] = {"auroc": None, "n": int(m.sum()), "note": "single_class"}
            continue
        auc = float(roc_auc_score(y, p))
        aucs.append(auc)
        per_label[c] = {"auroc": auc, "n": int(m.sum())}
        y_true_flat.append(y)
        y_pred_flat.append((p >= threshold).astype(np.float32))

    macro_auroc = float(np.mean(aucs)) if aucs else float("nan")
    if y_true_flat:
        yt = np.concatenate(y_true_flat)
        yp = np.concatenate(y_pred_flat)
        macro_f1 = float(f1_score(yt, yp, zero_division=0))
        # Also class-wise macro over labels with both classes:
        f1s = []
        for c in range(n_labels):
            m = mask[:, c] > 0.5
            if m.sum() < 1:
                continue
            y = labels[m, c]
            pred = (probs[m, c] >= threshold).astype(np.float32)
            if len(np.unique(y)) < 2 and y.sum() == 0 and pred.sum() == 0:
                continue
            f1s.append(float(f1_score(y, pred, zero_division=0)))
        macro_f1_per_label = float(np.mean(f1s)) if f1s else float("nan")
    else:
        macro_f1 = float("nan")
        macro_f1_per_label = float("nan")

    return {
        "macro_auroc": macro_auroc,
        "macro_f1_thresholded_pooled": macro_f1,
        "macro_f1_per_label": macro_f1_per_label,
        "n_labels_with_auroc": len(aucs),
        "per_label": per_label,
    }
