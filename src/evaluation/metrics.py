"""Unified metric suite for EVC evaluation.

All three traits are reported with:
    - accuracy, macro-F1, macro AUC (OvR, where feasible)
    - Expected Calibration Error (ECE)
    - for ordinal traits (hair/skin): MAE and Quadratic Weighted Kappa (QWK)

Handles classes absent from a split gracefully (AUC falls back to NaN).
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             cohen_kappa_score, mean_absolute_error,
                             confusion_matrix)

ORDINAL_TRAITS = {"hair", "skin"}


def _safe_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Macro-OvR AUC; returns NaN if fewer than 2 classes are present."""
    present = np.unique(y_true)
    if len(present) < 2 or probs is None:
        return float("nan")
    try:
        return float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro",
            labels=np.arange(n_classes)))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins on top-1 probability."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate(trait: str, y_true: np.ndarray, y_pred: np.ndarray,
             probs: np.ndarray | None, n_classes: int) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=np.arange(n_classes), zero_division=0)),
        "auc_macro_ovr": _safe_auc(y_true, probs, n_classes),
    }
    if probs is not None:
        out["ece"] = expected_calibration_error(y_true, probs)
    if trait in ORDINAL_TRAITS:
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
        out["qwk"] = float(cohen_kappa_score(
            y_true, y_pred, weights="quadratic",
            labels=np.arange(n_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    out["confusion_matrix"] = cm.tolist()
    return out


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Mean ± std across folds for each scalar metric."""
    if not fold_results:
        return {}
    keys = [k for k, v in fold_results[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in fold_results], dtype=float)
        agg[k] = {"mean": float(np.nanmean(vals)),
                  "std": float(np.nanstd(vals))}
    return agg
