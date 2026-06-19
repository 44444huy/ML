"""Metrics for the dog eye-color rare-event problem.

The positive rate is ~4%, so a model that predicts "all negative" would
score accuracy=0.96 and macro-F1=0.49 — both meaningless. We therefore
report:

    PR-AUC    — area under the precision-recall curve, the right
                 ranking metric when positives are rare.
    ROC-AUC   — for completeness / comparison with paper.
    F1 / precision / recall — at the supplied decision threshold.
    confusion_matrix         — for the report tables.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Unified metric block for the eye task.

    Args:
        y_true : (n,) int  in {0,1}
        y_pred : (n,) int  in {0,1}, hard labels after thresholding
        y_prob : (n,) float, P(blue=1)
    """
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr = float("nan")

    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pr_auc": pr,
        "roc_auc": roc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the PR-curve probability threshold that maximizes F1."""
    _, _, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Mean ± std across folds for scalar metrics."""
    keys = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
    out: dict = {}
    for k in keys:
        vals = np.asarray([r[k] for r in fold_results], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out
