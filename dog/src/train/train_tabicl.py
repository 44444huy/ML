"""TabICL baseline for dog eye-color prediction.

What is TabICL?
    TabICL is a tabular foundation model. Like TabPFN, it performs
    in-context learning: fit() stores the labeled training context, and
    predict_proba() predicts test rows with a pretrained Transformer
    without updating model weights on our dog dataset.

Why try it here?
    It is a natural foundation-model comparison to TabPFN. TabICL is
    designed around tabular structure: column-wise embeddings, row-wise
    interactions, then dataset-wise in-context learning.

Imbalance handling:
    TabICL does not expose a BCE-style pos_weight knob. We therefore
    mirror the TabPFN recipe:
      - standardize features using train statistics only
      - threshold-tune on validation to maximize F1

Output:
    experiments/eye/tabicl_results.json -- same schema as mlp_results.json

Usage:
    python dog/src/train/train_tabicl.py
    python dog/src/train/train_tabicl.py --n_estimators 4  # faster, less robust
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import aggregate_folds, evaluate  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXP_DIR / "tabicl_results.json"

SEED = 42


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Pick the probability threshold that maximizes F1 on validation."""
    from sklearn.metrics import f1_score

    thresholds = np.unique(np.concatenate([[0.5], np.linspace(0.01, 0.99, 99)]))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def _positive_probability(clf, X: np.ndarray) -> np.ndarray:
    """Return P(class=1), robust to class column ordering."""
    probs = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pos_cols = np.where(classes == 1)[0]
    if len(pos_cols) == 0:
        return np.zeros(len(X), dtype=np.float32)
    return probs[:, pos_cols[0]]


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    device: str | None,
    n_estimators: int,
    batch_size: int,
    kv_cache: bool,
    allow_auto_download: bool,
) -> np.ndarray:
    """Fit TabICL on the context and return P(blue=1) for X_te."""
    try:
        from tabicl import TabICLClassifier
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'tabicl'. Install dog requirements with "
            "`pip install -r dog/requirements.txt`."
        ) from e

    clf = TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=batch_size,
        kv_cache=kv_cache,
        allow_auto_download=allow_auto_download,
        device=device,
        random_state=SEED,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    return _positive_probability(clf, X_te)


def run(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    *,
    device: str | None,
    n_estimators: int,
    batch_size: int,
    kv_cache: bool,
    allow_auto_download: bool,
) -> dict:
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        prob = fit_predict(
            X_tr,
            y[tr],
            X_va,
            device=device,
            n_estimators=n_estimators,
            batch_size=batch_size,
            kv_cache=kv_cache,
            allow_auto_download=allow_auto_download,
        )
        t = best_f1_threshold(y[va], prob)
        pred = (prob >= t).astype(int)
        res = evaluate(y[va], pred, prob)
        res["threshold"] = t
        res["fold"] = i
        fold_results.append(res)
        print(
            f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
            f"F1={res['f1']:.3f} (t={t:.2f})"
        )

    cv_mean = aggregate_folds(fold_results)

    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    # Use a 90/10 internal split to pick the test threshold, matching TabPFN.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    prob_va = fit_predict(
        X_tv[idx_tr],
        y[tv][idx_tr],
        X_tv[idx_va],
        device=device,
        n_estimators=n_estimators,
        batch_size=batch_size,
        kv_cache=kv_cache,
        allow_auto_download=allow_auto_download,
    )
    t_test = best_f1_threshold(y[tv][idx_va], prob_va)

    prob_te = fit_predict(
        X_tv,
        y[tv],
        X_te,
        device=device,
        n_estimators=n_estimators,
        batch_size=batch_size,
        kv_cache=kv_cache,
        allow_auto_download=allow_auto_download,
    )
    pred_te = (prob_te >= t_test).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = t_test
    print(
        f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
        f"F1={test_res['f1']:.3f} (t={t_test:.2f})"
    )

    return {
        "cv": {"per_fold": fold_results, "cv_mean": cv_mean},
        "test": test_res,
        "config": {
            "n_estimators": n_estimators,
            "batch_size": batch_size,
            "kv_cache": kv_cache,
            "allow_auto_download": allow_auto_download,
            "device": device,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="TabICL device; default auto-selects.")
    parser.add_argument("--n_estimators", type=int, default=8,
                        help="TabICL ensemble size; larger is stronger but slower.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of ensemble members processed together.")
    parser.add_argument("--kv_cache", action="store_true",
                        help="Cache training context projections for repeated inference.")
    parser.add_argument("--no_auto_download", action="store_true",
                        help="Disable checkpoint download; requires a local cached checkpoint.")
    args = parser.parse_args()

    print(f"[train_tabicl] device = {args.device or 'auto'}")
    print(f"[train_tabicl] n_estimators = {args.n_estimators}")

    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    print("\n=== TabICL (pretrained tabular foundation model, ICL) ===")
    out = run(
        X,
        y,
        splits,
        device=args.device,
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache=args.kv_cache,
        allow_auto_download=not args.no_auto_download,
    )
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[train_tabicl] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
