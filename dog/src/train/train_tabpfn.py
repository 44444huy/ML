"""TabPFN baseline for dog eye-color prediction.

What is TabPFN?
    TabPFN (Hollmann et al., 2023, "TabPFN: A Transformer That Solves
    Small Tabular Classification Problems in a Second") is a pre-trained
    Transformer for tabular data. It was meta-trained on millions of
    synthetic tabular tasks, so at inference time it does *in-context
    learning*: you pass (X_train, y_train, X_test) into a single forward
    pass and it returns predictions — no gradient updates on your data.

Why try it here?
    Two beginner-friendly properties matter for this project:
      1. Zero hyper-parameter tuning. Same default config for every task.
      2. It is a strong baseline on small tabular problems (n < ~10k).
    Our dataset is exactly that: ~2,769 dogs × 52 SNPs, severely imbalanced.

Imbalance handling:
    TabPFN does not expose a `pos_weight` knob. We apply two common
    fixes used in the TabPFN paper's downstream evaluation:
      - Standardize features (helps the in-context attention)
      - Threshold-tune on validation (instead of fixed 0.5) — picks the
        threshold that maximises F1, since predict_proba is the only
        knob we have.

Output:
    experiments/eye/tabpfn_results.json — same schema as mlp_results.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

# Allow CPU inference on >1000 samples (default limit). Tabpfn paper
# trains on up to 1024 in-context examples; we have ~1772 per fold,
# which is slower but still works.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODEL_CACHE_DIR = ROOT / "data" / "processed" / "model_cache"
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODEL_CACHE_DIR / "tabpfn"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import evaluate, aggregate_folds  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXP_DIR / "tabpfn_results.json"

SEED = 42


def standardize(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


def best_f1_threshold(y_true, prob) -> float:
    """Pick the probability threshold that maximises F1 on validation."""
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


def fit_predict(X_tr, y_tr, X_te, device):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device=device, random_state=SEED,
                           ignore_pretraining_limits=True)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return prob


def run(X: np.ndarray, y: np.ndarray, splits: dict, device: str) -> dict:
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        prob = fit_predict(X_tr, y[tr], X_va, device)
        # Tune threshold on validation (TabPFN has no pos_weight)
        t = best_f1_threshold(y[va], prob)
        pred = (prob >= t).astype(int)
        res = evaluate(y[va], pred, prob)
        res["threshold"] = t
        res["fold"] = i
        fold_results.append(res)
        print(f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
              f"F1={res['f1']:.3f} (t={t:.2f})")

    cv_mean = aggregate_folds(fold_results)

    # Refit on full trainval, score test
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    # Use a 90/10 internal split to pick the threshold for test
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]
    prob_va = fit_predict(X_tv[idx_tr], y[tv][idx_tr], X_tv[idx_va], device)
    t_test = best_f1_threshold(y[tv][idx_va], prob_va)

    prob_te = fit_predict(X_tv, y[tv], X_te, device)
    pred_te = (prob_te >= t_test).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = t_test
    print(f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
          f"F1={test_res['f1']:.3f} (t={t_test:.2f})")

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_tabpfn] device = {device}")

    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    print("\n=== TabPFN (pretrained transformer, in-context learning) ===")
    out = run(X, y, splits, device)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[train_tabpfn] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
