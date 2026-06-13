"""TabNet for dog eye-color prediction.

What is TabNet?
    TabNet (Arik & Pfister, 2021, "TabNet: Attentive Interpretable
    Tabular Learning") is a neural architecture for tabular data that
    uses **sequential attention** to choose which features to look at
    at each decision step. This gives both:
      - competitive performance vs gradient boosting on tabular tasks,
      - built-in feature importance (which SNPs the model attends to).

Why try it here?
    Our MLP and the classical baselines treat all 52 SNPs equally. TabNet
    can learn to focus on the chr18 / ALX4 SNPs and ignore weaker ones,
    which is conceptually a better fit for this oligogenic trait.

Imbalance handling:
    pytorch-tabnet exposes `weights={0: w0, 1: w1}` for the
    cross-entropy loss. We mirror the MLP's `pos_weight = n_neg/n_pos`.

Output:
    experiments/eye/tabnet_results.json — same schema as mlp_results.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import evaluate, aggregate_folds  # noqa: E402

warnings.filterwarnings("ignore")

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXP_DIR / "tabnet_results.json"

SEED = 42


def standardize(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


def fit_predict(X_tr, y_tr, X_va, y_va, X_te, device):
    """Fit TabNet on (X_tr,y_tr) with early stopping on (X_va,y_va);
    return prob on X_te.
    """
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    # Inverse-frequency weights — matches MLP's pos_weight idea.
    weights = {0: 1.0, 1: float(n_neg) / max(n_pos, 1)}

    clf = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3,
        gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=20, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=SEED,
        device_name=device,
        verbose=0,
    )
    clf.fit(
        X_train=X_tr, y_train=y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=["auc"],
        max_epochs=200,
        patience=25,
        batch_size=256,
        virtual_batch_size=64,
        weights=weights,
    )
    return clf.predict_proba(X_te)[:, 1]


def run(X: np.ndarray, y: np.ndarray, splits: dict, device: str) -> dict:
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        prob = fit_predict(X_tr, y[tr], X_va, y[va], X_va, device)
        pred = (prob >= 0.5).astype(int)
        res = evaluate(y[va], pred, prob)
        res["fold"] = i
        fold_results.append(res)
        print(f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
              f"F1={res['f1']:.3f}")

    cv_mean = aggregate_folds(fold_results)

    # Refit on full trainval w/ internal 90/10 val split, score test
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]
    prob_te = fit_predict(
        X_tv[idx_tr], y[tv][idx_tr],
        X_tv[idx_va], y[tv][idx_va],
        X_te, device,
    )
    pred_te = (prob_te >= 0.5).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    print(f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
          f"F1={test_res['f1']:.3f}")

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_tabnet] device = {device}")

    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    print("\n=== TabNet (attentive tabular learning) ===")
    out = run(X, y, splits, device)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[train_tabnet] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
