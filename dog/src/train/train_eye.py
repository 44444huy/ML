"""Train an MLP for dog eye-color prediction.

Method:
    BCE-with-logits + class_weight (`pos_weight = n_neg / n_pos`).
    This is the standard fix for class imbalance: scale up the loss on
    the rare positive class so the model can't just learn to predict
    "always negative".

Pipeline:
    - Load the npz bundle and persistent splits (built by data/splits.py).
    - 5-fold CV on trainval, with early stopping on validation PR-AUC.
    - Refit on full trainval and score the held-out test set.
    - Save a JSON identical in layout to baseline_results.json.

Usage:
    python dog/src/train/train_eye.py
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import evaluate, aggregate_folds  # noqa: E402
from models.mlp import MLPBinary  # noqa: E402

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Hparams:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    patience: int = 25
    hidden: int = 128
    n_layers: int = 2
    dropout: float = 0.3


HP = Hparams()
SEED = 42


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t)
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def pr_auc_quick(prob, y):
    from sklearn.metrics import average_precision_score
    if y.sum() == 0:
        return 0.0
    return float(average_precision_score(y, prob))


def train_one(X_tr, y_tr, X_va, y_va, device) -> nn.Module:
    seed_all(SEED)
    model = MLPBinary(in_dim=X_tr.shape[1], hidden=HP.hidden,
                      n_layers=HP.n_layers, dropout=HP.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)

    # Inverse-frequency pos_weight for BCE
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    best_pr = -1.0
    best_state = None
    patience_left = HP.patience

    n = len(X_tr_t)
    for epoch in range(HP.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, HP.batch_size):
            idx = perm[s:s + HP.batch_size]
            opt.zero_grad()
            logits = model(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        pr = pr_auc_quick(va_prob, y_va)

        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = HP.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(X: np.ndarray, y: np.ndarray, splits: dict, device) -> dict:
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        model = train_one(X_tr, y[tr], X_va, y[va], device)
        prob, pred = predict(model, X_va, device)
        res = evaluate(y[va], pred, prob)
        res["fold"] = i
        fold_results.append(res)
        print(f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
              f"F1={res['f1']:.3f}")

    cv_mean = aggregate_folds(fold_results)

    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])
    # Internal val split (10%) for early stopping during refit
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]
    model = train_one(X_tv[idx_tr], y[tv][idx_tr],
                      X_tv[idx_va], y[tv][idx_va], device)
    prob_te, pred_te = predict(model, X_te, device)
    test_res = evaluate(y[te], pred_te, prob_te)
    print(f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
          f"F1={test_res['f1']:.3f}")

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_eye] device = {device}")

    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    print("\n=== MLP (BCE + pos_weight) ===")
    out = run(X, y, splits, device)
    out_path = EXP_DIR / "mlp_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[train_eye] saved {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
