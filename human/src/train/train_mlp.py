"""Train an MLP per trait, for method M1 (hard CE) or M2 (soft KL).

Pipeline:
    - Load npz bundle and persistent splits.
    - For each trait, run 5-fold CV on the trainval split.
      Each fold: train MLP with early stopping on validation macro-F1.
    - After CV, refit on the full trainval split and score the held-out test.
    - Save a structured JSON identical in layout to baseline_results.json.

Two methods share one architecture (src.models.mlp.MLPClassifier):
    M1: CrossEntropyLoss with class_weight=balanced.
    M2: soft-label KL loss on raw HIrisPlex p_values.

Usage:
    python src/train/train_mlp.py --method M1
    python src/train/train_mlp.py --method M2
    python src/train/train_mlp.py --method both   # runs both, one JSON each
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import evaluate, aggregate_folds  # noqa: E402
from losses.soft_label import soft_kl_loss  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402

NPZ_PATH = ROOT / "data" / "processed" / "evc_processed.npz"
EXP_DIR = ROOT / "experiments" / "mlp"
EXP_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["eye", "hair", "skin"]
N_CLASSES = {"eye": 3, "hair": 6, "skin": 5}


# ---------- Hyperparameters (fixed for a clean ablation) ----------
@dataclass
class Hparams:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    patience: int = 20
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


def load_data() -> dict:
    with np.load(NPZ_PATH, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using stats from training only (no leakage)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_other - mean) / std


def make_loader(X: np.ndarray, y: np.ndarray | None, p: np.ndarray | None,
                batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    tensors = [torch.tensor(X, dtype=torch.float32)]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.long))
    if p is not None:
        tensors.append(torch.tensor(p, dtype=torch.float32))
    ds = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def macro_f1_from_logits(logits: torch.Tensor, y_true: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import f1_score
    y_pred = logits.argmax(dim=-1).cpu().numpy()
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=np.arange(n_classes), zero_division=0))


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, hard_pred) for X."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return probs, preds


def train_one(method: str, X_tr: np.ndarray, y_tr: np.ndarray, p_tr: np.ndarray,
              X_va: np.ndarray, y_va: np.ndarray, n_classes: int,
              device: torch.device) -> nn.Module:
    """Train an MLP on the given fold and return the best model by val macro-F1."""
    seed_all(SEED)

    model = MLPClassifier(
        in_dim=X_tr.shape[1], n_classes=n_classes,
        hidden=HP.hidden, n_layers=HP.n_layers, dropout=HP.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)

    if method == "M1":
        classes_present = np.unique(y_tr)
        cw = compute_class_weight("balanced", classes=classes_present, y=y_tr)
        weight = np.ones(n_classes, dtype=np.float32)
        weight[classes_present] = cw.astype(np.float32)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=device))
    elif method == "M2":
        loss_fn = None  # handled inline below
    else:
        raise ValueError(method)

    train_loader = make_loader(
        X_tr, y_tr if method == "M1" else None,
        p_tr if method == "M2" else None,
        batch_size=HP.batch_size, shuffle=True,
    )

    best_f1 = -1.0
    best_state = None
    patience_left = HP.patience

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    for epoch in range(HP.epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            opt.zero_grad()
            logits = model(x)
            if method == "M1":
                y = batch[1].to(device)
                loss = loss_fn(logits, y)
            else:  # M2
                p = batch[1].to(device)
                loss = soft_kl_loss(logits, p)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t)
        f1 = macro_f1_from_logits(va_logits, y_va, n_classes)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = HP.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_trait(method: str, trait: str, data: dict, splits: dict, device: torch.device) -> dict:
    X_all = data["X"].astype(np.float32)
    y_all = data[f"y_{trait}"]
    p_all = data[f"p_{trait}"]
    nc = N_CLASSES[trait]

    # ---- CV ----
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X_all[tr], X_all[va])
        model = train_one(method, X_tr, y_all[tr], p_all[tr],
                          X_va, y_all[va], nc, device)
        probs, preds = predict(model, X_va, device)
        res = evaluate(trait, y_all[va], preds, probs, nc)
        res["fold"] = i
        fold_results.append(res)
        print(f"  fold {i}: macro_f1={res['macro_f1']:.4f} acc={res['accuracy']:.4f}")

    cv_mean = aggregate_folds(fold_results)

    # ---- Refit on trainval, score on test ----
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X_all[tv], X_all[te])
    # For refit we have no held-out validation; use a small internal split
    # only for early stopping (10% of trainval), still keeping test untouched.
    rng = np.random.default_rng(SEED)
    n_tv = len(tv)
    perm = rng.permutation(n_tv)
    cut = int(0.9 * n_tv)
    idx_refit_tr, idx_refit_va = perm[:cut], perm[cut:]
    model = train_one(
        method,
        X_tv[idx_refit_tr], y_all[tv][idx_refit_tr], p_all[tv][idx_refit_tr],
        X_tv[idx_refit_va], y_all[tv][idx_refit_va], nc, device,
    )
    probs_te, preds_te = predict(model, X_te, device)
    test_res = evaluate(trait, y_all[te], preds_te, probs_te, nc)
    print(f"  TEST  macro_f1={test_res['macro_f1']:.4f} acc={test_res['accuracy']:.4f}")

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["M1", "M2", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_mlp] device = {device}")

    data = load_data()
    splits = load_splits()

    methods = ["M1", "M2"] if args.method == "both" else [args.method]
    for method in methods:
        all_res = {}
        for trait in TRAITS:
            print(f"\n=== {method} / {trait} ===")
            all_res[trait] = run_trait(method, trait, data, splits, device)
        out_path = EXP_DIR / f"{method.lower()}_results.json"
        out_path.write_text(json.dumps(all_res, indent=2))
        print(f"\n[train_mlp] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
