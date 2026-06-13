"""Train binary classifiers for one Darwin's Ark coat-color label.

Default task:
    black coat vs not black

Usage:
    python dog/src/train/train_coat.py --label black
    python dog/src/train/train_coat.py --label red_brown_tan --models Majority LR RF MLP TabICL
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits_coat import load_splits  # noqa: E402
from evaluation.metrics import aggregate_folds, evaluate  # noqa: E402
from models.mlp import MLPBinary  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed" / "coat"
EXP_ROOT = ROOT / "experiments" / "coat"
MODEL_CACHE_DIR = ROOT / "data" / "processed" / "model_cache"
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODEL_CACHE_DIR / "tabpfn"))

SEED = 42


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


class MajorityProb:
    """Predict the training prior probability for every row."""

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        self.majority_ = int(self.p_ >= 0.5)
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_, dtype=np.float32)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=int)


def npz_path(label: str) -> Path:
    return PROCESSED_DIR / f"coat_{label}_processed.npz"


def exp_dir(label: str) -> Path:
    path = EXP_ROOT / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train).astype(np.float32), sc.transform(X_other).astype(np.float32)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
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


def train_mlp_one(X_tr, y_tr, X_va, y_va, device: torch.device) -> nn.Module:
    from sklearn.metrics import average_precision_score

    seed_all()
    model = MLPBinary(
        in_dim=X_tr.shape[1],
        hidden=HP.hidden,
        n_layers=HP.n_layers,
        dropout=HP.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)

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

    for _ in range(HP.epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for s in range(0, len(X_tr_t), HP.batch_size):
            idx = perm[s:s + HP.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        pr = float(average_precision_score(y_va, va_prob)) if y_va.sum() > 0 else 0.0
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


@torch.no_grad()
def predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sigmoid(model(X_t)).cpu().numpy()


def fit_predict(model_name: str, X_tr, y_tr, X_va, y_va, X_te, args, device) -> tuple[np.ndarray, float]:
    """Return (P(class=1), threshold)."""
    if model_name == "Majority":
        model = MajorityProb().fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1], 0.5

    if model_name == "LR":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="liblinear",
        ).fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1], 0.5

    if model_name == "RF":
        model = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ).fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1], 0.5

    if model_name == "MLP":
        model = train_mlp_one(X_tr, y_tr, X_va, y_va, device)
        prob = predict_mlp(model, X_te, device)
        return prob, 0.5

    if model_name == "TabPFN":
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier(
            device=str(device),
            random_state=SEED,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        threshold = best_f1_threshold(y_va, clf.predict_proba(X_va)[:, 1])
        return prob, threshold

    if model_name == "TabICL":
        from tabicl import TabICLClassifier

        clf = TabICLClassifier(
            n_estimators=args.tabicl_estimators,
            batch_size=args.tabicl_batch_size,
            kv_cache=args.tabicl_kv_cache,
            allow_auto_download=not args.no_auto_download,
            device=args.tabicl_device,
            random_state=SEED,
            verbose=False,
        )
        clf.fit(X_tr, y_tr)
        probs_te = clf.predict_proba(X_te)
        probs_va = clf.predict_proba(X_va)
        classes = np.asarray(clf.classes_)
        pos_cols = np.where(classes == 1)[0]
        if len(pos_cols) == 0:
            prob_te = np.zeros(len(X_te), dtype=np.float32)
            prob_va = np.zeros(len(X_va), dtype=np.float32)
        else:
            prob_te = probs_te[:, pos_cols[0]]
            prob_va = probs_va[:, pos_cols[0]]
        return prob_te, best_f1_threshold(y_va, prob_va)

    if model_name == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        clf = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params=dict(step_size=20, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=SEED,
            device_name=str(device),
            verbose=0,
        )
        clf.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["auc"],
            max_epochs=200,
            patience=25,
            batch_size=256,
            virtual_batch_size=64,
            weights={0: 1.0, 1: float(n_neg) / max(n_pos, 1)},
        )
        return clf.predict_proba(X_te)[:, 1], 0.5

    raise ValueError(f"Unknown model: {model_name}")


def run_model(model_name: str, X: np.ndarray, y: np.ndarray, splits: dict, args, device) -> dict:
    print(f"\n=== {model_name} ===")
    fold_results = []

    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        prob, threshold = fit_predict(
            model_name,
            X_tr,
            y[tr],
            X_va,
            y[va],
            X_va,
            args,
            device,
        )
        pred = (prob >= threshold).astype(int)
        res = evaluate(y[va], pred, prob)
        res["fold"] = i
        res["threshold"] = float(threshold)
        fold_results.append(res)
        print(
            f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
            f"F1={res['f1']:.3f} (t={threshold:.2f})"
        )

    cv_mean = aggregate_folds(fold_results)

    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])
    y_tv = y[tv]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    prob_te, threshold = fit_predict(
        model_name,
        X_tv[idx_tr],
        y_tv[idx_tr],
        X_tv[idx_va],
        y_tv[idx_va],
        X_te,
        args,
        device,
    )
    pred_te = (prob_te >= threshold).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = float(threshold)
    print(
        f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
        f"F1={test_res['f1']:.3f} (t={threshold:.2f})"
    )

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Majority", "LR", "RF", "MLP", "TabICL"],
        choices=["Majority", "LR", "RF", "MLP", "TabPFN", "TabICL", "TabNet"],
    )
    parser.add_argument("--rf_estimators", type=int, default=500)
    parser.add_argument("--tabicl_device", default=None, help="TabICL device; default auto-selects.")
    parser.add_argument("--tabicl_estimators", type=int, default=4)
    parser.add_argument("--tabicl_batch_size", type=int, default=4)
    parser.add_argument("--tabicl_kv_cache", action="store_true")
    parser.add_argument("--no_auto_download", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_coat] label={args.label}")
    print(f"[train_coat] torch device={device}")
    print(f"[train_coat] models={args.models}")

    bundle = np.load(npz_path(args.label), allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits(args.label)
    print(f"[train_coat] X.shape={X.shape}; positive rate={y.mean():.3%}")
    print(f"[train_coat] selection={str(bundle['selection_mode'])}")

    current_selection = str(bundle["selection_mode"])
    current_n_features = int(X.shape[1])
    out_path = exp_dir(args.label) / "results.json"
    if out_path.exists():
        results = json.loads(out_path.read_text())
        stale = (
            results.get("selection_mode") != current_selection
            or results.get("n_features") != current_n_features
        )
        if stale:
            print(
                "[train_coat] existing results use a different feature "
                "selection; resetting model results."
            )
            results = {"models": {}}
        else:
            results.setdefault("models", {})
    else:
        results = {"models": {}}

    results.update({
        "label": args.label,
        "label_column": str(bundle["label_column"]),
        "selection_mode": current_selection,
        "n_samples": int(len(y)),
        "n_features": current_n_features,
        "positive_rate": float(y.mean()),
    })

    for model_name in args.models:
        results["models"][model_name] = run_model(model_name, X, y, splits, args, device)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[train_coat] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
