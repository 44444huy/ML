"""Hyperparameter tuning for the coat-color MLP.

This mirrors the eye-color MLP tuning idea, but writes JSON only. The test
set is untouched during selection; configs are ranked by CV-mean PR-AUC on
trainval, then the best config is refit and evaluated on held-out test.

Usage:
    python dog/src/experiments/tune_mlp_coat.py --label black
    python dog/src/experiments/tune_mlp_coat.py --label black --full_grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits_coat import load_splits  # noqa: E402
from evaluation.metrics import aggregate_folds, best_f1_threshold, evaluate  # noqa: E402
from models.mlp import MLPBinary  # noqa: E402
from train.train_coat import exp_dir, npz_path, seed_all, standardize  # noqa: E402

SEED = 42
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64

FAST_GRID = {
    "hidden": [64, 128, 256],
    "n_layers": [1, 2],
    "dropout": [0.3, 0.5],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4],
}

FULL_GRID = {
    "hidden": [64, 128, 256],
    "n_layers": [1, 2, 3],
    "dropout": [0.3, 0.5],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
}


def train_one(X_tr, y_tr, X_va, y_va, device, hp: dict) -> MLPBinary:
    from sklearn.metrics import average_precision_score

    seed_all(SEED)
    model = MLPBinary(
        in_dim=X_tr.shape[1],
        hidden=hp["hidden"],
        n_layers=hp["n_layers"],
        dropout=hp["dropout"],
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    best_pr = -1.0
    best_state = None
    patience_left = PATIENCE

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for s in range(0, len(X_tr_t), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
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
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, device) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    return torch.sigmoid(model(X_t)).cpu().numpy()


def cv_score(X, y, splits, device, hp: dict) -> tuple[dict, list[dict]]:
    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        model = train_one(X_tr, y[tr], X_va, y[va], device, hp)
        prob = predict(model, X_va, device)
        threshold = best_f1_threshold(y[va], prob)
        pred = (prob >= threshold).astype(int)
        res = evaluate(y[va], pred, prob)
        res["fold"] = i
        res["threshold"] = float(threshold)
        fold_results.append(res)
    return aggregate_folds(fold_results), fold_results


def refit_and_test(X, y, splits, device, hp: dict) -> dict:
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])
    y_tv = y[tv]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    model = train_one(X_tv[idx_tr], y_tv[idx_tr], X_tv[idx_va], y_tv[idx_va], device, hp)
    prob_va = predict(model, X_tv[idx_va], device)
    threshold = best_f1_threshold(y_tv[idx_va], prob_va)
    prob_te = predict(model, X_te, device)
    pred_te = (prob_te >= threshold).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = float(threshold)
    return test_res


def build_grid(full_grid: bool) -> tuple[dict, list[dict]]:
    grid = FULL_GRID if full_grid else FAST_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    return grid, combos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black")
    parser.add_argument("--full_grid", action="store_true",
                        help="Use the 72-config eye-style grid. Default uses 24 configs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tune_mlp_coat] label={args.label}")
    print(f"[tune_mlp_coat] device={device}")

    bundle = np.load(npz_path(args.label), allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits(args.label)
    grid, combos = build_grid(args.full_grid)

    print(f"[tune_mlp_coat] X.shape={X.shape}; positive rate={y.mean():.3%}")
    print(f"[tune_mlp_coat] grid: {len(combos)} configs x {len(splits['folds'])} folds")

    ranked = []
    t_start = time.time()
    for i, hp in enumerate(combos, start=1):
        t0 = time.time()
        cv, _ = cv_score(X, y, splits, device, hp)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = (elapsed / i) * (len(combos) - i)

        row = {
            "config_id": i,
            **hp,
            "cv_pr_auc": cv["pr_auc"]["mean"],
            "cv_pr_std": cv["pr_auc"]["std"],
            "cv_roc_auc": cv["roc_auc"]["mean"],
            "cv_roc_std": cv["roc_auc"]["std"],
            "cv_f1": cv["f1"]["mean"],
            "cv_f1_std": cv["f1"]["std"],
            "config_time_s": dt,
        }
        ranked.append(row)
        print(
            f"[{i:02d}/{len(combos)}] h={hp['hidden']:3d} nl={hp['n_layers']} "
            f"d={hp['dropout']:.1f} lr={hp['lr']:.0e} wd={hp['weight_decay']:.0e} "
            f"CV PR={row['cv_pr_auc']:.4f} ROC={row['cv_roc_auc']:.4f} "
            f"F1={row['cv_f1']:.3f} ({dt:.0f}s, ETA {eta/60:.1f}m)"
        )

    ranked.sort(key=lambda r: r["cv_pr_auc"], reverse=True)
    best_hp = {
        "hidden": ranked[0]["hidden"],
        "n_layers": ranked[0]["n_layers"],
        "dropout": ranked[0]["dropout"],
        "lr": ranked[0]["lr"],
        "weight_decay": ranked[0]["weight_decay"],
    }

    print("\n[tune_mlp_coat] best by CV PR-AUC:")
    for k, v in best_hp.items():
        print(f"  {k}: {v}")
    print(f"  CV PR-AUC={ranked[0]['cv_pr_auc']:.4f}")

    best_cv, best_folds = cv_score(X, y, splits, device, best_hp)
    test_res = refit_and_test(X, y, splits, device, best_hp)
    print(
        f"[tune_mlp_coat] TEST PR-AUC={test_res['pr_auc']:.4f} "
        f"ROC={test_res['roc_auc']:.4f} F1={test_res['f1']:.3f}"
    )

    exp = exp_dir(args.label)
    out_tuning = exp / "hp_tuning_mlp.json"
    out_tuning.write_text(json.dumps({
        "label": args.label,
        "grid": grid,
        "full_grid": args.full_grid,
        "n_configs": len(combos),
        "ranked_configs": ranked,
        "best_hp": best_hp,
        "best_cv": {
            "pr_auc": ranked[0]["cv_pr_auc"],
            "roc_auc": ranked[0]["cv_roc_auc"],
            "f1": ranked[0]["cv_f1"],
        },
        "test_with_best_hp": test_res,
    }, indent=2), encoding="utf-8")

    tuned_result = {
        "cv": {"per_fold": best_folds, "cv_mean": best_cv},
        "test": test_res,
        "hyperparameters": best_hp,
        "tuning": {
            "selection_metric": "cv_pr_auc",
            "grid": "full" if args.full_grid else "fast",
            "n_configs": len(combos),
            "json": str(out_tuning),
        },
    }

    out_results = exp / "results.json"
    if out_results.exists():
        results = json.loads(out_results.read_text())
    else:
        results = {"label": args.label, "models": {}}
    results.setdefault("models", {})
    results["models"]["MLP (tuned)"] = tuned_result
    out_results.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[tune_mlp_coat] wrote {out_tuning}")
    print(f"[tune_mlp_coat] updated {out_results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
