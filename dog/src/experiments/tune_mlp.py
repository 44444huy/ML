"""Hyperparameter tuning for the MLP via grid search + 5-fold CV.

Method:
    For every combination of (hidden, n_layers, dropout, lr, weight_decay):
      1. Run 5-fold stratified CV on trainval (test is untouched).
      2. Record CV-mean PR-AUC.
    Pick the config with the highest CV-mean PR-AUC.
    Refit that config on the full trainval and report test metrics.

Why this exists:
    The MLP in train_eye.py uses a single fixed config (hidden=128,
    n_layers=2, dropout=0.3, lr=1e-3, wd=1e-4). This script tests
    72 combinations to confirm the choice or find a better one,
    using ONLY trainval — the test set is held out until the final
    refit.

Output:
    experiments/eye/hp_tuning_mlp.json   (every config + best)
    experiments/eye/hp_tuning_mlp.md     (top-10 ranking table)
    experiments/eye/mlp_best_results.json (refit on best config,
                                            same schema as
                                            mlp_results.json)

Usage:
    python dog/src/experiments/tune_mlp.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import aggregate_folds, best_f1_threshold, evaluate  # noqa: E402
from models.mlp import MLPBinary  # noqa: E402
from train.train_eye import SEED, predict, seed_all, standardize  # noqa: E402

import torch.nn as nn  # noqa: E402

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = EXP_DIR / "hp_tuning_mlp.json"
OUT_MD = EXP_DIR / "hp_tuning_mlp.md"
OUT_BEST = EXP_DIR / "mlp_best_results.json"

# ─── Grid ────────────────────────────────────────────────────────────────
GRID = {
    "hidden":       [64, 128, 256],
    "n_layers":     [1, 2, 3],
    "dropout":      [0.3, 0.5],
    "lr":           [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
}
# Fixed (not in grid)
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64


def train_one(X_tr, y_tr, X_va, y_va, device, *,
              hidden, n_layers, dropout, lr, weight_decay):
    """Same training loop as train_eye.train_one but with HP knobs."""
    from sklearn.metrics import average_precision_score
    seed_all(SEED)
    model = MLPBinary(in_dim=X_tr.shape[1], hidden=hidden,
                      n_layers=n_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)],
                      dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    best_pr, best_state, patience_left = -1.0, None, PATIENCE
    n = len(X_tr_t)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(X_tr_t[idx]), y_tr_t[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        pr = float(average_precision_score(y_va, va_prob)) \
            if y_va.sum() > 0 else 0.0

        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def cv_score(X, y, splits, device, hp: dict) -> dict:
    """Run 5-fold CV with one HP config, return CV-mean metrics."""
    fold_results = []
    for fold in splits["folds"]:
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        model = train_one(X_tr, y[tr], X_va, y[va], device, **hp)
        prob = predict(model, X_va, device)
        threshold = best_f1_threshold(y[va], prob)
        pred = (prob >= threshold).astype(int)
        res = evaluate(y[va], pred, prob)
        res["threshold"] = float(threshold)
        fold_results.append(res)
    return aggregate_folds(fold_results)


def refit_and_test(X, y, splits, device, hp: dict) -> dict:
    """Refit on full trainval (with 90/10 internal val for early stop),
    score the held-out test set. Same recipe as train_eye.run()."""
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    model = train_one(X_tv[idx_tr], y[tv][idx_tr],
                      X_tv[idx_va], y[tv][idx_va],
                      device, **hp)
    prob_va = predict(model, X_tv[idx_va], device)
    threshold = best_f1_threshold(y[tv][idx_va], prob_va)
    prob_te = predict(model, X_te, device)
    pred_te = (prob_te >= threshold).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = float(threshold)
    return test_res


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tune_mlp] device = {device}")

    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    # Build grid
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    n_configs = len(combos)
    print(f"[tune_mlp] grid: {n_configs} configs × 5 folds = "
          f"{n_configs * 5} trainings\n")

    results = []
    t_start = time.time()
    for i, vals in enumerate(combos, 1):
        hp = dict(zip(keys, vals))
        t0 = time.time()
        cv = cv_score(X, y, splits, device, hp)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = (elapsed / i) * (n_configs - i)
        row = {
            "config_id": i,
            **hp,
            "cv_pr_auc":  cv["pr_auc"]["mean"],
            "cv_pr_std":  cv["pr_auc"]["std"],
            "cv_roc_auc": cv["roc_auc"]["mean"],
            "cv_roc_std": cv["roc_auc"]["std"],
            "cv_f1":      cv["f1"]["mean"],
            "cv_f1_std":  cv["f1"]["std"],
            "fold_time_s": dt,
        }
        results.append(row)
        print(f"[{i:2d}/{n_configs}] "
              f"h={hp['hidden']:3d} nl={hp['n_layers']} "
              f"d={hp['dropout']:.1f} lr={hp['lr']:.0e} "
              f"wd={hp['weight_decay']:.0e}  "
              f"CV PR={cv['pr_auc']['mean']:.4f} ROC={cv['roc_auc']['mean']:.4f} "
              f"F1={cv['f1']['mean']:.3f}  ({dt:.0f}s, ETA {eta/60:.1f}min)")

    # Rank by CV PR-AUC
    results.sort(key=lambda r: r["cv_pr_auc"], reverse=True)
    best_hp = {k: results[0][k] for k in keys}
    print(f"\n[tune_mlp] BEST config (by CV PR-AUC):")
    for k, v in best_hp.items():
        print(f"    {k} = {v}")
    print(f"  CV PR-AUC  = {results[0]['cv_pr_auc']:.4f}")
    print(f"  CV ROC-AUC = {results[0]['cv_roc_auc']:.4f}")
    print(f"  CV F1      = {results[0]['cv_f1']:.4f}")

    # Refit best config on full trainval, score test
    print(f"\n[tune_mlp] refitting best config on full trainval ...")
    test_res = refit_and_test(X, y, splits, device, best_hp)
    print(f"  TEST PR-AUC  = {test_res['pr_auc']:.4f}")
    print(f"  TEST ROC-AUC = {test_res['roc_auc']:.4f}")
    print(f"  TEST F1      = {test_res['f1']:.4f}")

    # Save grid results
    OUT_JSON.write_text(json.dumps({
        "grid": GRID,
        "n_configs": n_configs,
        "ranked_configs": results,
        "best_hp": best_hp,
        "best_cv": {
            "pr_auc":  results[0]["cv_pr_auc"],
            "roc_auc": results[0]["cv_roc_auc"],
            "f1":      results[0]["cv_f1"],
        },
        "test_with_best_hp": test_res,
    }, indent=2))
    print(f"\n[tune_mlp] saved {OUT_JSON}")

    # Save best-config refit in same schema as mlp_results.json
    best_cv_only = cv_score(X, y, splits, device, best_hp)
    fold_results_for_best = []
    for fold in splits["folds"]:
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        model = train_one(X_tr, y[tr], X_va, y[va], device, **best_hp)
        prob = predict(model, X_va, device)
        threshold = best_f1_threshold(y[va], prob)
        pred = (prob >= threshold).astype(int)
        res = evaluate(y[va], pred, prob)
        res["threshold"] = float(threshold)
        fold_results_for_best.append(res)
    OUT_BEST.write_text(json.dumps({
        "cv": {"per_fold": fold_results_for_best,
               "cv_mean": best_cv_only},
        "test": test_res,
        "hyperparameters": best_hp,
    }, indent=2))
    print(f"[tune_mlp] saved {OUT_BEST}")

    # Markdown top-10 table
    md = [
        "# MLP Hyperparameter Tuning (5-fold CV on trainval)",
        "",
        f"**Grid size**: {n_configs} configs × 5 folds = "
        f"{n_configs * 5} trainings.  ",
        f"**Selection**: rank by CV-mean PR-AUC. Test set was NOT used "
        "for selection.",
        "",
        "## Best config",
        "",
        "| Param | Value |",
        "|---|---|",
    ]
    for k, v in best_hp.items():
        md.append(f"| {k} | {v} |")
    md += [
        "",
        f"CV-mean PR-AUC = **{results[0]['cv_pr_auc']:.4f}** "
        f"(± {results[0]['cv_pr_std']:.4f})  ",
        f"Test PR-AUC = **{test_res['pr_auc']:.4f}**, "
        f"ROC-AUC = {test_res['roc_auc']:.4f}, "
        f"F1 = {test_res['f1']:.4f}",
        "",
        "## Top 10 configs by CV PR-AUC",
        "",
        "| Rank | hidden | n_layers | dropout | lr | weight_decay "
        "| CV PR-AUC | CV ROC-AUC | CV F1 |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for rank, r in enumerate(results[:10], 1):
        md.append(
            f"| {rank} | {r['hidden']} | {r['n_layers']} "
            f"| {r['dropout']} | {r['lr']:.0e} | {r['weight_decay']:.0e} "
            f"| {r['cv_pr_auc']:.4f} ± {r['cv_pr_std']:.4f} "
            f"| {r['cv_roc_auc']:.4f} | {r['cv_f1']:.4f} |"
        )
    md += [
        "",
        "## Worst 3 configs (sanity check)",
        "",
        "| Rank | hidden | n_layers | dropout | lr | weight_decay "
        "| CV PR-AUC |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for rank, r in enumerate(results[-3:], n_configs - 2):
        md.append(
            f"| {rank} | {r['hidden']} | {r['n_layers']} "
            f"| {r['dropout']} | {r['lr']:.0e} | {r['weight_decay']:.0e} "
            f"| {r['cv_pr_auc']:.4f} |"
        )
    md += [""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[tune_mlp] saved {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
