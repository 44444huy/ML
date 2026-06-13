"""Test-only TabPFN/TabICL runner for coat-color tasks.

This exists because full 5-fold CV for TabPFN/TabICL on coat-color data is
too expensive on the available CPU/RAM. Classical models, MLP, tuned MLP,
and TabNet are still run with the full CV + held-out test protocol.

Protocol:
    1. Use the persistent 80/20 trainval/test split.
    2. Split trainval internally into 90% context and 10% validation.
    3. Fit the foundation model on the context.
    4. Pick a probability threshold on internal validation by F1.
    5. Evaluate once on held-out test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits_coat import load_splits  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402

MODEL_CACHE_DIR = ROOT / "data" / "processed" / "model_cache"
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODEL_CACHE_DIR / "tabpfn"))

SEED = 42


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train).astype(np.float32), scaler.transform(X_other).astype(np.float32)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate([[0.5], np.linspace(0.01, 0.99, 99)]))
    best_t, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t


def tabpfn_predict(X_tr: np.ndarray, y_tr: np.ndarray, X: np.ndarray, device: str) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(
        device=device,
        random_state=SEED,
        ignore_pretraining_limits=True,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X)[:, 1]


def tabicl_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X: np.ndarray,
    *,
    device: str | None,
    n_estimators: int,
    batch_size: int,
) -> np.ndarray:
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=batch_size,
        allow_auto_download=True,
        device=device,
        random_state=SEED,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pos_cols = np.where(classes == 1)[0]
    if len(pos_cols) == 0:
        return np.zeros(len(X), dtype=np.float32)
    return probs[:, pos_cols[0]]


def run(args: argparse.Namespace) -> dict:
    bundle_path = ROOT / "data" / "processed" / "coat" / f"coat_{args.label}_processed.npz"
    bundle = np.load(bundle_path, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits(args.label)

    tv = np.asarray(splits["trainval"], dtype=int)
    te = np.asarray(splits["test"], dtype=int)
    X_tv, X_te = standardize(X[tv], X[te])
    y_tv, y_te = y[tv], y[te]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    print(
        f"[coat_test_only] label={args.label} model={args.model} "
        f"X={X.shape} selection={str(bundle['selection_mode'])}"
    )
    print(
        f"[coat_test_only] context={len(idx_tr)} internal_val={len(idx_va)} "
        f"test={len(te)}"
    )

    if args.model == "TabPFN":
        prob_va = tabpfn_predict(X_tv[idx_tr], y_tv[idx_tr], X_tv[idx_va], args.device)
        threshold = best_f1_threshold(y_tv[idx_va], prob_va)
        prob_te = tabpfn_predict(X_tv[idx_tr], y_tv[idx_tr], X_te, args.device)
        config = {"device": args.device}
    elif args.model == "TabICL":
        prob_va = tabicl_predict(
            X_tv[idx_tr],
            y_tv[idx_tr],
            X_tv[idx_va],
            device=args.tabicl_device,
            n_estimators=args.tabicl_estimators,
            batch_size=args.tabicl_batch_size,
        )
        threshold = best_f1_threshold(y_tv[idx_va], prob_va)
        prob_te = tabicl_predict(
            X_tv[idx_tr],
            y_tv[idx_tr],
            X_te,
            device=args.tabicl_device,
            n_estimators=args.tabicl_estimators,
            batch_size=args.tabicl_batch_size,
        )
        config = {
            "device": args.tabicl_device,
            "n_estimators": args.tabicl_estimators,
            "batch_size": args.tabicl_batch_size,
        }
    else:
        raise ValueError(args.model)

    pred_te = (prob_te >= threshold).astype(int)
    test_res = evaluate(y_te, pred_te, prob_te)
    test_res["threshold"] = float(threshold)
    print(
        f"[coat_test_only] TEST {args.model}: PR-AUC={test_res['pr_auc']:.4f} "
        f"ROC={test_res['roc_auc']:.4f} F1={test_res['f1']:.3f} "
        f"precision={test_res['precision']:.3f} recall={test_res['recall']:.3f} "
        f"t={threshold:.2f}"
    )

    return {
        "test": test_res,
        "test_only": True,
        "config": {
            "runner": "dog/src/experiments/run_coat_foundation_test_only.py",
            "threshold_source": "10% internal trainval validation split",
            **config,
        },
    }


def save_result(label: str, model: str, result: dict) -> Path:
    results_path = ROOT / "experiments" / "coat" / label / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    data.setdefault("models", {})
    data["models"][model] = result
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return results_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black")
    parser.add_argument("--model", choices=["TabPFN", "TabICL"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tabicl_device", default="cpu")
    parser.add_argument("--tabicl_estimators", type=int, default=1)
    parser.add_argument("--tabicl_batch_size", type=int, default=1)
    args = parser.parse_args()

    result = run(args)
    out_path = save_result(args.label, args.model, result)
    print(f"[coat_test_only] updated {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
