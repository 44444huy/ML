"""Classical baselines for the dog eye task.

Three baselines, all using the same persistent splits and the same
metric suite as the proposed methods:

    Majority — always predict P(blue)=p_train. The "do nothing" floor.
    LR       — logistic regression with class_weight="balanced".
    RF       — random forest (n=500, balanced subsample) — handles
               non-linear SNP × SNP interactions.

For each baseline we run 5-fold CV on trainval and refit on the full
trainval to score test.

Output JSON:
    experiments/eye/baseline_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402
from evaluation.metrics import aggregate_folds, best_f1_threshold, evaluate  # noqa: E402

NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXP_DIR / "baseline_results.json"

SEED = 42


class MajorityProb:
    """Predict the training prior probability for every input."""
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - np.full(n, self.p_), np.full(n, self.p_)])

    def predict(self, X):
        # Hard predict the majority class (always 0 here)
        return np.zeros(len(X), dtype=int)


def make_models() -> dict:
    return {
        "Majority": MajorityProb(),
        "LR": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, solver="liblinear",
        ),
        "RF": RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
    }


def standardize(X_train, X_other):
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_other)


def threshold_for_model(model_name: str, y_valid: np.ndarray, prob_valid: np.ndarray) -> float:
    """Tune probabilistic models; keep Majority as a true majority baseline."""
    if model_name == "Majority":
        return 0.5
    return best_f1_threshold(y_valid, prob_valid)


def run_one(model_name: str, X: np.ndarray, y: np.ndarray, splits: dict) -> dict:
    print(f"\n=== {model_name} ===")

    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        m = make_models()[model_name]
        m.fit(X_tr, y[tr])
        prob = m.predict_proba(X_va)[:, 1]
        threshold = threshold_for_model(model_name, y[va], prob)
        pred = (prob >= threshold).astype(int)
        res = evaluate(y[va], pred, prob)
        res["fold"] = i
        res["threshold"] = float(threshold)
        fold_results.append(res)
        print(f"  fold {i}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
              f"F1={res['f1']:.3f} (t={threshold:.2f})")

    cv_mean = aggregate_folds(fold_results)

    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    m_threshold = make_models()[model_name]
    m_threshold.fit(X_tv[idx_tr], y[tv][idx_tr])
    prob_va = m_threshold.predict_proba(X_tv[idx_va])[:, 1]
    threshold = threshold_for_model(model_name, y[tv][idx_va], prob_va)

    # Refit on full trainval, score test using the validation-selected threshold.
    m = make_models()[model_name]
    m.fit(X_tv, y[tv])
    prob_te = m.predict_proba(X_te)[:, 1]
    pred_te = (prob_te >= threshold).astype(int)
    test_res = evaluate(y[te], pred_te, prob_te)
    test_res["threshold"] = float(threshold)
    print(f"  TEST  PR-AUC={test_res['pr_auc']:.4f} ROC={test_res['roc_auc']:.4f} "
          f"F1={test_res['f1']:.3f} (t={threshold:.2f})")

    return {"cv": {"per_fold": fold_results, "cv_mean": cv_mean}, "test": test_res}


def main() -> int:
    bundle = np.load(NPZ_PATH, allow_pickle=True)
    X = bundle["X"].astype(np.float32)
    y = bundle["y"].astype(int)
    splits = load_splits()

    out = {}
    for name in ["Majority", "LR", "RF"]:
        out[name] = run_one(name, X, y, splits)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[baselines] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
