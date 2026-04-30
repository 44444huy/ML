"""Baseline models B1/B2/B3: LR, SVM (RBF), Random Forest.

Each trait (eye/hair/skin) is trained as an independent classifier with
balanced class weights. Runs 5-fold CV for validation metrics and refits
on the full train+val for final test metrics.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import sys
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "data"))
sys.path.insert(0, str(SRC / "evaluation"))

from preprocess import build_dataset, TRAIT_NAMES
from splits import load_splits
from metrics import evaluate, aggregate_folds

ROOT = SRC.parent
EXP_DIR = ROOT / "experiments" / "baselines"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAITS = ["eye", "hair", "skin"]
N_CLASSES = {t: len(TRAIT_NAMES[t]) for t in TRAITS}


def make_model(name: str):
    if name == "LR":
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight="balanced", probability=True,
                   random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def run_cv(data: dict, splits: dict, model_name: str, trait: str) -> dict:
    X = data["X"].astype(np.float32)
    y = data[f"y_{trait}"]
    nc = N_CLASSES[trait]

    fold_results = []
    for i, fold in enumerate(splits["folds"]):
        tr, va = fold["train"], fold["valid"]
        clf = make_model(model_name)
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[va])
        probs = clf.predict_proba(X[va])
        # Pad probs if the classifier never saw some classes in training
        if probs.shape[1] != nc:
            full = np.zeros((len(va), nc), dtype=probs.dtype)
            full[:, clf.classes_] = probs
            probs = full
        res = evaluate(trait, y[va], y_pred, probs, nc)
        res["fold"] = i
        fold_results.append(res)

    agg = aggregate_folds(fold_results)
    return {"per_fold": fold_results, "cv_mean": agg}


def run_test(data: dict, splits: dict, model_name: str, trait: str) -> dict:
    X = data["X"].astype(np.float32)
    y = data[f"y_{trait}"]
    nc = N_CLASSES[trait]
    tv, te = splits["trainval"], splits["test"]

    clf = make_model(model_name)
    clf.fit(X[tv], y[tv])
    y_pred = clf.predict(X[te])
    probs = clf.predict_proba(X[te])
    if probs.shape[1] != nc:
        full = np.zeros((len(te), nc), dtype=probs.dtype)
        full[:, clf.classes_] = probs
        probs = full
    return evaluate(trait, y[te], y_pred, probs, nc)


def main():
    data = build_dataset(ROOT / "hirisplex_results_FN_v2.csv")
    splits = load_splits()

    all_results = {}
    for model_name in ["LR", "SVM", "RF"]:
        all_results[model_name] = {}
        for trait in TRAITS:
            print(f"[{model_name} / {trait}] running 5-fold CV...")
            cv = run_cv(data, splits, model_name, trait)
            print(f"[{model_name} / {trait}] fitting full train+val, scoring test...")
            test = run_test(data, splits, model_name, trait)
            all_results[model_name][trait] = {"cv": cv, "test": test}

            cvm = cv["cv_mean"]
            print(f"    CV   macro_f1={cvm['macro_f1']['mean']:.4f} ± {cvm['macro_f1']['std']:.4f}"
                  f"  acc={cvm['accuracy']['mean']:.4f}")
            print(f"    TEST macro_f1={test['macro_f1']:.4f}  acc={test['accuracy']:.4f}"
                  + (f"  mae={test['mae']:.4f}  qwk={test['qwk']:.4f}"
                     if trait in {"hair", "skin"} else ""))

    out_path = EXP_DIR / "baseline_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
