"""Persistent stratified splits for the dog eye dataset.

Saves a JSON with:
    test:     held-out test indices (~20% of dogs, stratified on label)
    trainval: indices used for cross-validation
    folds:    5 stratified folds inside trainval, each {train, valid}

All indices refer to rows of the npz bundle (eye_processed.npz).

Why stratified: positive class is ~4%, so a uniform split could end up with
zero positives in a fold.

Usage:
    python dog/src/data/splits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = ROOT / "data" / "processed" / "eye_processed.npz"
SPLIT_PATH = ROOT / "data" / "processed" / "eye_splits.json"

SEED = 42


def main() -> int:
    bundle = np.load(NPZ_PATH, allow_pickle=True)
    y = bundle["y"]
    n = len(y)
    idx = np.arange(n)

    trainval_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=SEED
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for tr, va in skf.split(trainval_idx, y[trainval_idx]):
        folds.append({
            "train": trainval_idx[tr].tolist(),
            "valid": trainval_idx[va].tolist(),
        })

    out = {
        "test": test_idx.tolist(),
        "trainval": trainval_idx.tolist(),
        "folds": folds,
        "seed": SEED,
    }
    SPLIT_PATH.write_text(json.dumps(out))
    print(f"[splits] saved {SPLIT_PATH}")
    print(f"  test: n={len(test_idx)}, pos={int(y[test_idx].sum())} ({y[test_idx].mean():.3%})")
    print(f"  trainval: n={len(trainval_idx)}, pos={int(y[trainval_idx].sum())} ({y[trainval_idx].mean():.3%})")
    for i, f in enumerate(folds):
        ytr, yva = y[f["train"]], y[f["valid"]]
        print(f"  fold {i}: train pos={int(ytr.sum())}/{len(ytr)} | valid pos={int(yva.sum())}/{len(yva)}")
    return 0


def load_splits() -> dict:
    return json.loads(SPLIT_PATH.read_text())


if __name__ == "__main__":
    sys.exit(main())
