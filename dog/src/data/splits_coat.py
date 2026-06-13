"""Persistent stratified splits for one binary coat-color label.

Usage:
    python dog/src/data/splits_coat.py --label black
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / "coat"
SEED = 42


def npz_path(label: str) -> Path:
    return PROCESSED_DIR / f"coat_{label}_processed.npz"


def split_path(label: str) -> Path:
    return PROCESSED_DIR / f"coat_{label}_splits.json"


def make_splits(y: np.ndarray, *, n_splits: int = 5) -> dict:
    n = len(y)
    idx = np.arange(n)
    trainval_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=SEED
    )

    min_class = int(np.bincount(y[trainval_idx]).min())
    n_splits = min(n_splits, min_class)
    if n_splits < 2:
        raise RuntimeError("Need at least two samples per class in trainval for CV.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = []
    for tr, va in skf.split(trainval_idx, y[trainval_idx]):
        folds.append({
            "train": trainval_idx[tr].tolist(),
            "valid": trainval_idx[va].tolist(),
        })

    return {
        "test": test_idx.tolist(),
        "trainval": trainval_idx.tolist(),
        "folds": folds,
        "seed": SEED,
        "n_splits": n_splits,
    }


def save_splits(label: str) -> dict:
    bundle = np.load(npz_path(label), allow_pickle=True)
    y = bundle["y"].astype(int)
    out = make_splits(y)
    path = split_path(label)
    path.write_text(json.dumps(out), encoding="utf-8")

    test_idx = np.asarray(out["test"])
    trainval_idx = np.asarray(out["trainval"])
    print(f"[splits_coat] label={label}")
    print(f"[splits_coat] saved {path}")
    print(f"  test: n={len(test_idx)}, pos={int(y[test_idx].sum())} ({y[test_idx].mean():.3%})")
    print(
        f"  trainval: n={len(trainval_idx)}, pos={int(y[trainval_idx].sum())} "
        f"({y[trainval_idx].mean():.3%})"
    )
    for i, f in enumerate(out["folds"]):
        ytr, yva = y[f["train"]], y[f["valid"]]
        print(
            f"  fold {i}: train pos={int(ytr.sum())}/{len(ytr)} | "
            f"valid pos={int(yva.sum())}/{len(yva)}"
        )
    return out


def load_splits(label: str) -> dict:
    return json.loads(split_path(label).read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black")
    args = parser.parse_args()
    save_splits(args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
