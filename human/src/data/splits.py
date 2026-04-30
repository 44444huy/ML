"""Persistent train/test splits and K-fold assignments.

Design:
    - Stratify the outer train/test split by hair (rarest classes).
    - Inner 5-fold CV stratified by hair as well.
    - Small classes (count < n_splits) get merged with a neighbor before
      stratification to avoid sklearn errors, but the original labels are
      preserved in the saved arrays.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5


def _safe_strata(y: np.ndarray, min_count: int) -> np.ndarray:
    """Merge any class with count < min_count into the nearest class by id.

    Used only to feed StratifiedKFold; does not modify real labels.
    """
    y = y.copy()
    counts = np.bincount(y)
    rare = np.where(counts < min_count)[0]
    if len(rare) == 0:
        return y
    valid = np.where(counts >= min_count)[0]
    for r in rare:
        nearest = valid[np.argmin(np.abs(valid - r))]
        y[y == r] = nearest
    return y


def build_splits(data: dict) -> dict:
    N = len(data["X"])
    idx_all = np.arange(N)
    strata = _safe_strata(data["y_hair"], min_count=N_FOLDS + 1)

    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SEED, stratify=strata)

    strata_tv = strata[idx_trainval]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for tr, va in skf.split(idx_trainval, strata_tv):
        folds.append({
            "train": idx_trainval[tr],
            "valid": idx_trainval[va],
        })
    return {"trainval": idx_trainval, "test": idx_test, "folds": folds}


def save_splits(splits: dict, out_dir: Path = SPLIT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trainval_idx.npy", splits["trainval"])
    np.save(out_dir / "test_idx.npy", splits["test"])
    for i, f in enumerate(splits["folds"]):
        np.save(out_dir / f"fold{i}_train.npy", f["train"])
        np.save(out_dir / f"fold{i}_valid.npy", f["valid"])


def load_splits(in_dir: Path = SPLIT_DIR) -> dict:
    trainval = np.load(in_dir / "trainval_idx.npy")
    test = np.load(in_dir / "test_idx.npy")
    folds = []
    for i in range(N_FOLDS):
        folds.append({
            "train": np.load(in_dir / f"fold{i}_train.npy"),
            "valid": np.load(in_dir / f"fold{i}_valid.npy"),
        })
    return {"trainval": trainval, "test": test, "folds": folds}


if __name__ == "__main__":
    from preprocess import build_dataset
    data = build_dataset(ROOT / "hirisplex_results_FN_v2.csv")
    splits = build_splits(data)
    save_splits(splits)
    print(f"Train+val: {len(splits['trainval'])}, Test: {len(splits['test'])}")
    for i, f in enumerate(splits["folds"]):
        print(f"  Fold {i}: train={len(f['train'])}, valid={len(f['valid'])}")
