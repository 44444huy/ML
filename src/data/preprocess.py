"""Preprocess HIrisPlex-S output into ML-ready dataset.

Produces:
    - X: (N, 41) SNP dosage matrix
    - y_hard: hard labels via argmax (eye, hair, skin)
    - y_soft: soft labels = raw p_value vectors per trait group
    - confidence: max p_value per trait group (used for filtering)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Trait group indices in the 14-dim p_value vector
EYE_SLICE = slice(0, 3)       # blue, intermediate, brown
HAIR_SLICE = slice(3, 9)      # blond, brown, red, black, light, dark
SKIN_SLICE = slice(9, 14)     # very pale, pale, intermediate, dark, dark-to-black

TRAIT_NAMES = {
    "eye": ["blue", "intermediate", "brown"],
    "hair": ["blond", "brown", "red", "black", "light", "dark"],
    "skin": ["very_pale", "pale", "intermediate", "dark", "dark_to_black"],
}

NUM_SNPS = 41
NUM_TRAITS = 14


def parse_snp_vector(cell: str) -> np.ndarray:
    """Parse a two-line input_csv cell into a 41-dim dosage vector."""
    lines = cell.strip().replace("\r\n", "\n").split("\n")
    return np.array(lines[1].split(","), dtype=np.int8)


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load raw HIrisPlex output, drop samples missing input_csv."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["input_csv"]).reset_index(drop=True)
    return df


def build_dataset(csv_path: str | Path) -> dict:
    """Build the full ML-ready dataset.

    Returns a dict with:
        X              : (N, 41) int8
        probs          : (N, 14) float32  — raw HIrisPlex p_values
        y_eye / y_hair / y_skin : (N,) int — argmax hard labels
        p_eye / p_hair / p_skin : (N, K) float32 — soft labels per trait
        conf_eye / conf_hair / conf_skin : (N,) float — max p within group
        samples        : (N,) sample IDs
    """
    df = load_raw(csv_path)

    X = np.stack([parse_snp_vector(c) for c in df["input_csv"].values]).astype(np.int8)

    prob_cols = [f"result/{i}/p_value" for i in range(NUM_TRAITS)]
    probs = df[prob_cols].values.astype(np.float32)

    p_eye = probs[:, EYE_SLICE]
    p_hair = probs[:, HAIR_SLICE]
    p_skin = probs[:, SKIN_SLICE]

    return {
        "X": X,
        "probs": probs,
        "p_eye": p_eye,
        "p_hair": p_hair,
        "p_skin": p_skin,
        "y_eye": p_eye.argmax(axis=1).astype(np.int64),
        "y_hair": p_hair.argmax(axis=1).astype(np.int64),
        "y_skin": p_skin.argmax(axis=1).astype(np.int64),
        "conf_eye": p_eye.max(axis=1),
        "conf_hair": p_hair.max(axis=1),
        "conf_skin": p_skin.max(axis=1),
        "samples": df["sample"].values,
    }


def save_processed(data: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "evc_processed.npz", **data)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data = build_dataset(root / "hirisplex_results_FN_v2.csv")
    save_processed(data, root / "data" / "processed")
    print(f"Saved dataset: N={len(data['X'])}, SNPs={data['X'].shape[1]}")
    print(f"  Eye  classes: {np.bincount(data['y_eye'], minlength=3)}")
    print(f"  Hair classes: {np.bincount(data['y_hair'], minlength=6)}")
    print(f"  Skin classes: {np.bincount(data['y_skin'], minlength=5)}")
