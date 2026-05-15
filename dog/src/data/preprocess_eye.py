"""Preprocess the canine eye-color dataset (Deane-Coe et al. 2018).

Pipeline:
    1. Load phenotype CSV → keep dogs in the discovery panel with a known
       eye-color label (blue=1 / brown=0). Heterochromia is treated as
       blue (positive class) per the original paper.
    2. Load GWAS results (.assoc.txt) → sort by p_wald → keep top-K SNPs.
       This is the "GWAS-informed feature selection" step that justifies
       using only K SNPs out of 213,245.
    3. Read genotype dosages for those K SNPs from the PLINK bed/bim/fam
       files (memory-mapped via pysnptools, no full load).
    4. Align rows: only keep dogs that appear both in genotype and
       phenotype tables.
    5. Save a single npz bundle with X, y, dog_id, breed, snp_ids,
       snp_chrom, snp_pos, p_wald.

Usage:
    python dog/src/data/preprocess_eye.py                # default p < 1.15e-7
    python dog/src/data/preprocess_eye.py --p 1e-5       # suggestive
    python dog/src/data/preprocess_eye.py --top_k 200    # legacy top-K mode

Why p < 1.15e-7 instead of an arbitrary top-K?
    1.15e-7 is the Bonferroni-corrected genome-wide threshold used in
    the original Deane-Coe et al. 2018 paper on THIS dataset
    (0.05 / ~430k markers after QC). Using the exact threshold from the
    reference paper is more defensible than picking K by hand. It is
    also appropriate here because the trait is oligogenic — Deane-Coe
    et al. show one dominant locus on chr18/ALX4 (p ~ 1.3e-68), so a
    small, statistically significant SNP set is sufficient.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# pysnptools reads PLINK bed without loading the whole matrix
from pysnptools.snpreader import Bed

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "eye"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHENO_PATH = RAW_DIR / "deane-coe_etal_canine_eye_color_GWAS_indiv_phenotype_logr_haplotype_breed.csv"
ASSOC_PATH = RAW_DIR / "deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel.assoc.txt"
BED_STEM = RAW_DIR / "deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel"
OUT_PATH = OUT_DIR / "eye_processed.npz"


def load_phenotype() -> pd.DataFrame:
    """Return phenotype frame restricted to the discovery panel with valid label."""
    df = pd.read_csv(PHENO_PATH, dtype={"anonymous_id": str})
    print(f"[pheno] raw rows: {len(df)}")
    print(f"[pheno] panels: {df['panel'].value_counts().to_dict()}")

    df = df[df["panel"] == "discovery"].copy()
    print(f"[pheno] discovery panel: {len(df)}")

    # Drop rows missing the label
    df = df.dropna(subset=["blue_eyes"])
    df["blue_eyes"] = df["blue_eyes"].astype(int)
    print(f"[pheno] with valid label: {len(df)}")
    print(f"[pheno] label balance: {df['blue_eyes'].value_counts().to_dict()}")

    return df.reset_index(drop=True)


def select_snps(p_threshold: float | None, top_k: int | None) -> pd.DataFrame:
    """Select SNPs by GWAS p_wald.

    Two modes:
      - p_threshold (default): keep SNPs with p_wald < threshold.
        Recommended: 1.15e-7 (Bonferroni cutoff used by Deane-Coe
        et al. 2018 on this dataset).
      - top_k (legacy): keep the K smallest-p_wald SNPs regardless of
        statistical significance.
    """
    df = pd.read_csv(ASSOC_PATH, sep=r"\s+")
    print(f"[gwas] total SNPs in assoc file: {len(df)}")
    df = df.dropna(subset=["p_wald"]).sort_values("p_wald").reset_index(drop=True)

    if top_k is not None:
        df = df.head(top_k).reset_index(drop=True)
        print(f"[gwas] mode = top_k ({top_k})")
    else:
        df = df[df["p_wald"] < p_threshold].reset_index(drop=True)
        print(f"[gwas] mode = p_wald < {p_threshold:.0e}")

    print(f"[gwas] kept {len(df)} SNPs")
    print(f"[gwas] best p_wald = {df['p_wald'].iloc[0]:.2e} on chr{df['chr'].iloc[0]}:{df['ps'].iloc[0]}")
    print(f"[gwas] worst p_wald kept = {df['p_wald'].iloc[-1]:.2e}")
    chr_counts = df['chr'].value_counts().sort_index().to_dict()
    print(f"[gwas] chromosome distribution: {chr_counts}")
    return df


def extract_genotypes(snp_df: pd.DataFrame, dog_ids_keep: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, kept_dog_ids_array, snp_id_list_in_X_order).

    Dosage encoding: 0/1/2 copies of allele1 (per pysnptools default).
    Missing values are filled with column mean.
    """
    bed = Bed(str(BED_STEM), count_A1=True)
    print(f"[geno] bed shape: {bed.iid_count} dogs × {bed.sid_count} SNPs")

    # SNP IDs in the .bim use the 'rs' column from the assoc file
    wanted_snp_ids = snp_df["rs"].astype(str).tolist()
    sid_to_idx = {sid: i for i, sid in enumerate(bed.sid)}
    snp_indices = [sid_to_idx[s] for s in wanted_snp_ids if s in sid_to_idx]
    found_ids = [s for s in wanted_snp_ids if s in sid_to_idx]
    missing = len(wanted_snp_ids) - len(found_ids)
    if missing:
        print(f"[geno] WARNING: {missing}/{len(wanted_snp_ids)} top SNPs not found in bim")
    print(f"[geno] extracting {len(snp_indices)} SNPs")

    # Dog IDs in .fam: column 'iid' (the second PLINK col). Format: "0001 0001 0 0 1 0"
    fam_iid = bed.iid[:, 1]  # second col = within-family ID
    iid_to_idx = {iid: i for i, iid in enumerate(fam_iid)}
    dog_indices = [iid_to_idx[d] for d in dog_ids_keep if d in iid_to_idx]
    kept_dog_ids = [d for d in dog_ids_keep if d in iid_to_idx]
    missing_dogs = len(dog_ids_keep) - len(kept_dog_ids)
    if missing_dogs:
        print(f"[geno] WARNING: {missing_dogs}/{len(dog_ids_keep)} phenotyped dogs not found in fam")
    print(f"[geno] extracting genotypes for {len(dog_indices)} dogs")

    # Read selected sub-matrix
    sub = bed[dog_indices, snp_indices].read()
    X = sub.val.astype(np.float32)  # (n_dogs, n_snps), dosage 0/1/2, NaN if missing

    # Fill missing with column mean
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_mean = np.nanmean(X, axis=0)
        # Any all-NaN columns → set mean to 0
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X = np.where(nan_mask, col_mean, X)
        print(f"[geno] filled {int(nan_mask.sum())} missing entries with column mean")

    return X, np.asarray(kept_dog_ids), found_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", dest="p_threshold", type=float, default=1.15e-7,
                        help="Keep SNPs with p_wald < this "
                             "(default 1.15e-7 = Bonferroni in Deane-Coe 2018)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Legacy: instead of --p, keep the top-K SNPs by p_wald")
    args = parser.parse_args()

    if args.top_k is not None:
        print(f"=== preprocess_eye.py (top_k={args.top_k}) ===\n")
    else:
        print(f"=== preprocess_eye.py (p_wald < {args.p_threshold:.0e}) ===\n")

    # 1. Phenotype
    pheno = load_phenotype()
    print()

    # 2. GWAS SNP selection
    snp_df = select_snps(args.p_threshold, args.top_k)
    print()

    # 3. Genotype extraction
    X, kept_ids, snp_ids = extract_genotypes(snp_df, pheno["anonymous_id"].tolist())
    print()

    # 4. Align phenotype rows with genotype rows
    pheno_idx = pheno.set_index("anonymous_id").loc[kept_ids]
    y = pheno_idx["blue_eyes"].to_numpy(dtype=np.int64)
    breed = pheno_idx["breed"].fillna("UNKNOWN").to_numpy()
    delta_logr = pheno_idx["delta_logr"].to_numpy(dtype=np.float32)
    haplotype_count = pheno_idx["haplotype_count"].to_numpy(dtype=np.float32)

    # 5. SNP metadata aligned to columns of X
    snp_df_kept = snp_df.set_index("rs").loc[snp_ids].reset_index()
    snp_chrom = snp_df_kept["chr"].to_numpy()
    snp_pos = snp_df_kept["ps"].to_numpy()
    p_wald = snp_df_kept["p_wald"].to_numpy(dtype=np.float64)

    print(f"[final] X.shape = {X.shape}")
    print(f"[final] y balance = {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[final] positive rate = {y.mean():.3%}")

    np.savez(
        OUT_PATH,
        X=X,
        y=y,
        dog_id=kept_ids,
        breed=breed,
        delta_logr=delta_logr,
        haplotype_count=haplotype_count,
        snp_id=np.asarray(snp_ids),
        snp_chrom=snp_chrom,
        snp_pos=snp_pos,
        p_wald=p_wald,
        selection_mode=np.asarray(
            f"top_k={args.top_k}" if args.top_k is not None
            else f"p_wald<{args.p_threshold:.0e}"
        ),
    )
    print(f"\n[final] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
