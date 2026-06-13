"""Preprocess Darwin's Ark coat-color GWAS data for binary ML tasks.

Pipeline:
    1. Load Q243 coat phenotype labels (multi-label binary columns).
    2. Stream the matching GCTA GWAS output file from the 14.6GB zip.
    3. Select SNPs by p-value (default: paper genome-wide significance,
       P < 5e-8).
    4. Read only those SNP columns from the PLINK bed/bim/fam genotype set.
    5. Save an npz bundle compatible with the eye-color training pipeline.

This intentionally starts with one binary label at a time. That keeps the
comparison with eye color clean: genotype -> one trait label, same metrics.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "coat"
OUT_DIR = ROOT / "data" / "processed" / "coat"
GWAS_CACHE_DIR = OUT_DIR / "gwas_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GWAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PHENO_PATH = RAW_DIR / "DarwinsArk_13Q_Q243_coat_color_N-1930.tsv"
GWAS_ZIP_PATH = RAW_DIR / "darwins_dogs_gwas_output_files.zip"
BED_STEM = RAW_DIR / "DarwinsDogs_2024_N-3277_canfam4_gp-0.70_biallelic"
BIM_PATH = Path(str(BED_STEM) + ".bim")
FAM_PATH = Path(str(BED_STEM) + ".fam")

LABELS = {
    "black": {
        "column": "Q243_black_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_black_coat_color_N-1930.loco.mlma",
    },
    "liver_brown": {
        "column": "Q243_liver_or_brown_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_liver_or_brown_coat_color_N-1930.loco.mlma",
    },
    "white": {
        "column": "Q243_white_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_white_coat_color_N-1930.loco.mlma",
    },
    "red": {
        "column": "Q243_red_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_red_coat_color_N-1930.loco.mlma",
    },
    "yellow": {
        "column": "Q243_yellow_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_yellow_coat_color_N-1930.loco.mlma",
    },
    "grey_blue": {
        "column": "Q243_grey_or_blue_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_grey_or_blue_coat_color_N-1930.loco.mlma",
    },
    "tan": {
        "column": "Q243_tan_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_tan_coat_color_N-1930.loco.mlma",
    },
    "cream": {
        "column": "Q243_cream_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_cream_coat_color_N-1930.loco.mlma",
    },
    "white_cream": {
        "column": "Q243_white_or_cream_coat_color",
        "gwas": "darwins_dogs_gwas_output_files/Q243_white_or_cream_coat_color_N-1930.loco.mlma",
    },
    "red_brown_tan": {
        "column": "Q243_red_or_liver_or_brown_or_tan_coat_color",
        "gwas": (
            "darwins_dogs_gwas_output_files/"
            "Q243_red_or_liver_or_brown_or_tan_coat_color_N-1930.loco.mlma"
        ),
    },
    "single_color": {
        "column": "single_color_in_coat",
        "gwas": "darwins_dogs_gwas_output_files/single_color_in_coat_N-1930.loco.mlma",
    },
}


def output_path(label: str) -> Path:
    return OUT_DIR / f"coat_{label}_processed.npz"


def load_phenotype(label: str) -> pd.DataFrame:
    info = LABELS[label]
    df = pd.read_csv(PHENO_PATH, sep="\t", dtype={"dog_id": str})
    df = df[["dog_id", info["column"]]].dropna().copy()
    df[info["column"]] = df[info["column"]].astype(int)

    print(f"[pheno] label = {label} ({info['column']})")
    print(f"[pheno] rows: {len(df)}")
    print(f"[pheno] balance: {df[info['column']].value_counts().to_dict()}")
    print(f"[pheno] positive rate: {df[info['column']].mean():.3%}")
    return df.reset_index(drop=True)


def _mode_name(top_k: int | None, p_threshold: float | None, max_snps: int | None) -> str:
    if top_k is not None:
        return f"top{top_k}"
    if max_snps is not None:
        return f"p{p_threshold:.0e}_max{max_snps}"
    return f"p{p_threshold:.0e}"


def select_snps(
    label: str,
    *,
    top_k: int | None,
    p_threshold: float | None,
    max_snps: int | None,
    chunksize: int,
    force: bool,
) -> pd.DataFrame:
    """Stream the label's GWAS file from the zip and keep selected SNPs."""
    if top_k is None and p_threshold is None:
        raise ValueError("Either top_k or p_threshold must be set.")
    keep_n = top_k if top_k is not None else max_snps
    mode = _mode_name(top_k, p_threshold, max_snps)
    cache_path = GWAS_CACHE_DIR / f"{label}_{mode}.tsv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, sep="\t")
        print(f"[gwas] loaded cached selection: {cache_path}")
        print(f"[gwas] kept {len(df)} SNPs; best p={df['p'].iloc[0]:.2e}")
        return df

    gwas_member = LABELS[label]["gwas"]
    print(f"[gwas] reading {gwas_member}")
    print(f"[gwas] mode = {mode}")

    selected: pd.DataFrame | None = None
    total_rows = 0
    threshold_rows = 0
    usecols = ["Chr", "SNP", "bp", "A1", "A2", "Freq", "b", "se", "p"]

    with zipfile.ZipFile(GWAS_ZIP_PATH) as zf:
        with zf.open(gwas_member) as fh:
            reader = pd.read_csv(fh, sep="\t", usecols=usecols, chunksize=chunksize)
            for chunk_i, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)
                chunk["p"] = pd.to_numeric(chunk["p"], errors="coerce")
                chunk = chunk.dropna(subset=["p"])

                if p_threshold is not None:
                    chunk = chunk[chunk["p"] < p_threshold]
                    threshold_rows += len(chunk)

                if len(chunk) == 0:
                    continue

                if keep_n is None:
                    selected = chunk if selected is None else pd.concat(
                        [selected, chunk], ignore_index=True
                    )
                else:
                    chunk_top = chunk.nsmallest(min(keep_n, len(chunk)), "p")
                    selected = chunk_top if selected is None else pd.concat(
                        [selected, chunk_top], ignore_index=True
                    )
                    selected = selected.nsmallest(min(keep_n, len(selected)), "p").reset_index(drop=True)

                if chunk_i % 10 == 0:
                    best = selected["p"].iloc[0] if selected is not None else float("nan")
                    print(f"[gwas] chunk {chunk_i}: rows={total_rows:,}, current best p={best:.2e}")

    if selected is None or len(selected) == 0:
        raise RuntimeError(f"No SNPs selected for label={label}, mode={mode}")

    selected = selected.sort_values("p").reset_index(drop=True)
    selected.to_csv(cache_path, sep="\t", index=False)

    print(f"[gwas] total rows scanned: {total_rows:,}")
    if p_threshold is not None:
        print(f"[gwas] rows passing p<{p_threshold:.0e}: {threshold_rows:,}")
    print(f"[gwas] kept {len(selected)} SNPs")
    print(f"[gwas] best: {selected['SNP'].iloc[0]} p={selected['p'].iloc[0]:.2e}")
    print(f"[gwas] worst kept p={selected['p'].iloc[-1]:.2e}")
    print(f"[gwas] cached {cache_path}")
    return selected


def find_snp_indices(wanted_snp_ids: list[str], *, chunksize: int = 1_000_000) -> tuple[list[int], list[str]]:
    """Find .bim column indices for a small selected SNP list.

    The coat .bim has ~10M SNPs. Building a dict for every SNP would waste
    memory, so we stream the .bim and only retain indices for selected SNPs.
    """
    wanted = set(wanted_snp_ids)
    found: dict[str, int] = {}
    offset = 0
    reader = pd.read_csv(BIM_PATH, sep=r"\s+", header=None, usecols=[1],
                         dtype={1: str}, chunksize=chunksize)
    for chunk_i, chunk in enumerate(reader, start=1):
        sids = chunk.iloc[:, 0].astype(str)
        mask = sids.isin(wanted).to_numpy()
        if mask.any():
            positions = np.flatnonzero(mask)
            for pos in positions:
                found[sids.iloc[pos]] = offset + int(pos)
        offset += len(chunk)
        if len(found) == len(wanted):
            break
        if chunk_i % 5 == 0:
            print(f"[bim] scanned {offset:,} SNPs; found {len(found)}/{len(wanted)}")

    found_ids = [sid for sid in wanted_snp_ids if sid in found]
    indices = [found[sid] for sid in found_ids]
    missing = len(wanted_snp_ids) - len(found_ids)
    if missing:
        print(f"[bim] WARNING: {missing}/{len(wanted_snp_ids)} selected SNPs not found in .bim")
    print(f"[bim] found {len(found_ids)} selected SNPs")
    return indices, found_ids


def find_dog_indices(dog_ids_keep: list[str]) -> tuple[list[int], list[str]]:
    fam = pd.read_csv(FAM_PATH, sep=r"\s+", header=None, dtype=str)
    fam_iid = fam.iloc[:, 1].astype(str).tolist()
    iid_to_idx = {iid: i for i, iid in enumerate(fam_iid)}
    dog_indices = [iid_to_idx[d] for d in dog_ids_keep if d in iid_to_idx]
    kept_dog_ids = [d for d in dog_ids_keep if d in iid_to_idx]
    missing = len(dog_ids_keep) - len(kept_dog_ids)
    if missing:
        print(f"[fam] WARNING: {missing}/{len(dog_ids_keep)} phenotyped dogs not found in .fam")
    print(f"[fam] keeping {len(kept_dog_ids)} dogs")
    return dog_indices, kept_dog_ids


def extract_genotypes(snp_df: pd.DataFrame, dog_ids_keep: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    wanted_snp_ids = snp_df["SNP"].astype(str).tolist()
    snp_indices, found_snp_ids = find_snp_indices(wanted_snp_ids)
    dog_indices, kept_dog_ids = find_dog_indices(dog_ids_keep)

    bed = Bed(str(BED_STEM), count_A1=True)
    print(f"[geno] bed shape: {bed.iid_count} dogs x {bed.sid_count} SNPs")
    print(f"[geno] reading submatrix: {len(dog_indices)} dogs x {len(snp_indices)} SNPs")

    sub = bed[dog_indices, snp_indices].read()
    X = sub.val.astype(np.float32)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X = np.where(nan_mask, col_mean, X)
        print(f"[geno] filled {int(nan_mask.sum())} missing values with column mean")

    return X, np.asarray(kept_dog_ids), found_snp_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black", choices=sorted(LABELS),
                        help="Binary coat label to preprocess.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Legacy: keep the top-K GWAS SNPs by p-value instead of --p.")
    parser.add_argument("--p", dest="p_threshold", type=float, default=5e-8,
                        help="P-value threshold mode. Default 5e-8 follows Lord et al. 2025.")
    parser.add_argument("--max_snps", type=int, default=None,
                        help="Optional cap selected SNPs when using --p. Default keeps all passing SNPs.")
    parser.add_argument("--chunksize", type=int, default=500_000,
                        help="GWAS rows per chunk while streaming from zip.")
    parser.add_argument("--force_gwas", action="store_true",
                        help="Ignore cached top-SNP table and rescan the GWAS zip.")
    args = parser.parse_args()

    top_k = None if args.top_k == 0 else args.top_k
    p_threshold = None if top_k is not None else args.p_threshold
    print(f"=== preprocess_coat.py label={args.label} ===\n")

    pheno = load_phenotype(args.label)
    print()

    snp_df = select_snps(
        args.label,
        top_k=top_k,
        p_threshold=p_threshold,
        max_snps=args.max_snps,
        chunksize=args.chunksize,
        force=args.force_gwas,
    )
    print()

    X, kept_ids, snp_ids = extract_genotypes(snp_df, pheno["dog_id"].tolist())
    print()

    info = LABELS[args.label]
    pheno_idx = pheno.set_index("dog_id").loc[kept_ids]
    y = pheno_idx[info["column"]].to_numpy(dtype=np.int64)

    snp_df_kept = snp_df.set_index("SNP").loc[snp_ids].reset_index()
    out_path = output_path(args.label)

    print(f"[final] X.shape = {X.shape}")
    print(f"[final] y balance = {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[final] positive rate = {y.mean():.3%}")

    mode = _mode_name(top_k, p_threshold, args.max_snps)
    np.savez(
        out_path,
        X=X,
        y=y,
        dog_id=kept_ids,
        label=np.asarray(args.label),
        label_column=np.asarray(info["column"]),
        snp_id=snp_df_kept["SNP"].astype(str).to_numpy(),
        snp_chrom=snp_df_kept["Chr"].to_numpy(),
        snp_pos=snp_df_kept["bp"].to_numpy(),
        allele1=snp_df_kept["A1"].astype(str).to_numpy(),
        allele2=snp_df_kept["A2"].astype(str).to_numpy(),
        allele1_freq=snp_df_kept["Freq"].to_numpy(dtype=np.float32),
        beta=snp_df_kept["b"].to_numpy(dtype=np.float32),
        se=snp_df_kept["se"].to_numpy(dtype=np.float32),
        p_value=snp_df_kept["p"].to_numpy(dtype=np.float64),
        selection_mode=np.asarray(mode),
        gwas_file=np.asarray(info["gwas"]),
    )
    print(f"\n[final] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
