"""K sensitivity analysis for SNP feature selection.

Why this exists:
    The main pipeline uses `p_wald < 1.15e-7` (Bonferroni cutoff used
    by Deane-Coe et al. 2018 on this same dataset: 0.05 / ~430k
    QC-passed markers) — a principled, citation-backed cutoff that
    yields 56 SNPs.

    This script proves the choice was not arbitrary by sweeping multiple
    K-values (and equivalent p-thresholds) and reporting both CV-mean
    and held-out-test PR-AUC / ROC-AUC / F1 for the proposed MLP. The
    output table goes into the report as an appendix so the reader can
    see the full K-vs-performance curve.

Usage:
    python dog/src/experiments/compare_k.py

Output:
    experiments/eye/k_sensitivity.json
    experiments/eye/k_sensitivity.md   (human-readable table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.preprocess_eye import (  # noqa: E402
    load_phenotype, select_snps, extract_genotypes,
)
from data.splits import load_splits  # noqa: E402
from evaluation.metrics import evaluate, aggregate_folds  # noqa: E402
from train.train_eye import train_one, predict, standardize  # noqa: E402

EXP_DIR = ROOT / "experiments" / "eye"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Configs: (label, p_threshold, top_k)
# Mix of statistical thresholds and ML-style top-K cuts.
CONFIGS = [
    ("p<1.15e-7 (Deane-Coe Bonferroni, default)", 1.15e-7, None),
    ("p<5e-8 (genome-wide standard)",             5e-8,    None),
    ("p<1e-5 (suggestive)",                       1e-5,    None),
    ("p<1e-4 (lenient)",                          1e-4,    None),
    ("top_k=100",                                 None,    100),
    ("top_k=200",                                 None,    200),
    ("top_k=500",                                 None,    500),
]


def run_mlp_cv(X: np.ndarray, y: np.ndarray, splits: dict, device) -> dict:
    fold_results = []
    for fold in splits["folds"]:
        tr, va = fold["train"], fold["valid"]
        X_tr, X_va = standardize(X[tr], X[va])
        model = train_one(X_tr, y[tr], X_va, y[va], device)
        prob, pred = predict(model, X_va, device)
        fold_results.append(evaluate(y[va], pred, prob))
    return aggregate_folds(fold_results)


def run_mlp_test(X: np.ndarray, y: np.ndarray, splits: dict, device) -> dict:
    """Refit on full trainval (90/10 internal val for early stop), score test."""
    tv, te = splits["trainval"], splits["test"]
    X_tv, X_te = standardize(X[tv], X[te])

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(tv))
    cut = int(0.9 * len(tv))
    idx_tr, idx_va = perm[:cut], perm[cut:]

    model = train_one(X_tv[idx_tr], y[tv][idx_tr],
                      X_tv[idx_va], y[tv][idx_va], device)
    prob, pred = predict(model, X_te, device)
    return evaluate(y[te], pred, prob)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare_k] device = {device}\n")

    pheno = load_phenotype()
    splits = load_splits()
    dog_ids_pheno = pheno["anonymous_id"].tolist()

    rows = []
    for label, p_thr, top_k in CONFIGS:
        print(f"\n=== {label} ===")
        snp_df = select_snps(p_thr, top_k)
        X, kept_ids, _ = extract_genotypes(snp_df, dog_ids_pheno)

        # Re-align y to kept_ids order (extract_genotypes preserves the
        # input order, filtering out missing dogs).
        y_map = dict(zip(pheno["anonymous_id"], pheno["blue_eyes"].astype(int)))
        y = np.array([y_map[d] for d in kept_ids], dtype=np.int64)
        X = X.astype(np.float32)

        cv = run_mlp_cv(X, y, splits, device)
        te = run_mlp_test(X, y, splits, device)
        rows.append({
            "config": label,
            "n_snps": int(X.shape[1]),
            "cv_pr_auc": cv["pr_auc"]["mean"],
            "cv_roc_auc": cv["roc_auc"]["mean"],
            "cv_f1": cv["f1"]["mean"],
            "test_pr_auc": te["pr_auc"],
            "test_roc_auc": te["roc_auc"],
            "test_f1": te["f1"],
            # Back-compat: keep these for code that still expects CV-only keys
            "pr_auc": cv["pr_auc"]["mean"],
            "roc_auc": cv["roc_auc"]["mean"],
            "f1": cv["f1"]["mean"],
        })
        print(f"  n_snps={X.shape[1]:4d}  "
              f"CV: PR={cv['pr_auc']['mean']:.4f} ROC={cv['roc_auc']['mean']:.4f} F1={cv['f1']['mean']:.3f}  "
              f"|  TEST: PR={te['pr_auc']:.4f} ROC={te['roc_auc']:.4f} F1={te['f1']:.3f}")

    # Save JSON
    out_json = EXP_DIR / "k_sensitivity.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\n[compare_k] saved {out_json}")

    # Markdown table
    md = ["# K sensitivity (MLP)", "",
          "All numbers below come from the same fixed splits "
          "(seed=42). CV = 5-fold stratified CV-mean on trainval. "
          "TEST = refit on full trainval, evaluated on the held-out 20 % test set.", ""]
    md.append("| Config | #SNPs | CV PR-AUC | CV ROC-AUC | CV F1 | TEST PR-AUC | TEST ROC-AUC | TEST F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['config']} | {r['n_snps']} | "
            f"{r['cv_pr_auc']:.3f} | {r['cv_roc_auc']:.3f} | {r['cv_f1']:.3f} | "
            f"{r['test_pr_auc']:.3f} | {r['test_roc_auc']:.3f} | {r['test_f1']:.3f} |"
        )
    md.append("")
    md.append("Default in main pipeline: **`p < 1.15e-7`** "
              "(Bonferroni cutoff used by Deane-Coe et al. 2018 on "
              "this dataset: 0.05 / ~430k QC-passed markers).")
    out_md = EXP_DIR / "k_sensitivity.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[compare_k] saved {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
