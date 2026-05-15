"""Plot the K-sensitivity results as a diagram for the report.

Reads:
    experiments/eye/k_sensitivity.json   (produced by experiments/compare_k.py)

Writes:
    report/figures/03_k_sensitivity.png

The figure shows MLP performance vs number of SNPs for both the
5-fold CV-mean and the held-out test set, with a vertical line at
K=56 marking the default Bonferroni cutoff (p < 1.15e-7) used by
Deane-Coe et al. 2018 on this dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "experiments" / "eye" / "k_sensitivity.json"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "03_k_sensitivity.png"


def main() -> int:
    if not JSON_PATH.exists():
        print(f"[plot_k] missing {JSON_PATH}. Run compare_k.py first.")
        return 1
    rows = json.loads(JSON_PATH.read_text())
    rows = sorted(rows, key=lambda r: r["n_snps"])

    # Back-compat: the older JSON only had `pr_auc`/`roc_auc`/`f1`
    # (which were CV values); the new schema has explicit
    # `cv_*` and `test_*`. Fall back gracefully.
    def get(r, key, default=None):
        return r.get(key, r.get(default))

    ns = np.array([r["n_snps"] for r in rows])
    labels = [r["config"] for r in rows]

    cv_pr = np.array([get(r, "cv_pr_auc", "pr_auc") for r in rows])
    cv_rc = np.array([get(r, "cv_roc_auc", "roc_auc") for r in rows])
    cv_f1 = np.array([get(r, "cv_f1", "f1") for r in rows])
    has_test = all("test_pr_auc" in r for r in rows)
    if has_test:
        te_pr = np.array([r["test_pr_auc"] for r in rows])
        te_rc = np.array([r["test_roc_auc"] for r in rows])
        te_f1 = np.array([r["test_f1"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ["PR-AUC", "ROC-AUC", "F1"]
    cv_series = [cv_pr, cv_rc, cv_f1]
    te_series = [te_pr, te_rc, te_f1] if has_test else [None, None, None]

    default_n = 56
    for ax, title, cv, te in zip(axes, titles, cv_series, te_series):
        ax.plot(ns, cv, marker="o", lw=2, color="tab:blue", label="CV (5-fold mean)")
        if te is not None:
            ax.plot(ns, te, marker="s", lw=2, color="tab:red", label="Test (held-out)")
        ax.axvline(default_n, color="black", ls="--", lw=1, alpha=0.6)
        ax.text(default_n * 1.06, ax.get_ylim()[0] + 0.02,
                "default\n(p<1.15e-7)\nK=56",
                fontsize=8, color="black", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Number of SNPs (log scale)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9)

        # Annotate #SNPs on the CV line
        for x, y in zip(ns, cv):
            ax.annotate(f"{x}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8,
                        color="tab:blue")

    fig.suptitle(
        "K sensitivity — feature-selection cutoff vs MLP performance "
        "(eye color). Default highlighted as dashed line.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_k] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
