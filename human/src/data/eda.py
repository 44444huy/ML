"""Exploratory Data Analysis for the EVC dataset.

Generates figures into report/figures/ and a markdown summary into report/eda.md.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import build_dataset, TRAIT_NAMES, NUM_SNPS

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def fig_label_distribution(data: dict) -> None:
    """Bar chart of class counts per trait (hard labels)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, trait in zip(axes, ["eye", "hair", "skin"]):
        y = data[f"y_{trait}"]
        names = TRAIT_NAMES[trait]
        counts = np.bincount(y, minlength=len(names))
        ax.bar(range(len(names)), counts, color=sns.color_palette("viridis", len(names)))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(f"{trait.capitalize()} — class counts (argmax)")
        for i, c in enumerate(counts):
            ax.text(i, c + max(counts) * 0.01, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_label_distribution.png", dpi=120)
    plt.close(fig)


def fig_confidence_histograms(data: dict) -> None:
    """Histogram of max p_value per trait — confidence distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, trait in zip(axes, ["eye", "hair", "skin"]):
        conf = data[f"conf_{trait}"]
        ax.hist(conf, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(0.6, color="red", ls="--", label="0.6")
        ax.axvline(0.7, color="orange", ls="--", label="0.7")
        ax.set_title(f"{trait.capitalize()} — max p_value")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_confidence_histograms.png", dpi=120)
    plt.close(fig)


def fig_low_conf_bar(data: dict) -> None:
    """Bars: % low-confidence at thresholds 0.5/0.6/0.7/0.8."""
    thresholds = [0.5, 0.6, 0.7, 0.8]
    traits = ["eye", "hair", "skin"]
    rows = []
    for t in traits:
        conf = data[f"conf_{t}"]
        for th in thresholds:
            frac = (conf < th).mean()
            rows.append({"trait": t, "threshold": th, "pct": frac * 100})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="threshold", y="pct", hue="trait", ax=ax, palette="Set2")
    ax.set_title("% samples with confidence < threshold")
    ax.set_ylabel("% low-confidence")
    ax.set_xlabel("threshold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_low_confidence_by_threshold.png", dpi=120)
    plt.close(fig)
    return df


def fig_soft_label_heatmap(data: dict, n_samples: int = 100) -> None:
    """Heatmap of soft labels (p_value) for a random sample of subjects."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data["X"]), size=n_samples, replace=False)
    idx.sort()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, trait in zip(axes, ["eye", "hair", "skin"]):
        probs = data[f"p_{trait}"][idx]
        sns.heatmap(probs, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                    xticklabels=TRAIT_NAMES[trait], yticklabels=False,
                    cbar_kws={"label": "p_value"})
        ax.set_title(f"{trait.capitalize()} soft labels (100 random samples)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_soft_label_heatmap.png", dpi=120)
    plt.close(fig)


def fig_snp_distribution(data: dict) -> None:
    """Per-SNP dosage (0/1/2) distribution."""
    X = data["X"]
    fig, ax = plt.subplots(figsize=(15, 4))
    counts = np.stack([np.bincount(X[:, j], minlength=3) for j in range(NUM_SNPS)])
    bottom = np.zeros(NUM_SNPS)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for d in range(3):
        ax.bar(range(NUM_SNPS), counts[:, d], bottom=bottom,
               color=colors[d], label=f"dosage={d}")
        bottom += counts[:, d]
    ax.set_xticks(range(NUM_SNPS))
    ax.set_xticklabels([f"s{j}" for j in range(NUM_SNPS)], rotation=90, fontsize=7)
    ax.set_title("SNP dosage distribution (stacked)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_snp_dosage_distribution.png", dpi=120)
    plt.close(fig)


def fig_snp_correlation(data: dict) -> None:
    """Pairwise Pearson correlation among 41 SNPs — detect linkage."""
    X = data["X"].astype(np.float32)
    # SNPs with zero variance produce NaN correlation; fill with 0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title("SNP-SNP Pearson correlation (41×41)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_snp_correlation.png", dpi=120)
    plt.close(fig)


def write_summary(data: dict, lowconf_df: pd.DataFrame) -> None:
    lines = ["# EDA Summary\n"]
    N = len(data["X"])
    lines.append(f"**N = {N} samples, {NUM_SNPS} SNPs**\n")

    lines.append("\n## Class distribution (argmax labels)\n")
    for trait in ["eye", "hair", "skin"]:
        y = data[f"y_{trait}"]
        names = TRAIT_NAMES[trait]
        counts = np.bincount(y, minlength=len(names))
        lines.append(f"- **{trait}**: " + ", ".join(
            f"{n}={c} ({c/N*100:.1f}%)" for n, c in zip(names, counts)))

    lines.append("\n## Low-confidence rates\n")
    pivot = lowconf_df.pivot(index="threshold", columns="trait", values="pct").round(2)
    header = "| threshold | " + " | ".join(pivot.columns) + " |"
    sep = "|" + "---|" * (len(pivot.columns) + 1)
    rows = [f"| {idx} | " + " | ".join(f"{v:.2f}" for v in row) + " |"
            for idx, row in zip(pivot.index, pivot.values)]
    lines.append("\n".join([header, sep] + rows))

    lines.append("\n## Key observations\n")
    lines.append("1. **Argmax collapses classes.** Eye `intermediate`, Hair `blond`/`black`, Skin `pale` "
                 "almost never win argmax, though HIrisPlex assigns them non-trivial probabilities.")
    lines.append("2. **Eye confidence is poor**: majority of samples have max p_value < 0.6.")
    lines.append("3. **Hair / Skin are imbalanced**: one class dominates (>60%).")
    lines.append("4. **SNPs show correlation blocks** — expected from linkage disequilibrium "
                 "(neighboring SNPs inherited together).")

    (ROOT / "report" / "eda.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    data = build_dataset(ROOT / "hirisplex_results_FN_v2.csv")

    fig_label_distribution(data)
    fig_confidence_histograms(data)
    lowconf = fig_low_conf_bar(data)
    fig_soft_label_heatmap(data)
    fig_snp_distribution(data)
    fig_snp_correlation(data)
    write_summary(data, lowconf)

    print("EDA complete.")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Summary: {ROOT / 'report' / 'eda.md'}")


if __name__ == "__main__":
    main()
