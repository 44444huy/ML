"""Render baseline_results.json into a markdown summary and confusion-matrix
figures for the report."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RES_PATH = ROOT / "experiments" / "baselines" / "baseline_results.json"
FIG_DIR = ROOT / "report" / "figures"
REPORT_PATH = ROOT / "report" / "baselines.md"

TRAIT_NAMES = {
    "eye": ["blue", "inter", "brown"],
    "hair": ["blond", "brown", "red", "black", "light", "dark"],
    "skin": ["v_pale", "pale", "inter", "dark", "very_dark"],
}
TRAITS = ["eye", "hair", "skin"]
MODELS = ["LR", "SVM", "RF"]


def fmt(m, std=None):
    return f"{m:.4f}" + (f" ± {std:.4f}" if std is not None else "")


def plot_confusions(results: dict):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for i, model in enumerate(MODELS):
        for j, trait in enumerate(TRAITS):
            cm = np.array(results[model][trait]["test"]["confusion_matrix"])
            ax = axes[i, j]
            names = TRAIT_NAMES[trait]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=names, yticklabels=names, cbar=False)
            ax.set_title(f"{model} / {trait}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_baseline_confusion_matrices.png", dpi=120)
    plt.close(fig)


def render_markdown(results: dict):
    lines = ["# Baseline Results\n"]
    lines.append("Three classifiers (Logistic Regression, SVM-RBF, Random Forest) "
                 "trained per trait with balanced class weights, 5-fold stratified CV "
                 "on the train+val split, then refit and scored on the held-out 20% test.\n")

    lines.append("\n## CV validation (mean ± std across 5 folds)\n")
    lines.append("| model | trait | accuracy | macro_f1 | auc_ovr | ece |")
    lines.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for trait in TRAITS:
            cv = results[model][trait]["cv"]["cv_mean"]
            row = [model, trait,
                   fmt(cv["accuracy"]["mean"], cv["accuracy"]["std"]),
                   fmt(cv["macro_f1"]["mean"], cv["macro_f1"]["std"]),
                   fmt(cv["auc_macro_ovr"]["mean"], cv["auc_macro_ovr"]["std"]),
                   fmt(cv["ece"]["mean"], cv["ece"]["std"])]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Test metrics (refit on full train+val)\n")
    lines.append("| model | trait | accuracy | macro_f1 | auc_ovr | ece | mae | qwk |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for trait in TRAITS:
            t = results[model][trait]["test"]
            mae = f"{t['mae']:.4f}" if "mae" in t else "—"
            qwk = f"{t['qwk']:.4f}" if "qwk" in t else "—"
            row = [model, trait, fmt(t["accuracy"]), fmt(t["macro_f1"]),
                   fmt(t["auc_macro_ovr"]), fmt(t["ece"]), mae, qwk]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Key takeaways\n")
    lines.append("1. **Accuracy is misleading.** All three models reach 92–100% accuracy "
                 "on every trait, yet macro-F1 sits at 0.42–0.67 because rare classes "
                 "(hair `red`, skin `pale`, eye `intermediate`) are collapsed by argmax "
                 "and therefore never predicted.")
    lines.append("2. **Hair and Skin are far from solved.** Even the best model reaches "
                 "only macro-F1 ≈ 0.55; ordinal information (intended order of classes) "
                 "is being ignored because CE loss treats them as categorical.")
    lines.append("3. **Eye collapses to 2 classes.** The `intermediate` class never "
                 "wins argmax in HIrisPlex output, so it is absent from train/test. "
                 "Any classifier trained on these labels cannot predict it.")
    lines.append("4. **Baselines essentially replicate HIrisPlex.** They map 41 SNPs "
                 "to the same kind of output HIrisPlex produced — by construction they "
                 "cannot exceed the silver standard.")
    lines.append("\nConfusion matrices: `figures/07_baseline_confusion_matrices.png`.\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    results = json.loads(RES_PATH.read_text())
    plot_confusions(results)
    render_markdown(results)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote figure: {FIG_DIR/'07_baseline_confusion_matrices.png'}")


if __name__ == "__main__":
    main()
