"""Render M1/M2 MLP results alongside baselines.

Reads:
    experiments/baselines/baseline_results.json
    experiments/mlp/m1_results.json
    experiments/mlp/m2_results.json

Writes:
    report/mlp.md
    report/figures/08_method_comparison.png
    report/figures/09_mlp_confusion_matrices.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments" / "baselines" / "baseline_results.json"
M1_PATH = ROOT / "experiments" / "mlp" / "m1_results.json"
M2_PATH = ROOT / "experiments" / "mlp" / "m2_results.json"
FIG_DIR = ROOT / "report" / "figures"
REPORT_PATH = ROOT / "report" / "mlp.md"

TRAIT_NAMES = {
    "eye": ["blue", "inter", "brown"],
    "hair": ["blond", "brown", "red", "black", "light", "dark"],
    "skin": ["v_pale", "pale", "inter", "dark", "very_dark"],
}
TRAITS = ["eye", "hair", "skin"]
BASELINES = ["LR", "SVM", "RF"]


def fmt(m, std=None):
    return f"{m:.4f}" + (f" ± {std:.4f}" if std is not None else "")


def load_all() -> dict:
    """Flatten into {method_name: {trait: {"cv": ..., "test": ...}}}."""
    base = json.loads(BASE_PATH.read_text())
    out: dict = {}
    for m in BASELINES:
        out[m] = base[m]
    out["M1_MLP_CE"] = json.loads(M1_PATH.read_text())
    out["M2_MLP_KL"] = json.loads(M2_PATH.read_text())
    return out


def cv_table(all_res: dict) -> list[str]:
    lines = ["| method | trait | accuracy | macro_f1 | auc_ovr | ece |",
             "|---|---|---|---|---|---|"]
    for m in all_res:
        for t in TRAITS:
            cv = all_res[m][t]["cv"]["cv_mean"]
            row = [m, t,
                   fmt(cv["accuracy"]["mean"], cv["accuracy"]["std"]),
                   fmt(cv["macro_f1"]["mean"], cv["macro_f1"]["std"]),
                   fmt(cv["auc_macro_ovr"]["mean"], cv["auc_macro_ovr"]["std"]),
                   fmt(cv["ece"]["mean"], cv["ece"]["std"])]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def test_table(all_res: dict) -> list[str]:
    lines = ["| method | trait | accuracy | macro_f1 | auc_ovr | ece | mae | qwk |",
             "|---|---|---|---|---|---|---|---|"]
    for m in all_res:
        for t in TRAITS:
            r = all_res[m][t]["test"]
            mae = f"{r['mae']:.4f}" if "mae" in r else "—"
            qwk = f"{r['qwk']:.4f}" if "qwk" in r else "—"
            row = [m, t, fmt(r["accuracy"]), fmt(r["macro_f1"]),
                   fmt(r["auc_macro_ovr"]), fmt(r["ece"]), mae, qwk]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def plot_comparison(all_res: dict) -> None:
    """Grouped bar chart of test macro-F1 per method × trait."""
    methods = list(all_res.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TRAITS))
    w = 0.8 / len(methods)
    palette = sns.color_palette("Set2", len(methods))
    for i, m in enumerate(methods):
        vals = [all_res[m][t]["test"]["macro_f1"] for t in TRAITS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=m, color=palette[i])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w - 0.4 + w / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(TRAITS)
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Method comparison — test macro-F1")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_method_comparison.png", dpi=120)
    plt.close(fig)


def plot_mlp_confusions(all_res: dict) -> None:
    """2×3 grid: M1 and M2 test confusion matrices."""
    methods = ["M1_MLP_CE", "M2_MLP_KL"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, m in enumerate(methods):
        for j, t in enumerate(TRAITS):
            cm = np.array(all_res[m][t]["test"]["confusion_matrix"])
            ax = axes[i, j]
            names = TRAIT_NAMES[t]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=names, yticklabels=names, cbar=False)
            ax.set_title(f"{m} / {t}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_mlp_confusion_matrices.png", dpi=120)
    plt.close(fig)


def takeaways(all_res: dict) -> list[str]:
    # Pull numbers for narrative
    def f1(m, t):
        return all_res[m][t]["test"]["macro_f1"]

    best_base = {t: max(BASELINES, key=lambda b: f1(b, t)) for t in TRAITS}
    lines = ["## Key takeaways\n"]
    lines.append(
        "1. **M1 (MLP + CE) vs best baseline.** "
        + "; ".join(
            f"{t}: best baseline {best_base[t]} F1={f1(best_base[t], t):.3f} "
            f"vs M1 F1={f1('M1_MLP_CE', t):.3f}" for t in TRAITS)
        + "."
    )
    lines.append(
        "2. **M2 (MLP + soft-label KL) vs M1.** "
        + "; ".join(
            f"{t}: M1 F1={f1('M1_MLP_CE', t):.3f} vs "
            f"M2 F1={f1('M2_MLP_KL', t):.3f}" for t in TRAITS)
        + ". Soft-label KL exploits HIrisPlex's probability distribution "
        "instead of collapsing it to argmax, so it can learn from samples "
        "that argmax turns into noise."
    )
    lines.append(
        "3. **Eye is still capped at F1 ≈ 2/3.** All methods trained on "
        "HIrisPlex argmax labels inherit the missing `intermediate` class. "
        "Raising eye F1 further requires either re-labelling with soft targets "
        "across the full 3-class output (which M2 does) or introducing "
        "ground-truth phenotypes from a different dataset (e.g. the dog dataset)."
    )
    lines.append(
        "4. **Calibration (ECE) matters for forensic use.** Compare ECE "
        "across methods in the table: deep models can be overconfident on "
        "rare classes, which is a liability when predictions inform "
        "investigations."
    )
    return lines


def main() -> int:
    all_res = load_all()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    parts = ["# MLP vs Baselines\n",
             "Identical preprocessing, splits, and metric suite as Week 2. "
             "Two MLP variants share an architecture and differ only in loss:\n",
             "- **M1**: CrossEntropy on argmax(HIrisPlex) labels.",
             "- **M2**: KL-divergence on raw HIrisPlex probability vectors (soft label).\n",
             "Hyperparameters (fixed): 2 hidden layers × 128, dropout=0.3, "
             "Adam lr=1e-3, wd=1e-4, batch=64, early stopping on val macro-F1 "
             "with patience=20.\n",
             "## CV validation (mean ± std across 5 folds)\n"]
    parts += cv_table(all_res)
    parts += ["\n## Test metrics (refit on full train+val)\n"]
    parts += test_table(all_res)
    parts += ["\n"]
    parts += takeaways(all_res)
    parts += ["\nFigures:\n"
              "- `figures/08_method_comparison.png` — test macro-F1 bar chart.\n"
              "- `figures/09_mlp_confusion_matrices.png` — M1/M2 confusion matrices.\n"]

    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")
    plot_comparison(all_res)
    plot_mlp_confusions(all_res)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FIG_DIR / '08_method_comparison.png'}")
    print(f"Wrote {FIG_DIR / '09_mlp_confusion_matrices.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
