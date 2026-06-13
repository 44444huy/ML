"""Render coat-color result artifacts without writing markdown.

Reads:
    dog/experiments/coat/<label>/results.json

Writes:
    dog/experiments/coat/<label>/test_metrics.csv
    dog/report/figures/coat_<label>_metric_bars.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "experiments" / "coat"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "Majority",
    "LR",
    "RF",
    "MLP",
    "MLP (tuned)",
    "TabPFN",
    "TabICL",
    "TabNet",
]


def load_results(label: str) -> dict:
    return json.loads((EXP_ROOT / label / "results.json").read_text())


def test_metrics_frame(results: dict) -> pd.DataFrame:
    rows = []
    models = results["models"]
    ordered_methods = [m for m in MODEL_ORDER if m in models]
    ordered_methods += [m for m in models if m not in ordered_methods]
    for method in ordered_methods:
        result = models[method]
        t = result["test"]
        rows.append({
            "method": method,
            "pr_auc": t["pr_auc"],
            "roc_auc": t["roc_auc"],
            "f1": t["f1"],
            "precision": t["precision"],
            "recall": t["recall"],
            "threshold": t.get("threshold", 0.5),
            "n_pos": t["n_pos"],
            "n": t["n"],
        })
    return pd.DataFrame(rows)


def plot_metric_bars(label: str, df: pd.DataFrame) -> Path:
    methods = df["method"].tolist()
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    palette = sns.color_palette("Set2", len(methods))

    for ax, (metric, title) in zip(axes, metrics):
        vals = df[metric].to_numpy()
        ax.bar(methods, vals, color=palette)
        for i, v in enumerate(vals):
            ax.text(i, min(v + 0.015, 1.02), f"{v:.3f}", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)

    fig.suptitle(f"Coat color: {label}")
    fig.tight_layout()
    out_path = FIG_DIR / f"coat_{label}_metric_bars.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="black")
    args = parser.parse_args()

    results = load_results(args.label)
    df = test_metrics_frame(results)
    out_csv = EXP_ROOT / args.label / "test_metrics.csv"
    df.to_csv(out_csv, index=False)
    out_png = plot_metric_bars(args.label, df)

    print(f"[report_coat] wrote {out_csv}")
    print(f"[report_coat] wrote {out_png}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
