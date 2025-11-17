import csv
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)


def save_table(path: Path, header: List[str], rows: List[List]):
    """Save a CSV table with a header."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_score_comparison(scores: List[Dict[str, float]]):
    labels = [row["model"] for row in scores]
    en_scores = [row["em_en"] for row in scores]
    zh_scores = [row["em_zh"] for row in scores]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, en_scores, width, label="English EM", color="#6baed6")
    ax.bar(x + width / 2, zh_scores, width, label="Chinese EM", color="#fd8d3c")

    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Answer Accuracy Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "score_comparison.png", dpi=200)
    plt.close(fig)


def plot_training_curve(curves: Dict[str, List[float]]):
    steps = np.arange(len(curves["train"]))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(steps, curves["train"], label="Train Loss", color="#3182bd")
    ax.plot(steps, curves["dev"], label="Dev Loss", color="#e6550d")

    ax.set_xlabel("Steps (x1k)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Dynamics")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "training_curve.png", dpi=200)
    plt.close(fig)


def plot_ablation(ablations: List[Dict[str, float]]):
    labels = [row["name"] for row in ablations]
    deltas = [row["delta"] for row in ablations]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, deltas, color="#31a354")
    ax.set_xlabel("Δ Overall Score vs. Full (points)")
    ax.set_title("Ablation Impact")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.bar_label(bars, fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(ROOT / "ablation_impact.png", dpi=200)
    plt.close(fig)


def plot_codeswitch(robustness: Dict[str, List[float]]):
    rates = robustness["rates"]
    en = robustness["en"]
    zh = robustness["zh"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates, en, marker="o", label="English EM", color="#3182bd")
    ax.plot(rates, zh, marker="o", label="Chinese EM", color="#e6550d")

    ax.set_xlabel("Code-switch ratio (%)")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Robustness to Code-Switching")
    ax.set_xticks(rates)
    ax.set_ylim(60, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "code_switch_robustness.png", dpi=200)
    plt.close(fig)


def plot_entity_heatmap(matrix: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.6, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Entity Agreement by Category")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.tight_layout()
    fig.savefig(ROOT / "entity_agreement_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    score_rows = [
        {"model": "EN Fine-tune", "em_en": 70.2, "em_zh": 0.0},
        {"model": "ZH Fine-tune", "em_en": 0.0, "em_zh": 67.8},
        {"model": "Bilingual Baseline", "em_en": 74.5, "em_zh": 72.1},
        {"model": "I-QGP (ours)", "em_en": 82.9, "em_zh": 84.1},
    ]

    save_table(
        ROOT / "score_comparison.csv",
        ["Model", "EN EM", "ZH EM", "Bilingual Agreement", "Graph F1", "Entity Jaccard"],
        [
            ["EN Fine-tune", 70.2, 41.8, 39.5, 58.2, 0.42],
            ["ZH Fine-tune", 49.6, 67.8, 44.1, 55.6, 0.47],
            ["Bilingual Baseline", 74.5, 72.1, 71.0, 64.8, 0.58],
            ["I-QGP (ours)", 82.9, 84.1, 85.2, 71.4, 0.71],
        ],
    )

    plot_score_comparison(score_rows)

    curve_points = 12
    train_loss = np.linspace(2.6, 1.1, curve_points) + np.random.normal(0, 0.03, curve_points)
    dev_loss = np.linspace(2.7, 1.25, curve_points) + np.random.normal(0, 0.02, curve_points)
    plot_training_curve({"train": train_loss.tolist(), "dev": dev_loss.tolist()})

    ablations = [
        {"name": "-EMD Alignment", "delta": -3.8},
        {"name": "-VQ Planner", "delta": -5.4},
        {"name": "-Entity Eraser", "delta": -2.6},
        {"name": "-CSD", "delta": -1.9},
        {"name": "-QID Anchoring", "delta": -4.1},
    ]
    save_table(
        ROOT / "ablations.csv",
        ["Ablation", "EN EM", "ZH EM", "Agreement", "Δ Overall"],
        [
            ["-EMD Alignment", 79.2, 80.4, 80.1, -3.8],
            ["-VQ Planner", 77.5, 78.1, 76.3, -5.4],
            ["-Entity Eraser", 81.3, 82.2, 82.0, -2.6],
            ["-CSD", 81.7, 82.0, 82.1, -1.9],
            ["-QID Anchoring", 78.6, 79.4, 78.8, -4.1],
        ],
    )
    plot_ablation(ablations)

    robustness = {
        "rates": [0, 10, 20, 30],
        "en": [82.9, 81.5, 80.4, 78.7],
        "zh": [84.1, 83.0, 81.6, 80.2],
    }
    save_table(
        ROOT / "code_switch_stress.csv",
        ["Code-switch %", "EN EM", "ZH EM", "Bilingual Agreement"],
        [
            [0, 82.9, 84.1, 85.2],
            [10, 81.5, 83.0, 84.5],
            [20, 80.4, 81.6, 83.1],
            [30, 78.7, 80.2, 82.4],
        ],
    )
    plot_codeswitch(robustness)

    categories = ["Math", "Commonsense", "Science", "QA"]
    matrix = np.array(
        [
            [0.78, 0.74, 0.69, 0.72],
            [0.75, 0.81, 0.71, 0.76],
            [0.72, 0.73, 0.79, 0.74],
            [0.74, 0.77, 0.72, 0.82],
        ]
    )
    plot_entity_heatmap(matrix, categories)

    save_table(
        ROOT / "agreement_breakdown.csv",
        ["Category", "Entity Jaccard", "Unit Match", "Graph F1"],
        [
            ["Math", 0.78, 0.83, 0.74],
            ["Commonsense", 0.81, 0.79, 0.77],
            ["Science", 0.79, 0.82, 0.73],
            ["QA", 0.82, 0.84, 0.78],
        ],
    )

    save_table(
        ROOT / "training_stability.csv",
        ["Seed", "EN EM", "ZH EM", "Agreement", "Graph F1"],
        [
            [0, 82.9, 84.1, 85.2, 71.4],
            [1, 83.1, 83.7, 84.6, 71.0],
            [2, 82.4, 83.9, 84.8, 70.9],
        ],
    )

    summary = ROOT / "README.md"
    summary.write_text(
        """# Experiment-style Results

This directory contains synthetic figures and tables that mimic the outcome of I-QGP bilingual reasoning experiments. All values are illustrative and generated from the lightweight `generate_results.py` script.

## Contents
- `score_comparison.png`: English/Chinese exact-match accuracy across baselines and the proposed I-QGP model.
- `training_curve.png`: Training vs. dev loss trajectory over optimization steps.
- `ablation_impact.png`: Overall score deltas when key components are removed.
- `code_switch_robustness.png`: Accuracy under varying code-switch ratios.
- `entity_agreement_heatmap.png`: Agreement statistics by category (entity alignment surrogate).
- CSV tables (`*.csv`) mirroring the plotted metrics, plus seed stability and agreement breakdowns.

To regenerate all assets, run:

```bash
python generate_results.py
```
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
