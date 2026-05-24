"""Visualization utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix (row-normalised)",
) -> None:
    """Save a row-normalised confusion matrix heatmap.

    Args:
        confusion_matrix: 23×23 nested list of int counts (rows=true, cols=pred).
        class_names: ordered list of class name strings, length == matrix side.
        output_path: PNG path to write.
        title: figure title.
    """
    import matplotlib.pyplot as plt  # imported here — backend set by caller

    cm = np.array(confusion_matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with no test samples
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    n = len(class_names)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(class_names, fontsize=6.5)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)

    # Annotate only the diagonal (correct predictions) to keep the chart readable
    for i in range(n):
        val = cm_norm[i, i]
        color = "white" if val > 0.6 else "black"
        ax.text(i, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(
    per_class: dict[str, dict],
    class_names: list[str],
    output_path: Path,
    title: str = "Per-class F1 (sorted descending)",
    low_f1_threshold: float = 0.3,
) -> None:
    """Save a horizontal bar chart of per-class F1 scores.

    Bars with F1 < low_f1_threshold are coloured red to flag problem classes.

    Args:
        per_class: dict keyed by str label index, values have 'f1' and 'support'.
        class_names: ordered list of class name strings.
        output_path: PNG path to write.
        title: figure title.
        low_f1_threshold: F1 below this value is highlighted in red.
    """
    import matplotlib.pyplot as plt

    # Build sorted records
    records = []
    for idx_str, metrics in per_class.items():
        idx = int(idx_str)
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        f1 = float(metrics.get("f1", 0.0))
        support = int(metrics.get("support", 0))
        records.append((name, f1, support))

    records.sort(key=lambda r: r[1])  # ascending so worst is at bottom

    names = [f"{r[0]}  (n={r[2]})" for r in records]
    f1s = [r[1] for r in records]
    colors = ["#d62728" if f < low_f1_threshold else "#4C72B0" for f in f1s]

    fig, ax = plt.subplots(figsize=(9, max(6, len(records) * 0.38)))
    bars = ax.barh(names, f1s, color=colors, alpha=0.85)

    # Value labels at bar ends
    for bar, val in zip(bars, f1s):
        ax.text(
            min(val + 0.01, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axvline(low_f1_threshold, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"F1 = {low_f1_threshold} threshold")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
