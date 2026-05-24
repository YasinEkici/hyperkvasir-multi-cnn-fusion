"""Generate comparison bar chart, training curves, confusion matrices, and per-class F1 chart."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.evaluation.visualization import plot_confusion_matrix, plot_per_class_f1


# ---------------------------------------------------------------------------
# Class name lookup
# ---------------------------------------------------------------------------

def _load_class_names(splits_dir: Path) -> list[str]:
    """Return class names sorted by label index from any fold manifest."""
    manifest = splits_dir / "hyperkvasir_official_5fold" / "fold_0.csv"
    if not manifest.exists():
        return []
    df = pd.read_csv(manifest)
    mapping = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
    )
    return mapping["class_name"].tolist()


# ---------------------------------------------------------------------------
# Run loader
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    method = config.get("method", {})
    training = config.get("training", {})
    backbone_names = method.get("backbone_names", [])
    fusion_type = method.get("fusion_type", "none")
    unfreeze_blocks = int(training.get("unfreeze_blocks", 0))
    mode = "frozen" if unfreeze_blocks == 0 else f"ft-{unfreeze_blocks}blk"

    short_label = "+".join(b[0].upper() for b in backbone_names)
    if fusion_type not in ("none", ""):
        short_label += f"\n{fusion_type}"
    short_label += f"\n({mode})"

    test = metrics.get("test", {})
    return {
        "id": run_dir.name,
        "run_dir": run_dir,
        "label": short_label,
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "history": metrics.get("history", []),
        "per_class": test.get("per_class", {}),
        "confusion_matrix": test.get("confusion_matrix"),
    }


# ---------------------------------------------------------------------------
# Plot 1 — Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(rows: list[dict], output_path: Path) -> None:
    rows = [r for r in rows if r["accuracy"] is not None]
    if not rows:
        print("No data for comparison chart.")
        return

    labels = [r["label"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.4), 5))
    bars_acc = ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="#4C72B0", alpha=0.88)
    bars_f1 = ax.bar([i + width / 2 for i in x], f1s, width, label="Macro F1", color="#DD8452", alpha=0.88)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison — Accuracy and Macro F1 (fold 0, official split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_acc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=6.5,
        )
    for bar in bars_f1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=6.5,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart -> {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(rows: list[dict], output_path: Path) -> None:
    rows = [r for r in rows if r["history"]]
    if not rows:
        print("No history data for training curves.")
        return

    n = len(rows)
    cols = min(3, n)
    row_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(row_count, cols, figsize=(5 * cols, 3.5 * row_count), squeeze=False)

    for idx, run in enumerate(rows):
        ax = axes[idx // cols][idx % cols]
        history = run["history"]
        epochs = [h["epoch"] for h in history]
        val_f1 = [h.get("val_macro_f1") for h in history]
        val_acc = [h.get("val_accuracy") for h in history]
        train_loss = [h.get("train_loss") for h in history]

        ax2 = ax.twinx()
        if any(v is not None for v in train_loss):
            ax2.plot(epochs, train_loss, color="grey", linewidth=0.9, linestyle="--", label="train loss", alpha=0.7)
            ax2.set_ylabel("Loss", fontsize=7, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=6)

        if any(v is not None for v in val_acc):
            ax.plot(epochs, val_acc, color="#4C72B0", linewidth=1.3, label="val acc")
        if any(v is not None for v in val_f1):
            ax.plot(epochs, val_f1, color="#DD8452", linewidth=1.3, label="val macro F1")

        ax.set_title(run["id"].replace("_official", ""), fontsize=7.5)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Score", fontsize=7)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.25)

    for idx in range(n, row_count * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Training Curves — val accuracy and macro F1", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves -> {output_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Confusion matrices (one per experiment, saved inside run dir)
# ---------------------------------------------------------------------------

def plot_all_confusion_matrices(rows: list[dict], class_names: list[str]) -> None:
    if not class_names:
        print("Warning: class names not available — skipping confusion matrices.")
        return

    for run in rows:
        cm = run.get("confusion_matrix")
        if cm is None:
            print(f"  No confusion_matrix in metrics.json for {run['id']} — skipping.")
            continue
        out_path = run["run_dir"] / "confusion_matrix.png"
        title = f"Confusion Matrix — {run['id'].replace('_official', '')}"
        plot_confusion_matrix(cm, class_names, out_path, title=title)
        print(f"Saved confusion matrix -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Per-class F1 chart for the best experiment
# ---------------------------------------------------------------------------

def plot_best_per_class_f1(rows: list[dict], class_names: list[str], output_path: Path) -> None:
    if not class_names:
        print("Warning: class names not available — skipping per-class F1 chart.")
        return

    candidates = [r for r in rows if r.get("macro_f1") is not None and r["per_class"]]
    if not candidates:
        print("No per_class data found — skipping per-class F1 chart.")
        return

    best = max(candidates, key=lambda r: r["macro_f1"])
    print(f"Per-class F1 chart: using best experiment '{best['id']}' (macro F1={best['macro_f1']:.4f})")

    title = f"Per-class F1 — {best['id'].replace('_official', '')} (macro F1={best['macro_f1']:.3f})"
    plot_per_class_f1(best["per_class"], class_names, output_path, title=title)
    print(f"Saved per-class F1 chart -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result plots.")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment IDs to include (default: all completed runs)",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Also generate confusion matrices per run and per-class F1 chart for the best run",
    )
    args = parser.parse_args()

    runs_dir = _ROOT / args.runs_dir
    output_dir = _ROOT / args.output_dir
    splits_dir = _ROOT / "data" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_run_dirs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if args.experiments:
        all_run_dirs = [d for d in all_run_dirs if d.name in args.experiments]

    rows = []
    for run_dir in all_run_dirs:
        row = _load_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No completed runs found.")
        return

    # Exclude Stage 0 smoke test duplicate from comparison bar chart only
    exclude_from_chart = {"04_triple_concat_frozen_official"}
    chart_rows = [r for r in rows if r["id"] not in exclude_from_chart]

    plot_comparison(chart_rows, output_dir / "comparison_bar_chart.png")
    plot_training_curves(rows, output_dir / "training_curves.png")

    if args.confusion_matrix:
        class_names = _load_class_names(splits_dir)
        if not class_names:
            print("Warning: could not load class names from fold manifest.")
        plot_all_confusion_matrices(rows, class_names)
        plot_best_per_class_f1(chart_rows, class_names, output_dir / "per_class_f1_best.png")


if __name__ == "__main__":
    main()
