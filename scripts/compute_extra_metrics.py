"""Compute MCC, micro/macro/weighted P/R/F1 from saved fold predictions.

No training required: operates purely on the `predictions.npz` files written by
train.py. Concatenates predictions across folds (same pooling convention as
compute_ci.py / D-05) so the reported metrics are on the full test pool.

Run-dir convention (must match train.py --fold logic):
  fold 0 : results/runs/{id}/
  fold k>=1: results/runs/{id}_fold_{k}/

Note: for single-label multiclass, micro-F1 == accuracy by definition; both are
reported for transparency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parents[1]


def _fold_run_dir(exp_id: str, fold: int) -> Path:
    if fold == 0:
        return _ROOT / "results" / "runs" / exp_id
    return _ROOT / "results" / "runs" / f"{exp_id}_fold_{fold}"


def load_pooled(
    exp_id: str, folds: list[int], pred_file: str = "predictions.npz"
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loaded: list[int] = []
    for fold in folds:
        npz_path = _fold_run_dir(exp_id, fold) / pred_file
        if not npz_path.exists():
            print(f"[WARN] Fold {fold}: not found at {npz_path} - skipping.")
            continue
        data = np.load(npz_path)
        all_preds.append(data["preds"])
        all_labels.append(data["labels"])
        loaded.append(fold)
    if not loaded:
        print(f"[ERROR] No predictions found for '{exp_id}'.")
        sys.exit(1)
    return np.concatenate(all_preds), np.concatenate(all_labels), loaded


def compute_extra(preds: np.ndarray, labels: np.ndarray) -> dict:
    accuracy = float(accuracy_score(labels, preds))
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy,
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--predictions",
        default="predictions.npz",
        help="Per-fold predictions filename to pool (e.g. predictions_tta.npz)",
    )
    args = parser.parse_args()

    preds, labels, loaded = load_pooled(args.experiment, args.folds, args.predictions)
    metrics = compute_extra(preds, labels)
    metrics["experiment"] = args.experiment
    metrics["folds_used"] = loaded
    metrics["n_samples"] = int(len(preds))

    print(f"\n{args.experiment}  (n={len(preds)}, folds={loaded})")
    print(f"  accuracy / micro-F1 : {metrics['accuracy']:.4f}")
    print(f"  macro    P/R/F1     : {metrics['macro_precision']:.4f} / "
          f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"  weighted P/R/F1     : {metrics['weighted_precision']:.4f} / "
          f"{metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}")
    print(f"  MCC                 : {metrics['mcc']:.4f}")

    metrics["predictions_file"] = args.predictions
    pred_tag = Path(args.predictions).stem.replace("predictions", "").strip("_")

    out_dir = _ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"extra_metrics_{args.experiment}.json"
        if not pred_tag
        else f"extra_metrics_{args.experiment}_{pred_tag}.json"
    )
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
