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
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[1]


def _fold_run_dir(exp_id: str, fold: int) -> Path:
    if fold == 0:
        return _ROOT / "results" / "runs" / exp_id
    return _ROOT / "results" / "runs" / f"{exp_id}_fold_{fold}"


def load_pooled(
    exp_id: str, folds: list[int], pred_file: str = "predictions.npz"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int]]:
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    loaded: list[int] = []
    have_probs = True
    for fold in folds:
        npz_path = _fold_run_dir(exp_id, fold) / pred_file
        if not npz_path.exists():
            print(f"[WARN] Fold {fold}: not found at {npz_path} - skipping.")
            continue
        data = np.load(npz_path)
        all_preds.append(data["preds"])
        all_labels.append(data["labels"])
        loaded.append(fold)
        if "probs" in data.files:
            all_probs.append(data["probs"])
        else:
            have_probs = False
    if not loaded:
        print(f"[ERROR] No predictions found for '{exp_id}'.")
        sys.exit(1)
    probs = np.concatenate(all_probs) if (have_probs and all_probs) else None
    return np.concatenate(all_preds), np.concatenate(all_labels), probs, loaded


def compute_extra(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None) -> dict:
    accuracy = float(accuracy_score(labels, preds))
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    out = {
        "accuracy": accuracy,
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
    }
    # macro one-vs-rest ROC-AUC needs class probabilities; skip gracefully if absent.
    if probs is not None:
        try:
            out["auc_macro_ovr"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] AUC skipped: {exc}")
            out["auc_macro_ovr"] = None
    else:
        out["auc_macro_ovr"] = None
    return out


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

    preds, labels, probs, loaded = load_pooled(args.experiment, args.folds, args.predictions)
    metrics = compute_extra(preds, labels, probs)
    metrics["experiment"] = args.experiment
    metrics["folds_used"] = loaded
    metrics["n_samples"] = int(len(preds))

    auc = metrics.get("auc_macro_ovr")
    print(f"\n{args.experiment}  (n={len(preds)}, folds={loaded})")
    print(f"  accuracy / micro-F1 : {metrics['accuracy']:.4f}")
    print(f"  macro    P/R/F1     : {metrics['macro_precision']:.4f} / "
          f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"  weighted P/R/F1     : {metrics['weighted_precision']:.4f} / "
          f"{metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}")
    print(f"  MCC                 : {metrics['mcc']:.4f}")
    print(f"  Cohen Kappa         : {metrics['cohen_kappa']:.4f}")
    print(f"  macro ROC-AUC (OVR) : {auc:.4f}" if auc is not None else "  macro ROC-AUC (OVR) : n/a (no probs)")

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
