"""Compute bootstrap 95% CI for macro F1 from concatenated fold predictions.

Concatenates predictions.npz across all specified folds before bootstrapping so
that the CI is estimated on the full ~N*2122 sample pool (D-05 decision: larger N
gives a narrower, more reliable interval than per-fold bootstrap).

Run-dir convention (must match train.py --fold logic):
  fold 0 : results/runs/{id}/               (canonical, backward compatible)
  fold k≥1: results/runs/{id}_fold_{k}/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical import bootstrap_ci


def _fold_run_dir(exp_id: str, fold: int) -> Path:
    if fold == 0:
        return _ROOT / "results" / "runs" / exp_id
    return _ROOT / "results" / "runs" / f"{exp_id}_fold_{fold}"


def _macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return compute_metrics(preds, labels)["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI for macro F1 from concatenated fold predictions."
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices to include"
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--predictions",
        default="predictions.npz",
        help="Per-fold predictions filename to pool (e.g. predictions_tta.npz)",
    )
    args = parser.parse_args()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loaded_folds: list[int] = []

    for fold in args.folds:
        run_dir = _fold_run_dir(args.experiment, fold)
        npz_path = run_dir / args.predictions
        if not npz_path.exists():
            print(f"[WARN] Fold {fold}: predictions.npz not found at {npz_path} — skipping.")
            continue
        data = np.load(npz_path)
        all_preds.append(data["preds"])
        all_labels.append(data["labels"])
        loaded_folds.append(fold)

    if not loaded_folds:
        print(f"[ERROR] No predictions found for experiment '{args.experiment}'.")
        sys.exit(1)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    print(f"Concatenated {len(preds)} samples from folds {loaded_folds}")

    point, ci_low, ci_high = bootstrap_ci(
        _macro_f1, preds, labels,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=42,
    )

    ci_pct = int(args.ci * 100)
    print(
        f"Macro F1 = {point:.4f}  "
        f"[{ci_low:.4f}, {ci_high:.4f}]  "
        f"({ci_pct}% CI, n_bootstrap={args.n_bootstrap})"
    )

    result = {
        "experiment": args.experiment,
        "folds_used": loaded_folds,
        "metric": "macro_f1",
        "point_estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": args.n_bootstrap,
        "ci": args.ci,
        "n_samples": int(len(preds)),
    }

    pred_tag = Path(args.predictions).stem.replace("predictions", "").strip("_")
    result["predictions_file"] = args.predictions

    out_dir = _ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"ci_{args.experiment}.json" if not pred_tag else f"ci_{args.experiment}_{pred_tag}.json"
    out_path = out_dir / name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
