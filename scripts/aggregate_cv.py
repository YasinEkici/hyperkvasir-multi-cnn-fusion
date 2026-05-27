"""Aggregate per-fold metrics.json files into a mean ± std CV summary.

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


def _fold_run_dir(exp_id: str, fold: int) -> Path:
    if fold == 0:
        return _ROOT / "results" / "runs" / exp_id
    return _ROOT / "results" / "runs" / f"{exp_id}_fold_{fold}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CV fold metrics into mean ± std.")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Fold indices to aggregate"
    )
    args = parser.parse_args()

    metric_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    per_metric: dict[str, list[float]] = {k: [] for k in metric_keys}
    loaded_folds: list[int] = []
    missing_folds: list[int] = []

    for fold in args.folds:
        run_dir = _fold_run_dir(args.experiment, fold)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] Fold {fold}: metrics.json not found at {metrics_path} — skipping.")
            missing_folds.append(fold)
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        test = metrics.get("test", {})
        missing_keys = [k for k in metric_keys if test.get(k) is None]
        if missing_keys:
            print(f"[WARN] Fold {fold}: missing keys {missing_keys} in test metrics — skipping fold.")
            missing_folds.append(fold)
            continue

        for k in metric_keys:
            per_metric[k].append(float(test[k]))
        loaded_folds.append(fold)

    if not loaded_folds:
        print(f"[ERROR] No valid fold results found for experiment '{args.experiment}'.")
        sys.exit(1)

    if missing_folds:
        print(f"[WARN] Missing folds: {missing_folds}. Aggregating over folds: {loaded_folds}.")

    summary: dict = {
        "experiment": args.experiment,
        "folds_used": loaded_folds,
        "folds_missing": missing_folds,
        "n_folds": len(loaded_folds),
    }

    print(f"\nCV summary — {args.experiment}  (n={len(loaded_folds)} folds: {loaded_folds})")
    for k in metric_keys:
        vals = np.array(per_metric[k])
        mean, std = float(vals.mean()), float(vals.std())
        summary[k] = {"mean": mean, "std": std, "per_fold": [round(v, 6) for v in per_metric[k]]}
        print(f"  {k:20s}: {mean:.4f} ± {std:.4f}   per-fold: {[round(v, 4) for v in per_metric[k]]}")

    out_dir = _ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cv_{args.experiment}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
