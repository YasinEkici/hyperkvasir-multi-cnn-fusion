"""Leakage-free seed ensemble: average per-fold softmax across seeds, then pool.

Per-fold held-out only (decision D-07): seeds are averaged within each fold's own
test set; the 5 folds are then pooled so each sample appears exactly once. NO
cross-fold model averaging (that would leak — every sample was in 4 folds' train
sets).

Reads the per-seed, per-fold softmax written by evaluate.py:
  predictions_base.npz  (default)   or   predictions_tta.npz  (--tta)

Seed -> run-dir convention (matches train.py):
  seed 42  : canonical {exp}/ , {exp}_fold_{k}/
  seed !=42: {exp}_seed{S}/ , {exp}_seed{S}_fold_{k}/

Writes the ensemble predictions into the CANONICAL fold dirs as
predictions_ensemble.npz (or predictions_ensemble_tta.npz) so compute_ci.py and
compute_extra_metrics.py can pool them via --predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import compute_metrics

CANONICAL_SEED = 42


def seed_fold_dir(exp: str, seed: int, fold: int) -> Path:
    name = exp if seed == CANONICAL_SEED else f"{exp}_seed{seed}"
    if fold != 0:
        name = f"{name}_fold_{fold}"
    return _ROOT / "results" / "runs" / name


def canonical_fold_dir(exp: str, fold: int) -> Path:
    name = exp if fold == 0 else f"{exp}_fold_{fold}"
    return _ROOT / "results" / "runs" / name


def average_softmax(prob_list: list[np.ndarray]) -> np.ndarray:
    """Average a list of (N, C) softmax arrays across seeds.

    Single seed returns it unchanged; the mean of rows that each sum to 1 also
    sums to 1.
    """
    if not prob_list:
        raise ValueError("average_softmax requires at least one seed")
    return np.stack(prob_list, axis=0).mean(axis=0)


def ensemble_fold(
    exp: str, seeds: list[int], fold: int, src_file: str, out_file: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    probs_list: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for s in seeds:
        path = seed_fold_dir(exp, s, fold) / src_file
        if not path.exists():
            raise FileNotFoundError(
                f"seed {s} fold {fold}: {path} missing — run evaluate.py for that seed first."
            )
        data = np.load(path)
        probs_list.append(data["probs"])
        if labels is None:
            labels = data["labels"]
        elif not np.array_equal(labels, data["labels"]):
            raise RuntimeError(f"Label order differs across seeds at fold {fold} — cannot ensemble.")

    mean_probs = average_softmax(probs_list)
    preds = mean_probs.argmax(axis=1).astype(np.int64)
    assert labels is not None
    out_dir = canonical_fold_dir(exp, fold)
    np.savez(
        out_dir / out_file,
        preds=preds,
        labels=labels.astype(np.int64),
        probs=mean_probs.astype(np.float32),
    )
    return preds, labels, compute_metrics(preds, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tta", action="store_true", help="Ensemble the TTA softmax instead of base")
    args = parser.parse_args()

    src_file = "predictions_tta.npz" if args.tta else "predictions_base.npz"
    out_file = "predictions_ensemble_tta.npz" if args.tta else "predictions_ensemble.npz"
    print(f"Seed ensemble {args.seeds} | source={src_file} | out={out_file}")

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for fold in args.folds:
        preds, labels, m = ensemble_fold(args.experiment, args.seeds, fold, src_file, out_file)
        print(f"  fold {fold}: n={len(labels)}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
        all_preds.append(preds)
        all_labels.append(labels)

    pooled_preds = np.concatenate(all_preds)
    pooled_labels = np.concatenate(all_labels)
    pooled = compute_metrics(pooled_preds, pooled_labels)
    print(
        f"POOLED n={len(pooled_labels)}  acc={pooled['accuracy']:.4f}  "
        f"macro_f1={pooled['macro_f1']:.4f}  (run compute_ci.py --predictions {out_file})"
    )


if __name__ == "__main__":
    main()
