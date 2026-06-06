"""Run one experiment across multiple folds via subprocess.

Each fold is launched as a separate train.py subprocess so that GPU memory is
fully released between folds and a fold crash does not abort remaining folds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment across multiple folds."
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID from the matrix")
    parser.add_argument(
        "--folds", nargs="+", type=int, required=True, help="Fold indices to run (e.g. 1 2 3 4)"
    )
    parser.add_argument(
        "--config", default="configs/experiment_matrix.yaml", help="Path to experiment_matrix.yaml"
    )
    parser.add_argument(
        "--training",
        default=None,
        help="Override the experiment's training config without mutating the matrix",
    )
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    train_script = str(_ROOT / "scripts" / "train.py")
    failed: list[int] = []

    for fold in args.folds:
        cmd = [
            sys.executable, train_script,
            "--config", args.config,
            "--experiment", args.experiment,
            "--fold", str(fold),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.training:
            cmd += ["--training", args.training]

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Fold {fold} | experiment: {args.experiment}")
        print(sep)

        result = subprocess.run(cmd, cwd=str(_ROOT))
        if result.returncode != 0:
            print(f"[FAIL] Fold {fold} returned exit code {result.returncode} — continuing.")
            failed.append(fold)
        else:
            print(f"[OK]   Fold {fold} completed.")

    print(f"\n{'=' * 60}")
    print(f"CV finished | experiment: {args.experiment}")
    if failed:
        print(f"FAILED folds: {failed}")
        sys.exit(1)
    else:
        print(f"All folds succeeded: {args.folds}")


if __name__ == "__main__":
    main()
