"""Collect the fixed Colab return-artifact set without overwriting prior runs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CONTROL_FILES = ("resolved_config.yaml", "runtime.json")
RUN_FILES = ("config.yaml", "best.pt", "epoch_log.jsonl", "metrics.json", "predictions.npz")


def fold_run_dir(results_root: Path, experiment: str, fold: int) -> Path:
    return results_root / (experiment if fold == 0 else f"{experiment}_fold_{fold}")


def collect_outputs(
    run_id: str,
    experiment: str,
    folds: list[int],
    results_root: Path,
    destination_root: Path,
) -> Path:
    destination = destination_root / run_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing returned run: {destination}")

    control_dir = results_root / run_id
    provenance_name = f"{run_id}_provenance.json"
    required_control = [*CONTROL_FILES, provenance_name]
    required: list[Path] = [control_dir / name for name in required_control]
    for fold in folds:
        required.extend(fold_run_dir(results_root, experiment, fold) / name for name in RUN_FILES)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + ", ".join(map(str, missing)))

    destination.mkdir(parents=True)
    for name in required_control:
        shutil.copy2(control_dir / name, destination / name)
    for fold in folds:
        source = fold_run_dir(results_root, experiment, fold)
        fold_destination = destination / f"fold_{fold}"
        fold_destination.mkdir()
        for name in RUN_FILES:
            shutil.copy2(source / name, fold_destination / name)
        figures = sorted([*source.glob("*.png"), *source.glob("*.svg")])
        if figures:
            figure_dir = fold_destination / "figures"
            figure_dir.mkdir()
            for figure in figures:
                shutil.copy2(figure, figure_dir / figure.name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--destination-root", required=True)
    args = parser.parse_args()

    try:
        destination = collect_outputs(
            args.run_id,
            args.experiment,
            args.folds,
            Path(args.results_root),
            Path(args.destination_root),
        )
    except (FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        print(f"[OK] {path.relative_to(destination)}")


if __name__ == "__main__":
    main()
