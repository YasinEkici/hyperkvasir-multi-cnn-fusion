"""Write an immutable, run-specific configuration for Colab execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build_resolved_config(
    matrix_path: Path,
    experiment_id: str,
    training_path: Path,
    run_id: str,
    device: str,
    folds: list[int],
) -> dict:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    experiments = {item["id"]: item for item in matrix["experiments"]}
    if experiment_id not in experiments:
        raise ValueError(f"Experiment '{experiment_id}' not found in {matrix_path}")

    experiment = dict(experiments[experiment_id])
    dataset_path = Path(experiment["dataset"])
    method_path = Path(experiment["method"])

    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "folds": folds,
        "runtime_overrides": {"device": device},
        "source_paths": {
            "experiment_matrix": matrix_path.as_posix(),
            "dataset": dataset_path.as_posix(),
            "method": method_path.as_posix(),
            "training": training_path.as_posix(),
        },
        "experiment": experiment,
        "dataset": yaml.safe_load(dataset_path.read_text(encoding="utf-8")),
        "method": yaml.safe_load(method_path.read_text(encoding="utf-8")),
        "training": yaml.safe_load(training_path.read_text(encoding="utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", default="results/runs")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "resolved_config.yaml"
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing resolved config: {output_path}")

    resolved = build_resolved_config(
        Path(args.matrix),
        args.experiment,
        Path(args.training),
        args.run_id,
        args.device,
        args.folds,
    )
    output_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
