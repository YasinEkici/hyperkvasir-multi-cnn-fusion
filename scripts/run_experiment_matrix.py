"""Run experiments from the experiment matrix."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix.")
    parser.add_argument("--config", default="configs/experiment_matrix.yaml")
    parser.parse_args()
    raise SystemExit("run_experiment_matrix.py is a scaffold placeholder.")


if __name__ == "__main__":
    main()
