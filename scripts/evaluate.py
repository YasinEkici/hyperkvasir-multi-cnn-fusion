"""Evaluate a trained experiment."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run.")
    parser.add_argument("--run-dir", required=True)
    parser.parse_args()
    raise SystemExit("evaluate.py is a scaffold placeholder.")


if __name__ == "__main__":
    main()
