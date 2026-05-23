"""Generate report-ready result tables."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report tables.")
    parser.add_argument("--results-dir", default="results")
    parser.parse_args()
    raise SystemExit("generate_report_tables.py is a scaffold placeholder.")


if __name__ == "__main__":
    main()
