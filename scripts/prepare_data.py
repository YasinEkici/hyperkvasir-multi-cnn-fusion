"""Prepare dataset files for experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifests import (
    count_by_class,
    create_hyperkvasir_manifest,
    load_dataset_config,
    resolve_config_path,
    write_manifest_csv,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Prepare datasets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config",
        default="configs/dataset/hyperkvasir_23class_official.yaml",
        help="Dataset config path.",
    )
    args = parser.parse_args()

    if args.dataset != "hyperkvasir":
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    config = load_dataset_config(args.config)
    root = resolve_config_path(config["root"])
    samples = create_hyperkvasir_manifest(root, num_classes=int(config["num_classes"]))
    output_path = Path(config["split_manifest_dir"]) / "hyperkvasir_manifest.csv"
    write_manifest_csv(samples, output_path)

    logger.info("manifest_path=%s", output_path)
    logger.info("sample_count=%s", len(samples))
    for class_name, count in sorted(count_by_class(samples).items()):
        logger.info("class_count %s=%s", class_name, count)


if __name__ == "__main__":
    main()
