"""Create split manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifests import (
    create_hyperkvasir_manifest,
    load_dataset_config,
    resolve_config_path,
)
from src.data.splits import create_official_fold_manifests


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Create dataset split manifests.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    if config.get("split_protocol") != "hyperkvasir_official_5fold":
        raise SystemExit(f"Unsupported split_protocol: {config.get('split_protocol')}")

    root = resolve_config_path(config["root"])
    official_split_file = Path(config["official_split_file"])
    samples = create_hyperkvasir_manifest(root, num_classes=int(config["num_classes"]))
    output_paths = create_official_fold_manifests(
        samples,
        official_split_file,
        Path(config["split_manifest_dir"]),
    )

    for output_path in output_paths:
        logger.info("split_manifest=%s", output_path)


if __name__ == "__main__":
    main()
