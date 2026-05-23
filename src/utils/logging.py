"""Structured logging utilities."""

import json
import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_epoch(run_dir: Path, epoch_data: dict) -> None:
    """Append one epoch record to epoch_log.jsonl."""
    log_file = run_dir / "epoch_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(epoch_data) + "\n")


def save_metrics(run_dir: Path, metrics: dict) -> None:
    """Write final metrics.json to the run directory."""
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
