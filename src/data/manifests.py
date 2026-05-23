"""Dataset manifest creation and validation utilities."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML dataset config."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must be a mapping: {config_path}")
    return config


def resolve_config_path(path_value: str | Path, *, data_root: str | Path = "data/raw") -> Path:
    """Resolve config paths, including the project's `${DATA_ROOT}` placeholder."""
    path_text = str(path_value)
    data_root_text = os.environ.get("DATA_ROOT", str(data_root))
    path_text = path_text.replace("${DATA_ROOT}", data_root_text)
    return Path(path_text).expanduser()


def create_hyperkvasir_manifest(
    root: str | Path,
    *,
    num_classes: int = 23,
    project_root: str | Path | None = None,
) -> list[dict[str, str | int]]:
    """Create a deterministic manifest for the HyperKvasir labeled image subset."""
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"HyperKvasir root does not exist: {dataset_root}")

    project_root_path = Path(project_root) if project_root is not None else Path.cwd()
    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No JPEG images found under: {dataset_root}")

    class_names = sorted({path.parent.name for path in image_paths})
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes, found {len(class_names)}: {class_names}"
        )
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}

    rows: list[dict[str, str | int]] = []
    seen_file_names: set[str] = set()
    duplicate_file_names: set[str] = set()
    for original_index, image_path in enumerate(image_paths):
        file_name = image_path.name
        if file_name in seen_file_names:
            duplicate_file_names.add(file_name)
        seen_file_names.add(file_name)

        class_name = image_path.parent.name
        path_value = _to_project_relative_posix(image_path, project_root_path)
        category_path = image_path.parent.parent.relative_to(dataset_root).as_posix()
        rows.append(
            {
                "original_index": original_index,
                "path": path_value,
                "file_name": file_name,
                "class_name": class_name,
                "label": class_to_label[class_name],
                "category_path": category_path,
            }
        )

    if duplicate_file_names:
        duplicates = ", ".join(sorted(duplicate_file_names)[:10])
        raise ValueError(f"Duplicate image file names found: {duplicates}")

    return rows


def count_by_class(samples: list[dict[str, str | int]]) -> Counter[str]:
    """Return per-class sample counts."""
    return Counter(str(sample["class_name"]) for sample in samples)


def write_manifest_csv(samples: list[dict[str, str | int]], output_path: str | Path) -> Path:
    """Write a manifest CSV with stable field ordering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["original_index", "path", "file_name", "class_name", "label", "category_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    return path


def read_manifest_csv(path: str | Path) -> list[dict[str, str | int]]:
    """Read a manifest CSV written by `write_manifest_csv` or split generation."""
    rows: list[dict[str, str | int]] = []
    int_fields = {"original_index", "label", "fold", "official_fold"}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, str | int] = {}
            for key, value in row.items():
                parsed[key] = int(value) if key in int_fields and value != "" else value
            rows.append(parsed)
    return rows


def _to_project_relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
