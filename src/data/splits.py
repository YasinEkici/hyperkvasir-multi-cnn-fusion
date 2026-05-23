"""Split manifest creation utilities."""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


def make_stratified_folds(
    samples: list[dict],
    n_splits: int,
    seed: int,
    output_dir: Path,
) -> list[Path]:
    """Create fallback stratified fold manifests without train/test leakage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [sample["label"] for sample in samples]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output_paths: list[Path] = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(samples, labels)):
        fold_path = output_dir / f"stratified_fold_{fold}.csv"
        rows = []
        for split_name, indices in (("train", train_indices), ("test", test_indices)):
            for idx in indices:
                row = dict(samples[int(idx)])
                row["fold"] = fold
                row["split"] = split_name
                rows.append(row)
        write_split_manifest(rows, fold_path)
        validate_no_leakage(rows)
        output_paths.append(fold_path)

    return output_paths


def create_official_fold_manifests(
    samples: list[dict],
    official_split_file: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Materialize train/val/test manifests from HyperKvasir official 5-fold CSV."""
    official_rows = read_official_split_csv(official_split_file)
    official_by_file = {row["file_name"]: row for row in official_rows}
    if len(official_by_file) != len(official_rows):
        raise ValueError("Official split CSV contains duplicate file names.")

    missing = sorted(str(sample["file_name"]) for sample in samples if sample["file_name"] not in official_by_file)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} manifest files are missing from official split: {preview}")

    extra = sorted(set(official_by_file) - {str(sample["file_name"]) for sample in samples})
    if extra:
        preview = ", ".join(extra[:10])
        raise ValueError(f"{len(extra)} official split files are missing locally: {preview}")

    for sample in samples:
        official = official_by_file[str(sample["file_name"])]
        if official["class_name"] != sample["class_name"]:
            raise ValueError(
                "Official split class name does not match local folder name for "
                f"{sample['file_name']}: {official['class_name']} != {sample['class_name']}"
            )

    official_folds = sorted({int(row["official_fold"]) for row in official_rows})
    if official_folds != list(range(5)):
        raise ValueError(f"Expected official folds [0, 1, 2, 3, 4], found {official_folds}")
    expected_num_classes = len({str(sample["class_name"]) for sample in samples})

    output_root = Path(output_dir) / "hyperkvasir_official_5fold"
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for fold in official_folds:
        val_fold = (fold + 1) % len(official_folds)
        rows = []
        for sample in samples:
            official_fold = int(official_by_file[str(sample["file_name"])]["official_fold"])
            split_name = "test" if official_fold == fold else "val" if official_fold == val_fold else "train"
            row = dict(sample)
            row["fold"] = fold
            row["official_fold"] = official_fold
            row["split"] = split_name
            rows.append(row)

        validate_no_leakage(rows)
        validate_split_class_coverage(rows, expected_num_classes=expected_num_classes)
        fold_path = output_root / f"fold_{fold}.csv"
        write_split_manifest(rows, fold_path)
        output_paths.append(fold_path)

    return output_paths


def read_official_split_csv(path: str | Path) -> list[dict[str, str | int]]:
    """Read the semicolon-delimited official HyperKvasir split CSV."""
    rows: list[dict[str, str | int]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        expected = {"file-name", "class-name", "split-index"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected official split header: {reader.fieldnames}")
        for row in reader:
            rows.append(
                {
                    "file_name": row["file-name"],
                    "class_name": row["class-name"],
                    "official_fold": int(row["split-index"]),
                }
            )
    return rows


def write_split_manifest(rows: list[dict], output_path: str | Path) -> Path:
    """Write split rows with stable field ordering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "original_index",
        "path",
        "file_name",
        "class_name",
        "label",
        "category_path",
        "fold",
        "official_fold",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def validate_no_leakage(rows: list[dict]) -> None:
    """Ensure each fold manifest assigns each image to exactly one split."""
    by_path: dict[str, set[str]] = {}
    by_index: dict[int, set[str]] = {}
    for row in rows:
        by_path.setdefault(str(row["path"]), set()).add(str(row["split"]))
        by_index.setdefault(int(row["original_index"]), set()).add(str(row["split"]))

    leaked_paths = [path for path, splits in by_path.items() if len(splits) > 1]
    leaked_indices = [idx for idx, splits in by_index.items() if len(splits) > 1]
    if leaked_paths or leaked_indices:
        raise ValueError(
            f"Split leakage detected: {len(leaked_paths)} paths, {len(leaked_indices)} indices"
        )


def validate_split_class_coverage(rows: list[dict], *, expected_num_classes: int) -> None:
    """Ensure train, validation, and test partitions contain every class."""
    for split_name in ("train", "val", "test"):
        classes = {str(row["class_name"]) for row in rows if row["split"] == split_name}
        if len(classes) != expected_num_classes:
            raise ValueError(
                f"Split '{split_name}' has {len(classes)} classes; expected {expected_num_classes}."
            )
