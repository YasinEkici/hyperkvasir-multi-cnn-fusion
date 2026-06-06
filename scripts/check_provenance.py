"""Validate staged Colab inputs and write the run provenance records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import torch
import torchvision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> tuple[str, int]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Dataset directory contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(files)


def dataset_class_counts(root: Path) -> dict[str, int]:
    extensions = {".jpg", ".jpeg", ".png"}
    counts = Counter(
        path.parent.name
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    return dict(sorted(counts.items()))


def manifest_class_counts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_path: dict[str, str] = {}
    for row in rows:
        image_path = row["path"]
        class_name = row["class_name"]
        if image_path in by_path and by_path[image_path] != class_name:
            raise ValueError(f"Manifest assigns two classes to {image_path}")
        by_path[image_path] = class_name
    return dict(sorted(Counter(by_path.values()).items()))


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def validate_provenance(
    source_root: Path,
    staged_root: Path,
    manifest_path: Path,
    expected_git_sha: str,
) -> dict:
    actual_git_sha = git_sha()
    if actual_git_sha != expected_git_sha:
        raise ValueError(
            f"Git SHA mismatch: expected {expected_git_sha}, found {actual_git_sha}"
        )

    source_hash, source_files = sha256_tree(source_root)
    staged_hash, staged_files = sha256_tree(staged_root)
    if (source_hash, source_files) != (staged_hash, staged_files):
        raise ValueError("Dataset copy mismatch between approved source and staged input")

    source_counts = dataset_class_counts(source_root)
    staged_counts = dataset_class_counts(staged_root)
    expected_counts = manifest_class_counts(manifest_path)
    if source_counts != staged_counts:
        raise ValueError("Per-class image counts differ between source and staged dataset")
    if staged_counts != expected_counts:
        raise ValueError("Staged dataset class counts do not match the split manifest")
    if len(staged_counts) != 23:
        raise ValueError(f"Expected 23 classes, found {len(staged_counts)}")

    return {
        "git_commit_sha": actual_git_sha,
        "dataset_sha256": staged_hash,
        "dataset_file_count": staged_files,
        "per_class_image_counts": staged_counts,
        "split_manifest": manifest_path.as_posix(),
        "split_manifest_sha256": sha256_file(manifest_path),
        "label_scheme": "hyperkvasir_23class",
    }


def runtime_record(device: str) -> dict:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "requested_device": device,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-dataset-root", required=True)
    parser.add_argument("--staged-dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--approved-source", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="results/runs")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / f"{args.run_id}_provenance.json"
    runtime_path = output_dir / "runtime.json"
    if provenance_path.exists() or runtime_path.exists():
        raise SystemExit(f"Refusing to overwrite provenance records in {output_dir}")

    try:
        provenance = validate_provenance(
            Path(args.source_dataset_root),
            Path(args.staged_dataset_root),
            Path(args.manifest),
            args.expected_git_sha,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"PROVENANCE GATE FAILED: {exc}") from exc

    runtime = runtime_record(args.device)
    provenance["approved_source"] = args.approved_source
    provenance["runtime"] = runtime
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(f"PROVENANCE GATE PASSED: {provenance_path}")


if __name__ == "__main__":
    main()
