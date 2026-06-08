from pathlib import Path

import pytest
import yaml

from scripts.check_provenance import validate_provenance
from scripts.collect_outputs import RUN_FILES, collect_outputs
from scripts.write_resolved_config import build_resolved_config


def _make_dataset(root: Path) -> None:
    for index in range(23):
        class_dir = root / f"class_{index:02d}"
        class_dir.mkdir(parents=True)
        (class_dir / "image.jpg").write_bytes(f"image-{index}".encode())


def _make_manifest(path: Path) -> None:
    rows = ["path,class_name"]
    rows.extend(
        f"data/raw/hyperkvasir/labeled-images/class_{index:02d}/image.jpg,class_{index:02d}"
        for index in range(23)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_provenance_gate_rejects_dataset_mismatch(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    _make_dataset(source)
    _make_dataset(staged)
    (staged / "class_00" / "image.jpg").write_bytes(b"changed")
    manifest = tmp_path / "fold_0.csv"
    _make_manifest(manifest)
    monkeypatch.setattr("scripts.check_provenance.git_sha", lambda: "abc123")

    with pytest.raises(ValueError, match="Dataset copy mismatch"):
        validate_provenance(source, staged, manifest, "abc123")


def test_provenance_gate_accepts_matching_inputs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    _make_dataset(source)
    _make_dataset(staged)
    manifest = tmp_path / "fold_0.csv"
    _make_manifest(manifest)
    monkeypatch.setattr("scripts.check_provenance.git_sha", lambda: "abc123")

    result = validate_provenance(source, staged, manifest, "abc123")

    assert result["git_commit_sha"] == "abc123"
    assert result["dataset_file_count"] == 23
    assert len(result["per_class_image_counts"]) == 23


def test_resolved_config_uses_override_without_mutating_base(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.yaml"
    method = tmp_path / "method.yaml"
    training = tmp_path / "training.yaml"
    matrix = tmp_path / "matrix.yaml"
    dataset.write_text("num_classes: 23\n", encoding="utf-8")
    method.write_text("fusion_type: weighted\n", encoding="utf-8")
    training.write_text("batch_size: 128\n", encoding="utf-8")
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "id": "exp",
                        "dataset": str(dataset),
                        "method": str(method),
                        "training": "base.yaml",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    original = training.read_bytes()

    resolved = build_resolved_config(matrix, "exp", training, "run", "cuda", [0, 1])

    assert resolved["training"]["batch_size"] == 128
    assert resolved["runtime_overrides"] == {"device": "cuda"}
    assert training.read_bytes() == original


def test_collect_outputs_refuses_overwrite(tmp_path: Path) -> None:
    results = tmp_path / "results"
    control = results / "run"
    control.mkdir(parents=True)
    for name in ("resolved_config.yaml", "runtime.json", "run_provenance.json"):
        (control / name).write_text("x", encoding="utf-8")
    for fold in (0, 1):
        run_dir = results / ("exp" if fold == 0 else "exp_fold_1")
        run_dir.mkdir()
        for name in RUN_FILES:
            (run_dir / name).write_bytes(b"x")

    destination_root = tmp_path / "returned"
    destination = collect_outputs("run", "exp", [0, 1], results, destination_root)
    assert (destination / "fold_1" / "metrics.json").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_outputs("run", "exp", [0, 1], results, destination_root)
