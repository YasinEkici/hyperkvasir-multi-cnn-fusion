from pathlib import Path

from PIL import Image

from src.data.manifests import create_hyperkvasir_manifest
from src.data.splits import create_official_fold_manifests, make_stratified_folds
from src.data.transforms import build_image_transform


def test_make_stratified_folds_imports() -> None:
    assert callable(make_stratified_folds)


def test_hyperkvasir_manifest_discovers_leaf_classes(tmp_path: Path) -> None:
    root = tmp_path / "labeled-images"
    for class_name in ("class-a", "class-b"):
        class_dir = root / "group" / "subgroup" / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(class_dir / f"{class_name}.jpg")

    samples = create_hyperkvasir_manifest(root, num_classes=2, project_root=tmp_path)

    assert len(samples) == 2
    assert {sample["class_name"] for sample in samples} == {"class-a", "class-b"}
    assert {sample["label"] for sample in samples} == {0, 1}
    assert all(str(sample["path"]).startswith("labeled-images/") for sample in samples)


def test_official_fold_manifests_have_no_leakage(tmp_path: Path) -> None:
    samples = []
    split_file = tmp_path / "5_fold_split.csv"
    with split_file.open("w", encoding="utf-8", newline="") as handle:
        handle.write("file-name;class-name;split-index\n")
        original_index = 0
        for fold in range(5):
            for class_name, label in (("class-a", 0), ("class-b", 1)):
                file_name = f"{class_name}-{fold}.jpg"
                handle.write(f"{file_name};{class_name};{fold}\n")
                samples.append(
                    {
                        "original_index": original_index,
                        "path": f"data/{file_name}",
                        "file_name": file_name,
                        "class_name": class_name,
                        "label": label,
                        "category_path": "group",
                    }
                )
                original_index += 1

    output_paths = create_official_fold_manifests(samples, split_file, tmp_path / "splits")

    assert len(output_paths) == 5
    first_manifest = output_paths[0].read_text(encoding="utf-8")
    assert "train" in first_manifest
    assert "val" in first_manifest
    assert "test" in first_manifest


def test_build_image_transform_outputs_224_tensor() -> None:
    transform = build_image_transform("val")
    image = Image.new("RGB", (320, 240))

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 224, 224)
