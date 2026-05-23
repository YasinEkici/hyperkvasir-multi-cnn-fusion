import json
import torch
import pytest
from pathlib import Path
from PIL import Image

from src.data.feature_cache import cache_frozen_features


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> Path:
    """Create a dummy dataset with 2 images and a manifest."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    
    # Create dummy images
    img1_path = data_dir / "img1.jpg"
    img2_path = data_dir / "img2.jpg"
    Image.new("RGB", (64, 64), color="red").save(img1_path)
    Image.new("RGB", (64, 64), color="blue").save(img2_path)
    
    # Create manifest
    manifest_path = tmp_path / "test_manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("path,label,original_index\n")
        f.write(f"{img1_path},0,10\n")
        f.write(f"{img2_path},1,20\n")
        
    return manifest_path


def test_cache_frozen_features_e2e(tmp_path: Path, dummy_dataset: Path) -> None:
    # We use a fast backbone like mobilenetv2
    out_dir = tmp_path / "cache"
    
    config = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "image_size": 224,
    }
    
    saved_paths = cache_frozen_features(
        backbones=["mobilenetv2"],
        dataset_config=config,
        split_manifest=dummy_dataset,
        output_dir=out_dir,
        batch_size=2,
        device="cpu", # Use CPU for tests
    )
    
    assert "mobilenetv2" in saved_paths
    cache_file = saved_paths["mobilenetv2"]
    assert cache_file.exists()
    
    # Verify cached data
    data = torch.load(cache_file)
    
    assert "features" in data
    assert data["features"].shape == (2, 1280)
    
    assert "labels" in data
    assert torch.equal(data["labels"], torch.tensor([0, 1]))
    
    assert "indices" in data
    assert data["indices"] == [10, 20]
    
    assert "config_hash" in data
    assert isinstance(data["config_hash"], str)
