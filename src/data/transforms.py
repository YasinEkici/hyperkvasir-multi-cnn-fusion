"""Image transform builders for train, validation, and test pipelines."""

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(
    split: str,
    *,
    image_size: int = 224,
    resize_size: int = 256,
    mean: list[float] | tuple[float, float, float] = IMAGENET_MEAN,
    std: list[float] | tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Build the Week 1 train/validation/test image preprocessing pipeline."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Expected train, val, or test.")

    crop = transforms.RandomCrop(image_size) if split == "train" else transforms.CenterCrop(image_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
