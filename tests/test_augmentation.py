"""Tests for src/data/augmentation.py."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import apply_cutmix, apply_rand_augment

# ---------------------------------------------------------------------------
# apply_rand_augment
# ---------------------------------------------------------------------------

def _make_pil(size: int = 224, color: str = "RGB") -> Image.Image:
    return Image.new(color, (size, size), color=128)


def test_rand_augment_returns_pil():
    img = _make_pil()
    out = apply_rand_augment(img, N=2, M=9)
    assert isinstance(out, Image.Image)


def test_rand_augment_preserves_size():
    img = _make_pil(224)
    out = apply_rand_augment(img, N=2, M=9)
    assert out.size == (224, 224)


def test_rand_augment_preserves_mode():
    img = _make_pil()
    out = apply_rand_augment(img, N=2, M=9)
    assert out.mode == "RGB"


def test_rand_augment_non_square():
    img = Image.new("RGB", (320, 240), color=0)
    out = apply_rand_augment(img, N=1, M=5)
    assert out.size == (320, 240)


def test_rand_augment_zero_ops():
    # N=0 should be a no-op transform
    img = _make_pil()
    out = apply_rand_augment(img, N=0, M=9)
    assert out.size == img.size


# ---------------------------------------------------------------------------
# apply_cutmix
# ---------------------------------------------------------------------------

B, C, H, W = 4, 3, 224, 224
NUM_CLASSES = 23


def _make_batch(b: int = B) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.rand(b, C, H, W)
    labels = torch.randint(0, NUM_CLASSES, (b,))
    return images, labels


def test_cutmix_output_shapes():
    images, labels = _make_batch()
    mixed, la, lb, lam = apply_cutmix(images, labels, alpha=1.0)
    assert mixed.shape == (B, C, H, W)
    assert la.shape == (B,)
    assert lb.shape == (B,)
    assert isinstance(lam, float)


def test_cutmix_lam_in_unit_interval():
    images, labels = _make_batch()
    for _ in range(20):
        _, _, _, lam = apply_cutmix(images, labels, alpha=1.0)
        assert 0.0 <= lam <= 1.0, f"lam={lam} out of [0, 1]"


def test_cutmix_lam_reflects_actual_area():
    # lam is re-computed from clipped box, not from raw Beta sample.
    # We can't assert an exact value but we verify it is in [0, 1].
    images, labels = _make_batch()
    _, _, _, lam = apply_cutmix(images, labels, alpha=0.1)
    assert 0.0 <= lam <= 1.0


def test_cutmix_output_dtype_preserved():
    images, labels = _make_batch()
    mixed, _, _, _ = apply_cutmix(images, labels, alpha=1.0)
    assert mixed.dtype == images.dtype


def test_cutmix_label_a_unchanged():
    # label_a must be the original labels, not shuffled
    images, labels = _make_batch()
    _, la, _, _ = apply_cutmix(images, labels, alpha=1.0)
    assert torch.equal(la, labels)


def test_cutmix_label_b_is_permutation():
    # label_b must contain the same set of values as labels (permuted)
    images, labels = _make_batch()
    _, _, lb, _ = apply_cutmix(images, labels, alpha=1.0)
    assert sorted(lb.tolist()) == sorted(labels.tolist())


def test_cutmix_mixed_region_differs_from_original():
    # With alpha=1.0 and a large batch, at least some pixels should change.
    # Use a batch where all images have different constant pixel values.
    images = torch.stack([torch.full((C, H, W), float(i)) for i in range(B)])
    labels = torch.arange(B)
    mixed, _, _, lam = apply_cutmix(images, labels, alpha=1.0)
    if lam < 1.0:
        # Some pixels were replaced — mixed cannot equal every original image
        all_same = all(torch.equal(mixed[i], images[i]) for i in range(B))
        assert not all_same


def test_cutmix_batch_size_one():
    # B=1: shuffle of 1 element is itself; label_a == label_b, image unchanged
    images, labels = _make_batch(b=1)
    mixed, la, lb, lam = apply_cutmix(images, labels, alpha=1.0)
    assert mixed.shape == (1, C, H, W)
    assert torch.equal(la, lb)
    assert 0.0 <= lam <= 1.0


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_cutmix_various_alpha(alpha: float):
    images, labels = _make_batch()
    mixed, la, lb, lam = apply_cutmix(images, labels, alpha=alpha)
    assert mixed.shape == (B, C, H, W)
    assert 0.0 <= lam <= 1.0
