"""Tests for the TTA logic in scripts/evaluate.py.

Covers the deterministic view set and the softmax aggregation — the pure,
model-free parts. Model inference itself is validated at run time by comparing
the base (no-TTA) predictions against the predictions saved during training.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import aggregate_softmax, make_views

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMG_SIZE = 224
_RESIZE = 256


def _asymmetric_image() -> Image.Image:
    """RGB image whose left half is black and right half white (flip-sensitive)."""
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, 150:, :] = 255
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# make_views — view set
# ---------------------------------------------------------------------------

def test_no_tta_returns_only_base():
    views = make_views(False, _IMG_SIZE, _RESIZE, _MEAN, _STD)
    assert len(views) == 1
    assert views[0][0] == "base"


def test_tta_returns_base_first_plus_extra_views():
    views = make_views(True, _IMG_SIZE, _RESIZE, _MEAN, _STD)
    names = [n for n, _ in views]
    assert names[0] == "base"  # identity view always included and first
    assert len(views) == 4
    assert set(names) == {"base", "hflip", "scale", "scale_hflip"}


def test_base_transform_is_deterministic():
    _, base_tfm = make_views(False, _IMG_SIZE, _RESIZE, _MEAN, _STD)[0]
    img = _asymmetric_image()
    out_a = base_tfm(img)
    out_b = base_tfm(img)
    assert torch.equal(out_a, out_b)


def test_base_transform_output_shape():
    _, base_tfm = make_views(False, _IMG_SIZE, _RESIZE, _MEAN, _STD)[0]
    out = base_tfm(_asymmetric_image())
    assert out.shape == (3, _IMG_SIZE, _IMG_SIZE)


def test_hflip_view_differs_from_base():
    views = dict(make_views(True, _IMG_SIZE, _RESIZE, _MEAN, _STD))
    img = _asymmetric_image()
    base_out = views["base"](img)
    flip_out = views["hflip"](img)
    assert base_out.shape == flip_out.shape
    assert not torch.equal(base_out, flip_out)  # flip changes an asymmetric image
    # horizontal flip of base equals the hflip view (deterministic p=1.0 flip)
    assert torch.allclose(torch.flip(base_out, dims=[2]), flip_out, atol=1e-6)


# ---------------------------------------------------------------------------
# aggregate_softmax — averaging
# ---------------------------------------------------------------------------

def test_aggregate_single_view_is_identity():
    probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]], dtype=np.float32)
    out = aggregate_softmax([probs])
    assert np.array_equal(out, probs)  # identity == base


def test_aggregate_preserves_shape():
    a = np.full((5, 23), 1.0 / 23, dtype=np.float32)
    b = np.full((5, 23), 1.0 / 23, dtype=np.float32)
    out = aggregate_softmax([a, b])
    assert out.shape == (5, 23)


def test_aggregate_probs_sum_to_one():
    rng = np.random.default_rng(0)
    views = []
    for _ in range(4):
        logits = rng.normal(size=(10, 23))
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        views.append(ex / ex.sum(axis=1, keepdims=True))  # valid softmax rows
    out = aggregate_softmax(views)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_aggregate_is_elementwise_mean():
    a = np.array([[0.2, 0.8]], dtype=np.float32)
    b = np.array([[0.6, 0.4]], dtype=np.float32)
    out = aggregate_softmax([a, b])
    assert np.allclose(out, [[0.4, 0.6]])


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_softmax([])
