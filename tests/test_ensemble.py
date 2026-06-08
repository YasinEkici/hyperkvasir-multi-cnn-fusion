"""Tests for the leakage-free seed-ensemble logic in scripts/ensemble_seeds.py.

Covers the softmax averaging and the seed -> run-dir convention (the pure,
file-free parts). The actual per-fold pooling is validated at run time against
the saved per-seed predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ensemble_seeds import average_softmax, canonical_fold_dir, seed_fold_dir

_EXP = "11_triple_weighted_finetune_wide_official"


# ---------------------------------------------------------------------------
# average_softmax
# ---------------------------------------------------------------------------

def test_single_seed_is_identity():
    probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]], dtype=np.float32)
    assert np.array_equal(average_softmax([probs]), probs)


def test_two_seed_elementwise_mean():
    a = np.array([[0.2, 0.8]], dtype=np.float32)
    b = np.array([[0.6, 0.4]], dtype=np.float32)
    assert np.allclose(average_softmax([a, b]), [[0.4, 0.6]])


def test_averaged_probs_sum_to_one():
    rng = np.random.default_rng(1)
    seeds = []
    for _ in range(3):
        logits = rng.normal(size=(8, 23))
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        seeds.append(ex / ex.sum(axis=1, keepdims=True))
    out = average_softmax(seeds)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_preserves_shape():
    a = np.full((5, 23), 1.0 / 23, dtype=np.float32)
    assert average_softmax([a, a, a]).shape == (5, 23)


def test_empty_raises():
    with pytest.raises(ValueError):
        average_softmax([])


# ---------------------------------------------------------------------------
# seed -> run-dir convention (no overwrite of canonical seed-42 runs)
# ---------------------------------------------------------------------------

def test_seed42_is_canonical():
    # seed 42 maps to the existing un-suffixed Week-3 dirs
    assert seed_fold_dir(_EXP, 42, 0).name == _EXP
    assert seed_fold_dir(_EXP, 42, 3).name == f"{_EXP}_fold_3"


def test_other_seed_is_tagged():
    assert seed_fold_dir(_EXP, 123, 0).name == f"{_EXP}_seed123"
    assert seed_fold_dir(_EXP, 123, 2).name == f"{_EXP}_seed123_fold_2"


def test_ensemble_writes_to_canonical_dirs():
    # ensemble output lands in the canonical dirs so compute_ci can pool it
    assert canonical_fold_dir(_EXP, 0).name == _EXP
    assert canonical_fold_dir(_EXP, 4).name == f"{_EXP}_fold_4"
