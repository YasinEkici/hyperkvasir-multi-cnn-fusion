"""Tests for bootstrap_ci in src/evaluation/statistical.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical import bootstrap_ci

# ---------------------------------------------------------------------------
# Synthetic data — 4-class, 200 samples, moderate noise (~70% correct)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)
_TRUE = _RNG.integers(0, 4, size=200)
_NOISY = np.where(_RNG.random(200) > 0.30, _TRUE, _RNG.integers(0, 4, size=200))


def _macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return compute_metrics(preds, labels)["macro_f1"]


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------

def test_ci_brackets_point_estimate():
    point, lo, hi = bootstrap_ci(_macro_f1, _NOISY, _TRUE)
    assert lo <= point <= hi, (
        f"CI [{lo:.4f}, {hi:.4f}] does not bracket point estimate {point:.4f}"
    )


def test_ci_width_positive():
    _, lo, hi = bootstrap_ci(_macro_f1, _NOISY, _TRUE)
    assert hi > lo, "CI width must be > 0"


def test_ci_returns_three_floats():
    result = bootstrap_ci(_macro_f1, _NOISY, _TRUE)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result), "All returned values must be float"


def test_ci_bounds_in_unit_interval():
    point, lo, hi = bootstrap_ci(_macro_f1, _NOISY, _TRUE)
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_ci_deterministic_same_seed():
    r1 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, seed=42)
    r2 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, seed=42)
    assert r1 == r2, "Same seed must produce identical results"


def test_ci_different_seeds_differ():
    _, lo1, hi1 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, seed=0)
    _, lo2, hi2 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, seed=99)
    assert (lo1, hi1) != (lo2, hi2), "Different seeds should (almost certainly) give different CIs"


# ---------------------------------------------------------------------------
# n_bootstrap is respected
# ---------------------------------------------------------------------------

def test_ci_n_bootstrap_affects_result():
    """Different n_bootstrap values produce different bootstrap distributions."""
    point_small, lo_small, hi_small = bootstrap_ci(
        _macro_f1, _NOISY, _TRUE, n_bootstrap=50, seed=42
    )
    point_large, lo_large, hi_large = bootstrap_ci(
        _macro_f1, _NOISY, _TRUE, n_bootstrap=1000, seed=42
    )
    # Point estimate is identical (same original data)
    assert point_small == point_large
    # Bootstrap bounds differ because resampling differs
    assert (lo_small, hi_small) != (lo_large, hi_large)


def test_ci_larger_n_bootstrap_not_wider_on_average():
    """Larger n_bootstrap should give a narrower or equal CI (law of large numbers)."""
    _, lo_small, hi_small = bootstrap_ci(
        _macro_f1, _NOISY, _TRUE, n_bootstrap=100, seed=42
    )
    _, lo_large, hi_large = bootstrap_ci(
        _macro_f1, _NOISY, _TRUE, n_bootstrap=2000, seed=42
    )
    # The 2000-sample CI should not be wider than the 100-sample CI
    # (allow a small tolerance for randomness)
    assert (hi_large - lo_large) <= (hi_small - lo_small) + 0.05


# ---------------------------------------------------------------------------
# CI level parameter
# ---------------------------------------------------------------------------

def test_ci_90_narrower_than_95():
    """90% CI must be strictly narrower than 95% CI (same data, same seed)."""
    _, lo90, hi90 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, ci=0.90, seed=42)
    _, lo95, hi95 = bootstrap_ci(_macro_f1, _NOISY, _TRUE, ci=0.95, seed=42)
    assert (hi90 - lo90) <= (hi95 - lo95) + 1e-9
