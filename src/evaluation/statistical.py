"""Statistical evaluation utilities."""

from collections.abc import Callable

import numpy as np


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    exact_threshold: int = 25,
) -> dict:
    raise NotImplementedError("McNemar testing is not implemented in the scaffold.")


def bootstrap_ci(
    metric_fn: Callable,
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap.

    Resamples (preds, labels) with replacement n_bootstrap times.
    point_estimate = metric_fn(preds, labels) on the original data.
    ci_low / ci_high = (1-ci)/2 and 1-(1-ci)/2 percentiles of the bootstrap
    distribution, i.e. a two-sided percentile-method interval.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    point_estimate = float(metric_fn(preds, labels))
    n = len(preds)
    boot_scores = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(preds[idx], labels[idx])
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_scores, 100.0 * alpha))
    ci_high = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    return point_estimate, ci_low, ci_high
