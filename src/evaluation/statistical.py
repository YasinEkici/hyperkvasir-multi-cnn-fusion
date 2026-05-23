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
    raise NotImplementedError("Bootstrap CI is not implemented in the scaffold.")
