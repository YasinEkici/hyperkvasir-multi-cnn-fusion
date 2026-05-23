"""Metric computation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    preds: np.ndarray | list,
    labels: np.ndarray | list,
    *,
    num_classes: int | None = None,
) -> dict:
    """Compute classification metrics, safe against zero-support classes.

    Returns a dict with scalar metrics and per-class breakdowns.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    accuracy = float(accuracy_score(labels, preds))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    zero_support_classes = [int(i) for i, s in enumerate(support) if s == 0]

    per_class = {
        int(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(precision))
    }

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)) if num_classes else None)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "zero_support_classes": zero_support_classes,
        "confusion_matrix": cm.tolist(),
    }
