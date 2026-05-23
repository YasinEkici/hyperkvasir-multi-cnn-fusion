"""Loss function builders."""

import torch.nn as nn


def build_loss(config: dict) -> nn.Module:
    """Build a loss function from config dict.

    Supported types: "cross_entropy", "ce_label_smooth".
    """
    loss_type = config.get("type", "cross_entropy")
    if loss_type in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss()
    if loss_type in ("ce_label_smooth", "label_smooth"):
        smoothing = float(config.get("label_smoothing", 0.1))
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    raise ValueError(f"Unknown loss type: {loss_type}")
