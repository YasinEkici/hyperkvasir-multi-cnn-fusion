"""Scheduler builders."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
    total_epochs: int,
) -> LambdaLR:
    """Build a cosine scheduler with linear warmup.

    Config keys: type, warmup_epochs.
    """
    sched_type = config.get("type", "cosine_with_warmup")
    warmup_epochs = int(config.get("warmup_epochs", 5))

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    if sched_type in ("cosine_with_warmup", "cosine"):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler type: {sched_type}")
