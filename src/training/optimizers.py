"""Optimizer builders."""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer from config dict.

    Config keys: head_lr, weight_decay.
    """
    lr = float(config.get("head_lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_adamw_with_llrd(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
) -> torch.optim.Optimizer:
    """AdamW with layer-wise learning rate decay (for fine-tuning, Week 2).

    Backbone parameter groups get exponentially decayed LR; head params use head_lr.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbones") or name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(param_groups)
