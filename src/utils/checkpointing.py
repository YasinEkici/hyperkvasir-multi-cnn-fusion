"""Checkpoint save/load utilities."""

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(Path(path), weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    return model
