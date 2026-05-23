"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Seed Python, NumPy, PyTorch CPU/GPU, and configure cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding function."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
