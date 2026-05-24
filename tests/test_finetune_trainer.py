"""Smoke tests for the fine-tune Trainer extensions (EMA + CutMix).

These tests use tiny synthetic models and TensorDataset loaders so they
run without image data or GPU.  The frozen code path (ema=None, cutmix_prob=0)
is also covered to catch regressions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.ema import EMA
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
FEAT_DIM = 16
IMG_C, IMG_H, IMG_W = 3, 16, 16  # tiny spatial size for CutMix tests
N_SAMPLES = 12
BATCH = 4


def _tiny_model() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(FEAT_DIM, NUM_CLASSES)


def _tiny_loader(n: int = N_SAMPLES) -> DataLoader:
    torch.manual_seed(1)
    x = torch.randn(n, FEAT_DIM)
    y = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH)


def _tiny_image_model() -> nn.Module:
    """Tiny CNN-free model that maps (B, C*H*W) -> (B, NUM_CLASSES)."""
    torch.manual_seed(2)
    return nn.Linear(IMG_C * IMG_H * IMG_W, NUM_CLASSES)


def _tiny_image_loader(n: int = N_SAMPLES) -> DataLoader:
    """Returns (B, C, H, W) image tensors — required by apply_cutmix."""
    torch.manual_seed(3)
    # Model flattens images, so we wrap it below; loader produces 4D tensors.
    x = torch.randn(n, IMG_C, IMG_H, IMG_W)
    y = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH)


class _FlattenLinear(nn.Module):
    """Linear classifier that accepts (B, C, H, W) by flattening first."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(IMG_C * IMG_H * IMG_W, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _make_trainer(
    model: nn.Module,
    run_dir: Path,
    *,
    ema: EMA | None = None,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    epochs: int = 2,
) -> Trainer:
    return Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        run_dir=run_dir,
        device="cpu",
        num_classes=NUM_CLASSES,
        early_stopping_patience=10,
        mixed_precision=False,
        ema=ema,
        cutmix_alpha=cutmix_alpha,
        cutmix_prob=cutmix_prob,
    )


# ---------------------------------------------------------------------------
# Constructor / attribute tests
# ---------------------------------------------------------------------------

def test_trainer_accepts_ema_param():
    model = _tiny_model()
    ema = EMA(model, decay=0.999, start_epoch=0)
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), ema=ema)
        assert t.ema is ema


def test_trainer_accepts_cutmix_params():
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), cutmix_alpha=1.0, cutmix_prob=0.5)
        assert t.cutmix_alpha == pytest.approx(1.0)
        assert t.cutmix_prob == pytest.approx(0.5)


def test_trainer_default_no_ema_no_cutmix():
    """Default Trainer must have ema=None and cutmix_prob=0 (frozen path unchanged)."""
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp))
        assert t.ema is None
        assert t.cutmix_prob == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_loss
# ---------------------------------------------------------------------------

def test_compute_loss_no_cutmix_finite():
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp))
        x = torch.randn(BATCH, FEAT_DIM)
        y = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = t._compute_loss(x, y)
        assert torch.isfinite(loss)


def test_compute_loss_cutmix_always_finite():
    """With cutmix_prob=1.0, every call applies CutMix; loss must be finite.

    CutMix requires 4D image tensors [B, C, H, W] — use _FlattenLinear model
    that accepts images and flattens internally.
    """
    model = _FlattenLinear()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), cutmix_alpha=1.0, cutmix_prob=1.0)
        x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W)
        y = torch.randint(0, NUM_CLASSES, (BATCH,))
        for _ in range(10):
            loss = t._compute_loss(x, y)
            assert torch.isfinite(loss), f"CutMix loss not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# _train_epoch
# ---------------------------------------------------------------------------

def test_train_epoch_no_cutmix_returns_finite_loss():
    model = _tiny_model()
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp))
        loss = t._train_epoch(loader)
        assert torch.isfinite(torch.tensor(loss))


def test_train_epoch_cutmix_returns_finite_loss():
    """_train_epoch with cutmix_prob=1.0 requires a 4D image loader."""
    model = _FlattenLinear()
    loader = _tiny_image_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), cutmix_alpha=1.0, cutmix_prob=1.0)
        loss = t._train_epoch(loader)
        assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# fit() — frozen path regression
# ---------------------------------------------------------------------------

def test_fit_frozen_path_history_length():
    """fit() without EMA/CutMix must return one record per epoch (no regression)."""
    model = _tiny_model()
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp))
        history = t.fit(loader, loader, epochs=3)
        assert len(history) == 3
        for rec in history:
            assert "epoch" in rec
            assert "train_loss" in rec


def test_fit_frozen_path_saves_best_pt():
    model = _tiny_model()
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp))
        t.fit(loader, loader, epochs=2)
        assert (Path(tmp) / "best.pt").exists()


# ---------------------------------------------------------------------------
# fit() — EMA integration
# ---------------------------------------------------------------------------

def test_fit_with_ema_no_crash():
    """EMA apply/update/restore must not raise during a short fit."""
    model = _tiny_model()
    ema = EMA(model, decay=0.9, start_epoch=0)
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), ema=ema)
        t.fit(loader, loader, epochs=3)


def test_fit_ema_not_applied_before_start_epoch():
    """Model weights must not be reverted to init during epochs < start_epoch."""
    model = _tiny_model()
    # start_epoch=100 → EMA never activates in this short fit
    ema = EMA(model, decay=0.999, start_epoch=100)
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), ema=ema)
        # Record initial weights
        init_w = {n: p.data.clone() for n, p in model.named_parameters()}
        t.fit(loader, loader, epochs=2)
        # Weights must have changed (training happened) — not reverted to init
        final_w = {n: p.data for n, p in model.named_parameters()}
        any_changed = any(not torch.equal(init_w[n], final_w[n]) for n in init_w)
        assert any_changed, "Weights unchanged — EMA may have incorrectly reverted to init"


def test_fit_ema_restore_leaves_live_weights_after_fit():
    """After fit(), ema._backup must be None (apply/restore balanced)."""
    model = _tiny_model()
    ema = EMA(model, decay=0.9, start_epoch=0)
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), ema=ema)
        t.fit(loader, loader, epochs=3)
        assert ema._backup is None, "EMA backup not cleared — apply/restore unbalanced"


# ---------------------------------------------------------------------------
# fit() — CutMix integration
# ---------------------------------------------------------------------------

def test_fit_with_cutmix_no_crash():
    """fit() with cutmix_prob=1.0 requires a 4D image loader."""
    model = _FlattenLinear()
    loader = _tiny_image_loader()
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_trainer(model, Path(tmp), cutmix_alpha=1.0, cutmix_prob=1.0)
        history = t.fit(loader, loader, epochs=2)
        assert len(history) == 2
