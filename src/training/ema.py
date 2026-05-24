"""Exponential Moving Average (EMA) for model parameters.

Maintains a shadow copy of model parameters updated as:
    shadow[k] = decay * shadow[k] + (1 - decay) * param[k]

Only nn.Module named parameters are tracked (not buffers such as
BatchNorm running_mean / running_var, which must stay frozen per
project_structure.md §2.2).

Usage in a training loop::

    ema = EMA(model, decay=0.999, start_epoch=5)
    for epoch in range(num_epochs):
        train(model)
        ema.update(model, epoch)   # no-op before start_epoch
        ema.apply(model)
        val_metrics = evaluate(model)
        ema.restore(model)
        # checkpoint model with shadow weights already applied, or
        # re-apply ema.apply() before saving best.pt / ema.pt
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


class EMA:
    """Exponential moving average of model parameters.

    Args:
        model:       The model whose parameters to track. Shadow weights
                     are initialised to a clone of the model's current
                     parameter values.
        decay:       EMA decay coefficient (e.g. 0.999). Higher = slower
                     update, smoother shadow weights.
        start_epoch: EMA updates are skipped for epochs strictly less than
                     this value, giving the model time to warm up first.
    """

    def __init__(self, model: nn.Module, decay: float, start_epoch: int = 0) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.start_epoch = start_epoch
        # Shadow weights: plain dict, keyed by parameter name.
        # Cloned so shadow is independent of the live model from the start.
        self._shadow: dict[str, Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }
        # Backup storage used by apply() / restore() — None when not applied.
        self._backup: dict[str, Tensor] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, model: nn.Module, epoch: int) -> None:
        """Update shadow weights from the current model parameters.

        No-op when ``epoch < self.start_epoch``.  Must be called with
        the *live* model weights (i.e. before ``apply()`` or after
        ``restore()``).

        Args:
            model: The model being trained (live weights).
            epoch: Current epoch index (0-based).
        """
        if epoch < self.start_epoch:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self._shadow:
                    # New parameter added after EMA init — skip gracefully.
                    continue
                shadow = self._shadow[name]
                # Shadow is initialised on CPU; model may be on GPU.  Move
                # lazily on first real update so in-place ops stay same-device.
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    self._shadow[name] = shadow
                shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Swap shadow weights into the model for evaluation.

        Stores the current live weights as a backup so ``restore()`` can
        recover them exactly. Calling ``apply()`` twice without an
        intervening ``restore()`` raises RuntimeError.

        Args:
            model: Model whose parameters will be replaced with shadow weights.
        """
        if self._backup is not None:
            raise RuntimeError(
                "EMA.apply() called while shadow weights are already applied. "
                "Call EMA.restore() first."
            )
        # Clone live weights into backup before overwriting.
        self._backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        # Copy shadow weights into model in-place.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._shadow:
                    param.data.copy_(self._shadow[name])

    def reset_shadow(self, model: nn.Module) -> None:
        """Warm-start: copy current model parameters into shadow.

        Called once in the trainer at the epoch when EMA first becomes active.
        Without this, shadow still holds the model's random-init weights from
        EMA.__init__, and applying it would collapse validation metrics.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self._shadow:
                    continue
                if self._shadow[name].device != param.device:
                    self._shadow[name] = param.data.clone().to(param.device)
                else:
                    self._shadow[name].copy_(param.data)

    def restore(self, model: nn.Module) -> None:
        """Restore the live weights saved by ``apply()``.

        After this call the model's parameters are byte-identical to what
        they were before ``apply()`` was invoked.

        Args:
            model: Model to restore (must be the same one passed to ``apply()``).
        """
        if self._backup is None:
            raise RuntimeError(
                "EMA.restore() called without a prior EMA.apply()."
            )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup[name])
        self._backup = None
