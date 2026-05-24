"""Tests for src/training/ema.py."""

import copy

import pytest
import torch
import torch.nn as nn

from src.training.ema import EMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(8, 4)


def _param_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached clone of all named parameters."""
    return {name: p.data.clone() for name, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_ema_shadow_initialised_to_model_weights():
    model = _make_model()
    ema = EMA(model, decay=0.999, start_epoch=0)
    for name, param in model.named_parameters():
        assert torch.equal(ema._shadow[name], param.data)


def test_ema_shadow_is_independent_clone():
    # Mutating model weights after init must not affect shadow.
    model = _make_model()
    ema = EMA(model, decay=0.999, start_epoch=0)
    original_shadow = {k: v.clone() for k, v in ema._shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)
    for name in original_shadow:
        assert torch.equal(ema._shadow[name], original_shadow[name])


def test_ema_invalid_decay_raises():
    model = _make_model()
    with pytest.raises(ValueError):
        EMA(model, decay=1.0)
    with pytest.raises(ValueError):
        EMA(model, decay=-0.1)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update_skipped_before_start_epoch():
    model = _make_model()
    ema = EMA(model, decay=0.9, start_epoch=5)
    initial_shadow = {k: v.clone() for k, v in ema._shadow.items()}
    # Move model weights somewhere different.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(42.0)
    ema.update(model, epoch=4)  # below start_epoch — must be a no-op
    for name in initial_shadow:
        assert torch.equal(ema._shadow[name], initial_shadow[name])


def test_update_at_start_epoch_applies():
    model = _make_model()
    ema = EMA(model, decay=0.9, start_epoch=5)
    initial_shadow = {k: v.clone() for k, v in ema._shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(42.0)
    ema.update(model, epoch=5)  # at start_epoch — must update
    for name, shadow_val in ema._shadow.items():
        assert not torch.equal(shadow_val, initial_shadow[name]), (
            "Shadow was not updated at start_epoch"
        )


def test_update_formula_correct():
    """Verify shadow = decay * shadow + (1 - decay) * param."""
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    decay = 0.9
    ema = EMA(model, decay=decay, start_epoch=0)  # shadow = 1.0
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model, epoch=0)
    expected = 0.9 * 1.0 + 0.1 * 3.0  # = 1.2
    assert torch.allclose(ema._shadow["weight"], torch.full_like(model.weight, expected))


def test_shadow_converges_to_model_after_many_updates():
    """After many updates shadow must be close to the current model weights."""
    model = _make_model(seed=1)
    ema = EMA(model, decay=0.999, start_epoch=0)
    # Drive model to a fixed target and keep updating EMA.
    target = 5.0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(target)
    for epoch in range(10_000):
        ema.update(model, epoch=epoch)
    for name, shadow_val in ema._shadow.items():
        assert torch.allclose(
            shadow_val,
            torch.full_like(shadow_val, target),
            atol=1e-2,
        ), f"Shadow did not converge for parameter '{name}'"


# ---------------------------------------------------------------------------
# apply / restore
# ---------------------------------------------------------------------------

def test_apply_replaces_model_weights_with_shadow():
    model = _make_model()
    ema = EMA(model, decay=0.9, start_epoch=0)
    # Drive shadow to a known value.
    with torch.no_grad():
        for v in ema._shadow.values():
            v.fill_(7.0)
    ema.apply(model)
    for p in model.parameters():
        assert torch.all(p.data == 7.0)
    ema.restore(model)  # clean up


def test_restore_returns_original_weights_exactly():
    """After apply → restore, every parameter must be byte-identical to the pre-apply snapshot."""
    model = _make_model(seed=2)
    ema = EMA(model, decay=0.999, start_epoch=0)
    pre_apply = _param_dict(model)
    ema.apply(model)
    ema.restore(model)
    for name, original in pre_apply.items():
        restored = dict(model.named_parameters())[name].data
        assert torch.equal(restored, original), (
            f"Parameter '{name}' not byte-identical after restore"
        )


def test_double_apply_raises():
    model = _make_model()
    ema = EMA(model, decay=0.9, start_epoch=0)
    ema.apply(model)
    with pytest.raises(RuntimeError):
        ema.apply(model)
    ema.restore(model)  # clean up


def test_restore_without_apply_raises():
    model = _make_model()
    ema = EMA(model, decay=0.9, start_epoch=0)
    with pytest.raises(RuntimeError):
        ema.restore(model)


def test_apply_restore_cycle_idempotent():
    """Two apply/restore cycles must leave weights identical each time."""
    model = _make_model(seed=3)
    ema = EMA(model, decay=0.9, start_epoch=0)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)
    ema.update(model, epoch=0)

    pre = _param_dict(model)
    for _ in range(3):
        ema.apply(model)
        ema.restore(model)
    post = _param_dict(model)
    for name in pre:
        assert torch.equal(pre[name], post[name])


# ---------------------------------------------------------------------------
# start_epoch boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("epoch,should_update", [(5, False), (6, True), (7, True)])
def test_update_respects_start_epoch_boundary(epoch: int, should_update: bool):
    """start_epoch=6: epochs < 6 skip, epochs >= 6 update."""
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA(model, decay=0.5, start_epoch=6)
    initial = ema._shadow["weight"].clone()
    with torch.no_grad():
        model.weight.fill_(10.0)
    ema.update(model, epoch=epoch)
    changed = not torch.equal(ema._shadow["weight"], initial)
    assert changed == should_update, (
        f"epoch={epoch}: expected changed={should_update}, got changed={changed}"
    )
