"""Tests for the GMU fusion module (src/models/fusion/gmu.py).

Covers:
  - output shape for N=1, 2, 3 branches
  - output_dim property
  - gate weights sum to 1 (softmax guarantee)
  - no parameter overlap between branch_transforms and gate
  - B=1 batch edge case
  - gradient flows through all parameters
  - ValueError on wrong number of input tensors
  - FrozenHeadModel triple GMU forward pass (shape check)
  - FrozenHeadModel pair GMU forward pass (shape check)
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion.gmu import FusionModule
from scripts.train import FrozenHeadModel

BATCH = 4
D = 512
NUM_CLASSES = 23

R = 2048
M = 1280
E = 1280


# ---------------------------------------------------------------------------
# Unit tests: FusionModule
# ---------------------------------------------------------------------------

def _make_features(num_branches, batch=BATCH, dim=D):
    return [torch.randn(batch, dim) for _ in range(num_branches)]


def test_output_shape_n1():
    m = FusionModule(num_branches=1, feature_dim=D)
    out = m(_make_features(1))
    assert out.shape == (BATCH, D)


def test_output_shape_n2():
    m = FusionModule(num_branches=2, feature_dim=D)
    out = m(_make_features(2))
    assert out.shape == (BATCH, D)


def test_output_shape_n3():
    m = FusionModule(num_branches=3, feature_dim=D)
    out = m(_make_features(3))
    assert out.shape == (BATCH, D)


def test_output_dim_property():
    m = FusionModule(num_branches=3, feature_dim=D)
    assert m.output_dim == D


def test_gate_weights_sum_to_one():
    """softmax gate must produce weights that sum to 1 across branches."""
    m = FusionModule(num_branches=3, feature_dim=D)
    features = _make_features(3)
    gate_input = torch.cat(features, dim=-1)
    import torch.nn.functional as F
    z = F.softmax(m.gate(gate_input), dim=-1)
    sums = z.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


def test_no_param_overlap_branch_transforms_vs_gate():
    """branch_transforms and gate must not share any Parameter objects."""
    m = FusionModule(num_branches=3, feature_dim=D)
    bt_ids = {id(p) for p in m.branch_transforms.parameters()}
    gate_ids = {id(p) for p in m.gate.parameters()}
    assert bt_ids.isdisjoint(gate_ids)


def test_batch_size_one():
    m = FusionModule(num_branches=2, feature_dim=D)
    out = m(_make_features(2, batch=1))
    assert out.shape == (1, D)


def test_gradients_flow_all_params():
    m = FusionModule(num_branches=3, feature_dim=D)
    features = _make_features(3)
    out = m(features)
    loss = out.sum()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_wrong_num_features_raises():
    m = FusionModule(num_branches=3, feature_dim=D)
    with pytest.raises(ValueError, match="Expected 3 feature tensors, got 2"):
        m(_make_features(2))


# ---------------------------------------------------------------------------
# Integration: FrozenHeadModel with GMU
# ---------------------------------------------------------------------------

def _make_frozen(backbone_names, fusion_type):
    return FrozenHeadModel(
        backbone_names=backbone_names,
        projection_dim=D,
        fusion_type=fusion_type,
        num_classes=NUM_CLASSES,
        mlp_hidden=[256],
        dropout=0.0,
    )


def test_frozen_head_triple_gmu_output_shape():
    model = _make_frozen(["resnet50", "mobilenetv2", "efficientnetb0"], "gmu")
    x = torch.randn(BATCH, R + M + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_frozen_head_pair_gmu_output_shape():
    model = _make_frozen(["mobilenetv2", "efficientnetb0"], "gmu")
    x = torch.randn(BATCH, M + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_frozen_head_triple_gmu_output_dim_matches_projection():
    """GMU output_dim == projection_dim, not N*projection_dim."""
    model = _make_frozen(["resnet50", "mobilenetv2", "efficientnetb0"], "gmu")
    assert model.fusion.output_dim == D
