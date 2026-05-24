"""Tests for build_adamw_with_llrd in src/training/optimizers.py."""

import pytest
import torch
import torch.nn as nn

from src.models.backbones import BackboneFeatureExtractor
from src.training.optimizers import build_adamw_with_llrd

HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
DECAY = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel(nn.Module):
    """Minimal stand-in for MultiCNNFusionClassifier.

    Has model.backbones (nn.ModuleDict of BackboneFeatureExtractor) and a
    small head linear layer, which is the minimum structure that
    build_adamw_with_llrd expects.
    """

    def __init__(self, backbone_names: list[str], unfreeze_blocks: int):
        super().__init__()
        self.backbones = nn.ModuleDict({
            name: BackboneFeatureExtractor(
                name=name, pretrained=False, unfreeze_blocks=unfreeze_blocks
            )
            for name in backbone_names
        })
        # Represents projection / fusion / classifier params (head).
        self.head = nn.Linear(512, 23)

    def forward(self, x):  # not called in tests
        raise NotImplementedError


def _build_opt(backbone_names: list[str], unfreeze_blocks: int):
    model = _FakeModel(backbone_names, unfreeze_blocks)
    opt = build_adamw_with_llrd(
        model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR,
        weight_decay=1e-4, llrd_decay=DECAY,
    )
    return model, opt


def _all_group_param_ids(opt) -> list[int]:
    return [id(p) for g in opt.param_groups for p in g["params"]]


def _all_model_requires_grad_ids(model: nn.Module) -> set[int]:
    return {id(p) for p in model.parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# Coverage: every requires_grad param in exactly one group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone_names,unfreeze_blocks", [
    (["resnet50"], 1),
    (["resnet50"], 2),
    (["mobilenetv2"], 1),
    (["efficientnetb0"], 1),
    (["resnet50", "mobilenetv2", "efficientnetb0"], 2),
])
def test_no_param_double_counted(backbone_names, unfreeze_blocks):
    model, opt = _build_opt(backbone_names, unfreeze_blocks)
    all_ids = _all_group_param_ids(opt)
    assert len(all_ids) == len(set(all_ids)), "Duplicate param found across groups"


@pytest.mark.parametrize("backbone_names,unfreeze_blocks", [
    (["resnet50"], 1),
    (["resnet50"], 2),
    (["mobilenetv2"], 1),
    (["efficientnetb0"], 1),
    (["resnet50", "mobilenetv2", "efficientnetb0"], 2),
])
def test_all_requires_grad_params_covered(backbone_names, unfreeze_blocks):
    """Every requires_grad param in the model must appear in exactly one group."""
    model, opt = _build_opt(backbone_names, unfreeze_blocks)
    expected = _all_model_requires_grad_ids(model)
    actual = set(_all_group_param_ids(opt))
    assert actual == expected, (
        f"Missing: {expected - actual}, Extra: {actual - expected}"
    )


# ---------------------------------------------------------------------------
# No empty groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone_names,unfreeze_blocks", [
    (["resnet50"], 1),
    (["resnet50"], 2),
    (["mobilenetv2"], 2),
    (["efficientnetb0"], 2),
    (["resnet50", "mobilenetv2", "efficientnetb0"], 2),
])
def test_no_empty_groups(backbone_names, unfreeze_blocks):
    _, opt = _build_opt(backbone_names, unfreeze_blocks)
    for i, g in enumerate(opt.param_groups):
        assert len(g["params"]) > 0, f"Group {i} is empty"


# ---------------------------------------------------------------------------
# Group count matches expected block count
#
# ResNet50 _freeze_parameters: start_idx = max(4, 8 - unfreeze_blocks)
#   unfreeze_blocks=1 → start_idx=7 → backbone[7:] → layer4 only → 1 backbone block
#   unfreeze_blocks=2 → start_idx=6 → backbone[6:] → layer3, layer4 → 2 backbone blocks
#   unfreeze_blocks=3 → start_idx=5 → backbone[5:] → layer2..4 → 3 backbone blocks
#   unfreeze_blocks=4 → start_idx=4 → backbone[4:] → layer1..4 → 4 backbone blocks
# Head is always 1 group (model.head params).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unfreeze_blocks,expected_backbone_groups", [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_resnet50_group_count(unfreeze_blocks, expected_backbone_groups):
    _, opt = _build_opt(["resnet50"], unfreeze_blocks)
    # 1 head group + expected_backbone_groups
    assert len(opt.param_groups) == 1 + expected_backbone_groups


def test_fully_frozen_backbone_only_head_group():
    """unfreeze_blocks=0 → backbone has no requires_grad params → only head group."""
    _, opt = _build_opt(["resnet50"], unfreeze_blocks=0)
    assert len(opt.param_groups) == 1
    lrs = {g["lr"] for g in opt.param_groups}
    assert HEAD_LR in lrs


# ---------------------------------------------------------------------------
# LR values: deepest block == backbone_lr, decreasing toward stem
# ---------------------------------------------------------------------------

def test_deepest_block_lr_equals_backbone_lr():
    """The deepest backbone block (depth=0) must have LR == backbone_lr."""
    _, opt = _build_opt(["resnet50"], unfreeze_blocks=1)
    # With 1 block, the single backbone group must be at backbone_lr.
    backbone_lrs = [g["lr"] for g in opt.param_groups if g["lr"] != HEAD_LR]
    assert backbone_lrs, "No backbone groups found"
    assert max(backbone_lrs) == pytest.approx(BACKBONE_LR)


@pytest.mark.parametrize("backbone_name", ["resnet50", "mobilenetv2", "efficientnetb0"])
def test_backbone_lrs_decrease_toward_stem(backbone_name):
    """LRs across backbone groups must be non-increasing (deeper = higher LR)."""
    _, opt = _build_opt([backbone_name], unfreeze_blocks=3)
    backbone_lrs = sorted(
        [g["lr"] for g in opt.param_groups if g["lr"] != HEAD_LR],
        reverse=True,
    )
    assert backbone_lrs, "No backbone groups found"
    for i in range(len(backbone_lrs) - 1):
        assert backbone_lrs[i] >= backbone_lrs[i + 1], (
            f"LR not monotonically decreasing: {backbone_lrs}"
        )


def test_head_params_at_head_lr():
    """Projection/classifier params must be in a group with lr == head_lr."""
    model, opt = _build_opt(["resnet50"], unfreeze_blocks=1)
    head_param_ids = {id(p) for p in model.head.parameters()}
    for g in opt.param_groups:
        group_ids = {id(p) for p in g["params"]}
        if head_param_ids & group_ids:
            assert g["lr"] == pytest.approx(HEAD_LR), (
                f"Head params found in group with lr={g['lr']}, expected {HEAD_LR}"
            )


# ---------------------------------------------------------------------------
# Triple backbone: no cross-backbone parameter overlap
# ---------------------------------------------------------------------------

def test_triple_backbone_no_double_counting():
    model, opt = _build_opt(
        ["resnet50", "mobilenetv2", "efficientnetb0"], unfreeze_blocks=2
    )
    all_ids = _all_group_param_ids(opt)
    assert len(all_ids) == len(set(all_ids)), "Cross-backbone param overlap detected"
    assert set(all_ids) == _all_model_requires_grad_ids(model)
