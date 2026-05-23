import torch
import torch.nn as nn
import pytest

from src.models.backbones import BackboneFeatureExtractor


@pytest.mark.parametrize(
    "name,expected_dim",
    [
        ("resnet50", 2048),
        ("mobilenetv2", 1280),
        ("efficientnetb0", 1280),
    ],
)
def test_backbone_dimensions(name: str, expected_dim: int) -> None:
    # Use pretrained=False for faster tests
    backbone = BackboneFeatureExtractor(name, pretrained=False, unfreeze_blocks=0)
    assert backbone.feature_dim == expected_dim

    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, expected_dim)


@pytest.mark.parametrize(
    "name",
    ["resnet50", "mobilenetv2", "efficientnetb0"],
)
def test_backbone_freeze_all(name: str) -> None:
    backbone = BackboneFeatureExtractor(name, pretrained=False, unfreeze_blocks=0)
    for param in backbone.parameters():
        assert not param.requires_grad


@pytest.mark.parametrize(
    "name",
    ["resnet50", "mobilenetv2", "efficientnetb0"],
)
def test_backbone_unfreeze_some(name: str) -> None:
    backbone = BackboneFeatureExtractor(name, pretrained=False, unfreeze_blocks=1)
    
    # Check that at least some parameters are frozen and some are unfrozen
    has_frozen = any(not p.requires_grad for p in backbone.parameters())
    has_unfrozen = any(p.requires_grad for p in backbone.parameters())
    
    assert has_frozen, "Expected some parameters to be frozen"
    assert has_unfrozen, "Expected some parameters to be unfrozen"


@pytest.mark.parametrize(
    "name",
    ["resnet50", "mobilenetv2", "efficientnetb0"],
)
def test_backbone_batchnorm_eval_when_frozen(name: str) -> None:
    backbone = BackboneFeatureExtractor(name, pretrained=False, unfreeze_blocks=0)
    
    # Set to train mode
    backbone.train()
    
    # Check all BatchNorm layers are in eval mode
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert not module.training, "Frozen BatchNorm should be in eval mode even when model.train() is called"
