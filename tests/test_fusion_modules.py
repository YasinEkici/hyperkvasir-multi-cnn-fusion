import torch
from src.models.fusion import aff, concat, gmu, lmf, weighted


def test_fusion_module_output_dims() -> None:
    assert concat.FusionModule(3, 512).output_dim == 1536
    assert weighted.FusionModule(3, 512).output_dim == 512
    assert gmu.FusionModule(3, 512).output_dim == 512
    assert aff.FusionModule(3, 512).output_dim == 512
    assert lmf.FusionModule(3, 512).output_dim == 512


def test_concat_fusion_forward() -> None:
    fusion = concat.FusionModule(num_branches=3, feature_dim=512)
    features = [torch.randn(2, 512) for _ in range(3)]
    out = fusion(features)
    assert out.shape == (2, 1536)


def test_weighted_fusion_forward() -> None:
    fusion = weighted.FusionModule(num_branches=3, feature_dim=512)
    features = [torch.randn(2, 512) for _ in range(3)]
    out = fusion(features)
    assert out.shape == (2, 512)
