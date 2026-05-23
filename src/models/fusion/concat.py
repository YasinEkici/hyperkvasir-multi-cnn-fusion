"""Concatenation fusion."""

from torch import Tensor, cat, nn


class FusionModule(nn.Module):
    def __init__(self, num_branches: int, feature_dim: int, **kwargs):
        super().__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        return cat(features, dim=1)

    @property
    def output_dim(self) -> int:
        return self.num_branches * self.feature_dim
