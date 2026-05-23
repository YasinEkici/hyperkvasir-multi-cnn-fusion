"""Low-rank Multimodal Fusion stretch placeholder."""

from torch import Tensor, nn


class FusionModule(nn.Module):
    def __init__(self, num_branches: int, feature_dim: int, **kwargs):
        super().__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        raise NotImplementedError("LMF is a stretch feature and is not implemented.")

    @property
    def output_dim(self) -> int:
        return self.feature_dim
