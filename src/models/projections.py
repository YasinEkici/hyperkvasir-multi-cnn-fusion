"""Projection layers from raw backbone features into a shared space."""

from torch import Tensor, nn


class BranchProjection(nn.Module):
    """Linear + LayerNorm + GELU projection from raw dim to shared dim."""

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)
