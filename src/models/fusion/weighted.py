"""Learned weighted feature fusion."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FusionModule(nn.Module):
    def __init__(self, num_branches: int, feature_dim: int, **kwargs):
        super().__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim
        # Learnable weights initialized to zeros for uniform start
        self.branch_weights = nn.Parameter(torch.zeros(num_branches))

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} features, got {len(features)}")
        
        # Stack features: (B, num_branches, D)
        stacked_features = torch.stack(features, dim=1)
        
        # Apply softmax to weights: (num_branches,)
        weights = F.softmax(self.branch_weights, dim=0)
        
        # Reshape weights for broadcasting: (1, num_branches, 1)
        weights = weights.view(1, self.num_branches, 1)
        
        # Weighted sum: (B, D)
        output = (stacked_features * weights).sum(dim=1)
        return output

    @property
    def output_dim(self) -> int:
        return self.feature_dim
