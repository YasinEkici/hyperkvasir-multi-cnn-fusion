"""Gated Multimodal Unit (GMU) fusion.

Implements the Gated Multimodal Unit from:
    Arevalo et al. (2017), "Gated Multimodal Units for Information Fusion",
    ICLR Workshop. Section 3.1.

Bimodal formulation (verbatim from paper §3.1):

    h_v = tanh(W_v · x_v)
    h_t = tanh(W_t · x_t)
    z   = σ(W_z · [x_v, x_t])
    h   = z * h_v + (1−z) * h_t
    Θ   = {W_v, W_t, W_z}

    with [·,·] the concatenation operator.

N-branch softmax generalization used in this project (N ≥ 1):

    h_i = tanh(W_i · x_i)                        for i = 1 .. N
    z   = softmax(W_z · concat([x_1, ..., x_N])) → (B, N)
    h   = Σ_i  z_i * h_i                         → (B, feature_dim)

The gate W_z receives the raw (pre-transform) branch features as input,
matching the paper specification: "each gate neuron receives as input the
feature vectors from all the modalities" (Arevalo et al. §3.1).

The softmax replaces the bimodal tied σ / (1−σ) pair with a normalised
N-way competition, preserving the property that gate weights sum to 1.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FusionModule(nn.Module):
    """Gated Multimodal Unit fusion for N input branches.

    Each branch feature is linearly transformed and passed through tanh.
    A shared gate network receives the concatenation of all raw branch
    features and outputs a softmax-normalised weight vector that blends
    the transformed branches into a single representation.

    Args:
        num_branches: Number of input feature branches (N ≥ 1).
        feature_dim:  Dimensionality of each branch's projected vector.
        **kwargs:     Ignored; present for interface compatibility.

    Input:
        features: list of N tensors, each (B, feature_dim).

    Output:
        Fused tensor of shape (B, feature_dim).
    """

    def __init__(self, num_branches: int, feature_dim: int, **kwargs) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim

        # W_i: per-branch linear transform — h_i = tanh(W_i · x_i)
        self.branch_transforms = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(num_branches)]
        )

        # W_z: gate — z = softmax(W_z · concat([x_1, ..., x_N]))
        self.gate = nn.Linear(num_branches * feature_dim, num_branches)

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} feature tensors, got {len(features)}."
            )

        # h_i = tanh(W_i · x_i)
        h_list = [
            torch.tanh(self.branch_transforms[i](features[i]))
            for i in range(self.num_branches)
        ]

        # z = softmax(W_z · concat([x_1, ..., x_N]))
        # Gate receives raw pre-transform features (paper §3.1).
        gate_input = torch.cat(features, dim=-1)           # (B, N * D)
        z = F.softmax(self.gate(gate_input), dim=-1)        # (B, N)

        # h = Σ_i  z_i * h_i
        h_stack = torch.stack(h_list, dim=1)                # (B, N, D)
        output = (z.unsqueeze(-1) * h_stack).sum(dim=1)     # (B, D)
        return output

    @property
    def output_dim(self) -> int:
        return self.feature_dim
