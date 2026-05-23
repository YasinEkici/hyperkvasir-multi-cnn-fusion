"""MLP classifier head."""

from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """The only classifier used in the project."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = [256],
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
