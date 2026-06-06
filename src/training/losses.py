"""Loss function builders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification (Lin et al. 2017).

    Verbatim from Lin et al. 2017 §3.2 (ICCV), Equation 5 —
    α-balanced focal loss:

        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)                    (5)

    where p_t is the model's estimated probability for the ground-truth class
    (Equation 2), γ ≥ 0 is the focusing parameter, and α_t is the per-class
    balancing weight.

    Properties (from the paper):
    - When γ = 0, FL is equivalent to α-balanced CE (Equation 3).
    - When γ > 0, the modulating factor (1 - p_t)^γ down-weights easy
      examples (p_t > 0.5) and focuses training on hard examples.
    - The paper recommends γ = 2 with α = 0.25 for binary detection;
      "extending the focal loss to the multi-class case is straightforward"
      (footnote 1, §3).

    Multi-class extension:
        For C classes, p_t = softmax(logits)[ground-truth class].
        The per-class α vector has shape (C,); when None, no class weighting
        is applied (equivalent to α_t = 1 for all classes).

    Numerical stability:
        log(p_t) is computed via F.cross_entropy(reduction='none') to leverage
        PyTorch's fused log-softmax implementation, avoiding manual softmax +
        log which is numerically unstable.

    Note:
        Focal loss and label smoothing are NOT combined. Label smoothing
        modifies the target distribution, which conflicts with the focal
        modulating factor that relies on the true-class probability p_t.
        These two techniques address class imbalance from different angles
        and their interaction is not well-studied. Use one or the other.

    Args:
        gamma: Focusing parameter γ ≥ 0. Default: 2.0.
        alpha: Optional per-class weight tensor of shape (C,). When provided,
            each sample's loss is scaled by α[y] where y is the ground-truth
            class. When None, no class weighting is applied.
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'.

    Reference:
        Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal Loss for Dense Object Detection. ICCV, 2980-2988.
        references/methodology_imbalance/focal_loss_lin_2017_iccv/paper.md
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        if alpha is not None:
            # Register as buffer so it moves with .to(device) automatically.
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw (unnormalized) predictions, shape (N, C).
            labels: Ground-truth class indices, shape (N,), dtype long.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss
            tensor of shape (N,) if reduction='none'.
        """
        # -log(p_t) via fused log-softmax (numerically stable).
        ce = F.cross_entropy(logits, labels, reduction="none")  # (N,)

        # p_t = exp(-CE) = softmax(logits)[ground-truth class].
        # Using exp(-ce) is equivalent and avoids a separate softmax call.
        p_t = torch.exp(-ce)  # (N,)

        # Modulating factor: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma  # (N,)

        # FL(p_t) = -(1 - p_t)^γ log(p_t) = (1 - p_t)^γ * CE
        loss = focal_weight * ce  # (N,)

        # Optional α-balancing: scale by α[y] per sample.
        if self.alpha is not None:
            alpha_t = self.alpha[labels]  # (N,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(config: dict) -> nn.Module:
    """Build a loss function from config dict.

    Supported types: "cross_entropy", "ce_label_smooth", "focal",
    "focal_balanced".
    """
    loss_type = config.get("type", "cross_entropy")
    if loss_type in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss()
    if loss_type in ("ce_label_smooth", "label_smooth"):
        smoothing = float(config.get("label_smoothing", 0.1))
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if loss_type == "focal":
        gamma = float(config.get("gamma", 2.0))
        return FocalLoss(gamma=gamma, alpha=None)
    if loss_type == "focal_balanced":
        gamma = float(config.get("gamma", 2.0))
        # alpha must be passed at runtime from training-fold class frequencies.
        # build_loss creates the FocalLoss with alpha=None; the caller is
        # responsible for setting alpha via set_alpha_from_counts() or by
        # passing a pre-computed tensor through the config.
        alpha = config.get("alpha")
        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"Unknown loss type: {loss_type}")
