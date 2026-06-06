"""Tests for FocalLoss and build_loss focal extensions.

Acceptance criteria from exec plan §8, Track A:
- γ=0 reduces to weighted CE (or standard CE when α=None)
- γ>0 down-weights easy examples (high p_t) relative to hard examples
- α-vector shape matches class count
- No NaN at p→0 and p→1 (numerical stability)
- Gradient flows through the loss
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss, build_loss


# ---------------------------------------------------------------------------
# γ=0 equivalence to CE
# ---------------------------------------------------------------------------


class TestGammaZeroEquivalsToCE:
    """When γ=0, focal loss must reduce to standard cross-entropy."""

    def test_gamma_zero_no_alpha_equals_ce(self) -> None:
        """FL(γ=0, α=None) ≡ CE for random logits."""
        torch.manual_seed(42)
        logits = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))

        focal = FocalLoss(gamma=0.0, alpha=None, reduction="mean")
        ce = F.cross_entropy(logits, labels, reduction="mean")

        torch.testing.assert_close(focal(logits, labels), ce, atol=1e-6, rtol=1e-5)

    def test_gamma_zero_no_alpha_equals_ce_per_sample(self) -> None:
        """FL(γ=0, α=None, reduction='none') ≡ CE per sample."""
        torch.manual_seed(123)
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))

        focal = FocalLoss(gamma=0.0, alpha=None, reduction="none")
        ce = F.cross_entropy(logits, labels, reduction="none")

        torch.testing.assert_close(focal(logits, labels), ce, atol=1e-6, rtol=1e-5)

    def test_gamma_zero_with_uniform_alpha_equals_scaled_ce(self) -> None:
        """FL(γ=0, α=[1,1,...]) ≡ CE (uniform α doesn't change relative losses)."""
        torch.manual_seed(7)
        C = 8
        logits = torch.randn(24, C)
        labels = torch.randint(0, C, (24,))

        alpha = torch.ones(C)
        focal = FocalLoss(gamma=0.0, alpha=alpha, reduction="mean")
        ce = F.cross_entropy(logits, labels, reduction="mean")

        torch.testing.assert_close(focal(logits, labels), ce, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# γ>0 down-weights easy examples
# ---------------------------------------------------------------------------


class TestFocusingEffect:
    """γ>0 must reduce loss for well-classified (easy) examples."""

    def test_easy_example_loss_reduced(self) -> None:
        """An easy example (high p_t) should have lower loss with γ>0 than γ=0."""
        # Create a logit where class 0 has very high confidence.
        logits = torch.tensor([[10.0, -5.0, -5.0]])
        labels = torch.tensor([0])

        fl_gamma0 = FocalLoss(gamma=0.0, reduction="none")
        fl_gamma2 = FocalLoss(gamma=2.0, reduction="none")

        loss_g0 = fl_gamma0(logits, labels).item()
        loss_g2 = fl_gamma2(logits, labels).item()

        # γ=2 should dramatically reduce loss for this easy example.
        assert loss_g2 < loss_g0, f"γ=2 loss ({loss_g2}) should be < γ=0 loss ({loss_g0})"
        # With p_t ≈ 1.0, the ratio should be very small.
        assert loss_g2 < 0.01 * loss_g0

    def test_hard_example_loss_barely_changed(self) -> None:
        """A hard example (low p_t) should have similar loss with γ=0 and γ=2."""
        # Create a logit where model is very wrong.
        logits = torch.tensor([[-5.0, 10.0, 10.0]])
        labels = torch.tensor([0])

        fl_gamma0 = FocalLoss(gamma=0.0, reduction="none")
        fl_gamma2 = FocalLoss(gamma=2.0, reduction="none")

        loss_g0 = fl_gamma0(logits, labels).item()
        loss_g2 = fl_gamma2(logits, labels).item()

        # For hard examples (p_t ≈ 0), (1-p_t)^γ ≈ 1, so losses are similar.
        # The modulating factor for p_t ≤ 0.5 is at most 4× with γ=2 (paper §3.2).
        ratio = loss_g2 / loss_g0
        assert ratio > 0.25, f"Hard example ratio ({ratio}) should be > 0.25"

    def test_increasing_gamma_increases_focus(self) -> None:
        """Higher γ should produce larger loss reduction for easy examples."""
        logits = torch.tensor([[5.0, -2.0, -2.0]])
        labels = torch.tensor([0])

        losses = []
        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
            fl = FocalLoss(gamma=gamma, reduction="none")
            losses.append(fl(logits, labels).item())

        # Losses should be monotonically decreasing as γ increases.
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1], (
                f"Loss at γ={[0, 0.5, 1, 2, 5][i]} ({losses[i]}) should be >= "
                f"loss at γ={[0, 0.5, 1, 2, 5][i+1]} ({losses[i+1]})"
            )


# ---------------------------------------------------------------------------
# α-vector shape and behavior
# ---------------------------------------------------------------------------


class TestAlphaWeighting:
    """α per-class weights must have correct shape and effect."""

    def test_alpha_shape_mismatch_raises(self) -> None:
        """α length must match number of classes in logits at forward time."""
        alpha = torch.ones(5)
        fl = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(4, 10)  # 10 classes, α has 5
        labels = torch.randint(0, 10, (4,))
        with pytest.raises(IndexError):
            fl(logits, labels)

    def test_alpha_scales_per_class(self) -> None:
        """Higher α for a class should produce higher loss for that class."""
        C = 4
        logits = torch.zeros(2, C)  # uniform logits → p_t = 1/C for all
        labels = torch.tensor([0, 1])

        # Class 0 has high weight, class 1 has low weight.
        alpha = torch.tensor([10.0, 0.1, 1.0, 1.0])
        fl = FocalLoss(gamma=2.0, alpha=alpha, reduction="none")
        losses = fl(logits, labels)

        assert losses[0] > losses[1], (
            f"Class 0 (α=10) loss ({losses[0]}) should be > class 1 (α=0.1) loss ({losses[1]})"
        )

    def test_alpha_23_classes(self) -> None:
        """α vector with 23 elements (HyperKvasir class count) works correctly."""
        C = 23
        alpha = torch.rand(C) + 0.01  # positive weights
        fl = FocalLoss(gamma=2.0, alpha=alpha)

        logits = torch.randn(16, C)
        labels = torch.randint(0, C, (16,))
        loss = fl(logits, labels)

        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """No NaN or Inf for extreme logit values."""

    def test_very_confident_correct(self) -> None:
        """Very high logit for correct class → loss ≈ 0, no NaN."""
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        labels = torch.tensor([0])

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, labels)

        assert not torch.isnan(loss), "NaN for very confident correct prediction"
        assert not torch.isinf(loss), "Inf for very confident correct prediction"
        assert loss.item() >= 0.0

    def test_very_confident_wrong(self) -> None:
        """Very high logit for wrong class → high loss, no NaN."""
        logits = torch.tensor([[-100.0, 100.0, -100.0]])
        labels = torch.tensor([0])

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, labels)

        assert not torch.isnan(loss), "NaN for very confident wrong prediction"
        assert not torch.isinf(loss), "Inf for very confident wrong prediction"
        assert loss.item() > 0.0

    def test_zero_logits(self) -> None:
        """All-zero logits → uniform softmax, no NaN."""
        logits = torch.zeros(8, 5)
        labels = torch.randint(0, 5, (8,))

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, labels)

        assert not torch.isnan(loss)
        assert loss.item() > 0.0

    def test_large_batch_no_nan(self) -> None:
        """Large batch with random logits produces no NaN."""
        torch.manual_seed(99)
        logits = torch.randn(512, 23) * 10  # wide range
        labels = torch.randint(0, 23, (512,))

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_stability_with_alpha(self) -> None:
        """Extreme logits + α weighting → no NaN."""
        C = 23
        alpha = torch.rand(C) + 0.01
        fl = FocalLoss(gamma=2.0, alpha=alpha)

        logits = torch.tensor([[200.0] + [-200.0] * (C - 1)])
        labels = torch.tensor([0])
        loss = fl(logits, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients must flow through FocalLoss for training."""

    def test_gradient_flows(self) -> None:
        """Logits with requires_grad should have non-None gradient after backward."""
        logits = torch.randn(8, 10, requires_grad=True)
        labels = torch.randint(0, 10, (8,))

        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, labels)
        loss.backward()

        assert logits.grad is not None, "Gradient did not flow through FocalLoss"
        assert not torch.all(logits.grad == 0), "Gradient is all zeros"

    def test_gradient_flows_with_alpha(self) -> None:
        """Gradient flow works when α is provided."""
        C = 5
        logits = torch.randn(8, C, requires_grad=True)
        labels = torch.randint(0, C, (8,))
        alpha = torch.ones(C)

        fl = FocalLoss(gamma=2.0, alpha=alpha)
        loss = fl(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_gradient_with_gamma_zero(self) -> None:
        """Gradient at γ=0 matches CE gradient."""
        torch.manual_seed(42)
        C = 5

        logits_focal = torch.randn(8, C, requires_grad=True)
        logits_ce = logits_focal.detach().clone().requires_grad_(True)
        labels = torch.randint(0, C, (8,))

        fl = FocalLoss(gamma=0.0)
        loss_focal = fl(logits_focal, labels)
        loss_focal.backward()

        loss_ce = F.cross_entropy(logits_ce, labels)
        loss_ce.backward()

        torch.testing.assert_close(
            logits_focal.grad, logits_ce.grad, atol=1e-5, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# Reduction modes
# ---------------------------------------------------------------------------


class TestReductionModes:
    """Test mean, sum, and none reduction modes."""

    def test_reduction_none_shape(self) -> None:
        fl = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))
        loss = fl(logits, labels)
        assert loss.shape == (16,)

    def test_reduction_sum(self) -> None:
        fl_none = FocalLoss(gamma=2.0, reduction="none")
        fl_sum = FocalLoss(gamma=2.0, reduction="sum")

        torch.manual_seed(0)
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))

        loss_none = fl_none(logits, labels)
        loss_sum = fl_sum(logits, labels)

        torch.testing.assert_close(loss_none.sum(), loss_sum, atol=1e-6, rtol=1e-5)

    def test_reduction_mean(self) -> None:
        fl_none = FocalLoss(gamma=2.0, reduction="none")
        fl_mean = FocalLoss(gamma=2.0, reduction="mean")

        torch.manual_seed(0)
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))

        loss_none = fl_none(logits, labels)
        loss_mean = fl_mean(logits, labels)

        torch.testing.assert_close(loss_none.mean(), loss_mean, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# build_loss integration
# ---------------------------------------------------------------------------


class TestBuildLoss:
    """build_loss correctly creates FocalLoss from config dicts."""

    def test_build_focal(self) -> None:
        loss = build_loss({"type": "focal", "gamma": 2.0})
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 2.0
        assert loss.alpha is None

    def test_build_focal_default_gamma(self) -> None:
        loss = build_loss({"type": "focal"})
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 2.0  # default

    def test_build_focal_balanced_no_alpha(self) -> None:
        loss = build_loss({"type": "focal_balanced", "gamma": 1.0})
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 1.0
        assert loss.alpha is None  # alpha not provided, caller must set it

    def test_build_focal_balanced_with_alpha(self) -> None:
        loss = build_loss({
            "type": "focal_balanced",
            "gamma": 2.0,
            "alpha": [0.1, 0.2, 0.3, 0.4],
        })
        assert isinstance(loss, FocalLoss)
        assert loss.alpha is not None
        assert loss.alpha.shape == (4,)

    def test_build_existing_types_unchanged(self) -> None:
        """Existing loss types still work (no regressions)."""
        import torch.nn as nn

        ce = build_loss({"type": "cross_entropy"})
        assert isinstance(ce, nn.CrossEntropyLoss)

        ce2 = build_loss({"type": "ce"})
        assert isinstance(ce2, nn.CrossEntropyLoss)

        smooth = build_loss({"type": "ce_label_smooth", "label_smoothing": 0.05})
        assert isinstance(smooth, nn.CrossEntropyLoss)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss type"):
            build_loss({"type": "nonexistent"})


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """FocalLoss constructor validates inputs."""

    def test_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            FocalLoss(gamma=-1.0)

    def test_gamma_zero_accepted(self) -> None:
        fl = FocalLoss(gamma=0.0)
        assert fl.gamma == 0.0

    def test_alpha_registered_as_buffer(self) -> None:
        alpha = torch.tensor([1.0, 2.0, 3.0])
        fl = FocalLoss(gamma=2.0, alpha=alpha)
        # Buffers move with .to() but are not parameters.
        assert "alpha" in dict(fl.named_buffers())
        assert "alpha" not in dict(fl.named_parameters())
