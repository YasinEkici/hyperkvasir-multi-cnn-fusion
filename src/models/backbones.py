"""Torchvision backbone feature extractors."""

import torch
from torch import Tensor, nn


class BackboneFeatureExtractor(nn.Module):
    """Wraps a torchvision backbone and exposes penultimate features only."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        unfreeze_blocks: int = 0,
    ):
        super().__init__()
        self.name = name.lower()
        self.pretrained = pretrained
        self.unfreeze_blocks = unfreeze_blocks

        if self.name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
        elif self.name == "mobilenetv2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            model = mobilenet_v2(weights=weights)
            self.backbone = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.name == "efficientnetb0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights)
            self.backbone = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported backbone: {self.name}")

        self._freeze_parameters()

    def _freeze_parameters(self):
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_blocks <= 0:
            return

        if self.name == "resnet50":
            start_idx = max(4, 8 - self.unfreeze_blocks)
            for module in self.backbone[start_idx:]:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.name == "mobilenetv2":
            start_idx = max(1, 19 - 2 * self.unfreeze_blocks)
            for module in self.backbone[start_idx:]:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.name == "efficientnetb0":
            start_idx = max(1, 9 - self.unfreeze_blocks)
            for module in self.backbone[start_idx:]:
                for param in module.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Check if weight is defined and frozen
                    if getattr(module, "weight", None) is not None and not module.weight.requires_grad:
                        module.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Input: (B, 3, 224, 224). Output: (B, D) penultimate features."""
        x = self.backbone(x)
        if self.name != "resnet50":
            x = self.pool(x)
        return torch.flatten(x, 1)

    @property
    def feature_dim(self) -> int:
        """Return the documented penultimate feature dimension."""
        dims = {"resnet50": 2048, "mobilenetv2": 1280, "efficientnetb0": 1280}
        if self.name not in dims:
            raise ValueError(f"Unsupported backbone: {self.name}")
        return dims[self.name]

    def trainable_param_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        llrd_decay: float = 0.75,
    ) -> list[dict]:
        """Return optimizer parameter groups for fine-tuning."""
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return []
        return [{"params": params, "lr": backbone_lr}]
