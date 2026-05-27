"""End-to-end multi-CNN fusion classifier."""

from torch import Tensor, nn

from src.models.backbones import BackboneFeatureExtractor
from src.models.projections import BranchProjection
from src.models.classifiers import MLPClassifier


class MultiCNNFusionClassifier(nn.Module):
    """End-to-end model: backbones -> projections -> fusion -> MLP head."""

    def __init__(
        self,
        backbone_names: list[str],
        unfreeze_blocks: int,
        projection_dim: int,
        fusion_type: str,
        num_classes: int,
        mlp_hidden: list[int] = [256],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_names = backbone_names
        self.unfreeze_blocks = unfreeze_blocks
        self.projection_dim = projection_dim
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout

        # Initialize backbones
        self.backbones = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        
        for name in backbone_names:
            bb = BackboneFeatureExtractor(name=name, pretrained=True, unfreeze_blocks=unfreeze_blocks)
            self.backbones[name] = bb
            self.projections[name] = BranchProjection(in_dim=bb.feature_dim, out_dim=projection_dim)

        num_branches = len(backbone_names)
        
        # Initialize fusion module
        if num_branches == 1 or fusion_type == "none":
            self.fusion = nn.Identity()
            fusion_out_dim = projection_dim
        elif fusion_type == "concat":
            from src.models.fusion.concat import FusionModule as ConcatFusion
            self.fusion = ConcatFusion(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "weighted":
            from src.models.fusion.weighted import FusionModule as WeightedFusion
            self.fusion = WeightedFusion(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "gmu":
            from src.models.fusion.gmu import FusionModule as GMUFusion
            self.fusion = GMUFusion(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Initialize MLP head
        self.classifier = MLPClassifier(
            input_dim=fusion_out_dim,
            num_classes=num_classes,
            hidden_dims=mlp_hidden,
            dropout=dropout
        )

    def extract_features(self, x: Tensor) -> dict[str, Tensor]:
        """Extract raw features from all backbones."""
        features = {}
        for name, backbone in self.backbones.items():
            features[name] = backbone(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        # Extract features and project
        projected = []
        for name in self.backbone_names:
            raw_feat = self.backbones[name](x)
            proj_feat = self.projections[name](raw_feat)
            projected.append(proj_feat)
        
        # Fuse
        if len(self.backbone_names) == 1 or self.fusion_type == "none":
            fused = projected[0]
        else:
            fused = self.fusion(projected)
            
        # Classify
        return self.classifier(fused)
