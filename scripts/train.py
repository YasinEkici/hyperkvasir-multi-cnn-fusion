"""Train one configured experiment from the experiment matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data.feature_cache import cache_frozen_features
from src.data.manifests import read_manifest_csv
from src.models.classifiers import MLPClassifier
from src.models.projections import BranchProjection
from src.training.losses import build_loss
from src.training.optimizers import build_optimizer
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer
from src.utils.checkpointing import load_checkpoint
from src.utils.logging import get_logger, save_metrics
from src.utils.paths import feature_cache_dir, project_root, results_dir
from src.utils.reproducibility import seed_all

logger = get_logger(__name__)

BACKBONE_DIMS: dict[str, int] = {
    "resnet50": 2048,
    "mobilenetv2": 1280,
    "efficientnetb0": 1280,
}


# ---------------------------------------------------------------------------
# Lightweight frozen head model
# ---------------------------------------------------------------------------

class FrozenHeadModel(nn.Module):
    """Projection + optional fusion + MLP classifier over cached features."""

    def __init__(
        self,
        backbone_names: list[str],
        projection_dim: int,
        fusion_type: str,
        num_classes: int,
        mlp_hidden: list[int],
        dropout: float,
    ):
        super().__init__()
        self.backbone_names = backbone_names
        self.fusion_type = fusion_type
        self.backbone_dims = [BACKBONE_DIMS[n] for n in backbone_names]

        self.projections = nn.ModuleDict(
            {
                name: BranchProjection(in_dim=dim, out_dim=projection_dim)
                for name, dim in zip(backbone_names, self.backbone_dims)
            }
        )

        num_branches = len(backbone_names)
        if num_branches == 1 or fusion_type == "none":
            self.fusion = None
            fusion_out_dim = projection_dim
        elif fusion_type == "concat":
            from src.models.fusion.concat import FusionModule
            self.fusion = FusionModule(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "weighted":
            from src.models.fusion.weighted import FusionModule
            self.fusion = FusionModule(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = MLPClassifier(
            input_dim=fusion_out_dim,
            num_classes=num_classes,
            hidden_dims=mlp_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, sum(backbone_dims)]
        splits = torch.split(x, self.backbone_dims, dim=1)
        projected = [self.projections[name](feat) for name, feat in zip(self.backbone_names, splits)]

        if self.fusion is None:
            fused = projected[0]
        else:
            fused = self.fusion(projected)

        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def _load_split_features(
    backbone_names: list[str],
    fold: int,
    cache_dir: Path,
    fold_manifest: Path,
    dataset_config: dict,
    batch_size: int,
    device: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return train/val/test (features, labels) tensors, building cache if needed."""
    # Read fold manifest to get per-split row indices
    rows = read_manifest_csv(fold_manifest)
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, row in enumerate(rows):
        split = str(row.get("split", ""))
        if split in split_indices:
            split_indices[split].append(i)

    # Validate all expected cache files exist; build missing ones
    missing_backbones = [
        name
        for name in backbone_names
        if not (cache_dir / f"{fold_manifest.stem}_{name}_features.pt").exists()
    ]
    if missing_backbones:
        logger.info("Building feature cache for backbones: %s", missing_backbones)
        mean = dataset_config.get("normalize_mean", dataset_config.get("mean", [0.485, 0.456, 0.406]))
        std = dataset_config.get("normalize_std", dataset_config.get("std", [0.229, 0.224, 0.225]))
        cache_config = {
            "mean": mean,
            "std": std,
            "image_size": dataset_config.get("image_size", 224),
        }
        cache_frozen_features(
            backbones=missing_backbones,
            dataset_config=cache_config,
            split_manifest=fold_manifest,
            output_dir=cache_dir,
            batch_size=batch_size,
            device=device,
        )

    # Load and concatenate features across backbones for each split
    per_backbone: list[torch.Tensor] = []
    labels_tensor: torch.Tensor | None = None

    for name in backbone_names:
        cache_path = cache_dir / f"{fold_manifest.stem}_{name}_features.pt"
        cache = torch.load(cache_path, weights_only=True)
        per_backbone.append(cache["features"])
        if labels_tensor is None:
            labels_tensor = cache["labels"]

    assert labels_tensor is not None
    all_features = torch.cat(per_backbone, dim=1)  # [N, sum(dims)]

    result: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for split_name, idxs in split_indices.items():
        if not idxs:
            continue
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        result[split_name] = (all_features[idx_tensor], labels_tensor[idx_tensor])

    return result


def _make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    weighted_sampler: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    sampler = None
    if weighted_sampler:
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train one experiment from the matrix.")
    parser.add_argument("--config", required=True, help="Path to experiment_matrix.yaml")
    parser.add_argument("--experiment", required=True, help="Experiment ID to run")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    root = project_root()

    # Load experiment matrix
    with open(args.config, "r") as f:
        matrix = yaml.safe_load(f)

    experiments = {exp["id"]: exp for exp in matrix["experiments"]}
    if args.experiment not in experiments:
        raise SystemExit(f"Experiment '{args.experiment}' not found in {args.config}")
    exp = experiments[args.experiment]

    # Load sub-configs
    with open(root / exp["dataset"], "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(root / exp["method"], "r") as f:
        method_cfg = yaml.safe_load(f)
    with open(root / exp["training"], "r") as f:
        training_cfg = yaml.safe_load(f)

    fold: int = int(exp.get("fold", 0))
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    seed_all(
        int(training_cfg.get("reproducibility", {}).get("seed", 42)),
        deterministic=training_cfg.get("reproducibility", {}).get("deterministic", True),
        cudnn_benchmark=training_cfg.get("reproducibility", {}).get("cudnn_benchmark", False),
    )
    logger.info("Device: %s | Experiment: %s | Fold: %d", device, args.experiment, fold)

    # Resolve paths
    split_protocol = dataset_cfg.get("split_protocol", "hyperkvasir_official_5fold")
    fold_manifest = root / "data" / "splits" / split_protocol / f"fold_{fold}.csv"
    if not fold_manifest.exists():
        raise FileNotFoundError(f"Fold manifest not found: {fold_manifest}")

    cache_dir = feature_cache_dir()
    run_dir = results_dir(args.experiment)

    # Model config
    backbone_names: list[str] = method_cfg["backbone_names"]
    projection_dim: int = int(method_cfg.get("projection_dim", 512))
    fusion_type: str = method_cfg.get("fusion_type", "none")
    mlp_hidden: list[int] = method_cfg.get("mlp_hidden", [256])
    dropout: float = float(method_cfg.get("dropout", 0.3))
    num_classes: int = int(dataset_cfg["num_classes"])

    # Training config
    epochs: int = int(training_cfg.get("epochs", 60))
    batch_size: int = int(training_cfg.get("batch_size", 64))
    mixed_precision: bool = bool(training_cfg.get("mixed_precision", False))
    early_stopping_patience: int = int(training_cfg.get("early_stopping", {}).get("patience", 8))
    class_imbalance: str = dataset_cfg.get("class_imbalance_handling", "none")

    # Load / build feature cache
    logger.info("Loading features for fold %d from %s", fold, cache_dir)
    split_data = _load_split_features(
        backbone_names=backbone_names,
        fold=fold,
        cache_dir=cache_dir,
        fold_manifest=fold_manifest,
        dataset_config=dataset_cfg,
        batch_size=batch_size,
        device=device,
    )

    train_features, train_labels = split_data["train"]
    val_features, val_labels = split_data["val"]
    test_features, test_labels = split_data["test"]

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_labels), len(val_labels), len(test_labels),
    )

    use_sampler = class_imbalance == "weighted_sampler"
    train_loader = _make_loader(train_features, train_labels, batch_size, weighted_sampler=use_sampler)
    val_loader = _make_loader(val_features, val_labels, batch_size)
    test_loader = _make_loader(test_features, test_labels, batch_size)

    # Build lightweight frozen head model
    model = FrozenHeadModel(
        backbone_names=backbone_names,
        projection_dim=projection_dim,
        fusion_type=fusion_type,
        num_classes=num_classes,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    )

    optimizer = build_optimizer(model, training_cfg.get("optimizer", {}))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        training_cfg.get("scheduler", {}),
        steps_per_epoch=steps_per_epoch,
        total_epochs=epochs,
    )
    criterion = build_loss(training_cfg.get("loss", {}))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        run_dir=run_dir,
        device=device,
        num_classes=num_classes,
        early_stopping_patience=early_stopping_patience,
        mixed_precision=mixed_precision,
    )

    # Train
    logger.info("Starting training — %d epochs, patience %d", epochs, early_stopping_patience)
    history = trainer.fit(train_loader, val_loader, epochs)

    # Evaluate on test set using best checkpoint
    load_checkpoint(model, run_dir / "best.pt")
    model.to(device)
    test_metrics = trainer.evaluate(test_loader)
    preds, labels_arr = trainer.predict(test_loader)

    logger.info(
        "Test — accuracy: %.4f  macro_f1: %.4f  zero_support_classes: %s",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["zero_support_classes"],
    )

    # Save outputs
    final_metrics = {
        "experiment_id": args.experiment,
        "fold": fold,
        "device": device,
        "best_val_macro_f1": trainer.best_val_f1,
        "test": test_metrics,
        "history": history,
    }
    save_metrics(run_dir, final_metrics)

    np.savez(
        run_dir / "predictions.npz",
        preds=preds,
        labels=labels_arr,
    )

    # Save a copy of the resolved config for reproducibility
    combined_cfg = {
        "experiment": exp,
        "dataset": dataset_cfg,
        "method": method_cfg,
        "training": training_cfg,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(combined_cfg, f, allow_unicode=True)

    logger.info("Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
