"""Evaluate a trained experiment with optional Test-Time Augmentation (TTA).

Inference only — loads the frozen `best.pt` checkpoint and runs it forward on the
fold's held-out test set. No training. Reuses the same model builder
(`MultiCNNFusionClassifier`), checkpoint loader, dataset, and metrics as
`train.py`, so the base (no-TTA) view reproduces the predictions saved during
training.

TTA = average the softmax over a fixed, DETERMINISTIC set of test-time views
(identity, horizontal flip, and a second scale ± flip). Per-model, per-image —
no leakage, no model averaging across folds.

Run-dir convention (matches train.py --fold logic):
  fold 0 : results/runs/{exp}/
  fold k>=1: results/runs/{exp}_fold_{k}/

Writes per fold: predictions_tta.npz (preds, labels) and metrics_tta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.datasets import HyperKvasirImageDataset
from src.data.manifests import read_manifest_csv
from src.evaluation.metrics import compute_metrics
from src.models.full_model import MultiCNNFusionClassifier
from src.utils.checkpointing import load_checkpoint


# ---------------------------------------------------------------------------
# TTA views (deterministic) and softmax aggregation — pure, unit-testable
# ---------------------------------------------------------------------------

def make_views(
    tta: bool,
    image_size: int,
    resize_to: int,
    mean: list[float],
    std: list[float],
) -> list[tuple[str, transforms.Compose]]:
    """Return the ordered list of (name, transform) test-time views.

    The first view is always `base` — identical to train.py's val/test transform
    (Resize(resize_to) -> CenterCrop(image_size)), so a single-view run
    reproduces the base predictions. When tta=False only `base` is returned.

    All views are deterministic: RandomHorizontalFlip(p=1.0) always flips, and
    no random augmentation (RandAugment/CutMix) is used at test time.
    """
    norm = transforms.Normalize(mean, std)

    def _compose(resize: int, flip: bool) -> transforms.Compose:
        ops: list = [transforms.Resize(resize), transforms.CenterCrop(image_size)]
        if flip:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        ops += [transforms.ToTensor(), norm]
        return transforms.Compose(ops)

    base = ("base", _compose(resize_to, flip=False))
    if not tta:
        return [base]

    # 2 scales x 2 flips = 4 deterministic views. resize_to (e.g. 256) gives the
    # standard centre crop; image_size (e.g. 224) gives a slightly wider field of
    # view. Identity (base) is included so TTA never discards the original view.
    return [
        base,
        ("hflip", _compose(resize_to, flip=True)),
        ("scale", _compose(image_size, flip=False)),
        ("scale_hflip", _compose(image_size, flip=True)),
    ]


def aggregate_softmax(view_probs: list[np.ndarray]) -> np.ndarray:
    """Average a list of (N, C) softmax-probability arrays over the views.

    With a single view this returns that view's probabilities unchanged
    (identity == base). Inputs are assumed to be valid softmax rows; the mean of
    rows that each sum to 1 also sums to 1.
    """
    if not view_probs:
        raise ValueError("aggregate_softmax requires at least one view")
    stacked = np.stack(view_probs, axis=0)  # (V, N, C)
    return stacked.mean(axis=0)  # (N, C)


# ---------------------------------------------------------------------------
# Inference plumbing
# ---------------------------------------------------------------------------

class _ImageLabel(Dataset):
    """Strip the manifest-row dict so the loader yields (image, label) pairs."""

    def __init__(self, base: HyperKvasirImageDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, _ = self.base[index]
        return image, label


def _fold_run_dir(exp_id: str, fold: int) -> Path:
    if fold == 0:
        return _ROOT / "results" / "runs" / exp_id
    return _ROOT / "results" / "runs" / f"{exp_id}_fold_{fold}"


def _build_model_from_config(cfg: dict) -> MultiCNNFusionClassifier:
    method = cfg["method"]
    dataset = cfg["dataset"]
    training = cfg["training"]
    return MultiCNNFusionClassifier(
        backbone_names=method["backbone_names"],
        unfreeze_blocks=int(training.get("unfreeze_blocks", 3)),
        projection_dim=int(method.get("projection_dim", 512)),
        fusion_type=method.get("fusion_type", "none"),
        num_classes=int(dataset["num_classes"]),
        mlp_hidden=method.get("mlp_hidden", [256]),
        dropout=float(method.get("dropout", 0.3)),
    )


@torch.no_grad()
def _view_probabilities(
    model: MultiCNNFusionClassifier,
    test_rows: list[dict],
    transform: transforms.Compose,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the test set under one transform; return (probs, labels)."""
    base = HyperKvasirImageDataset(test_rows, transform=transform, project_root=_ROOT)
    loader = DataLoader(_ImageLabel(base), batch_size=batch_size, shuffle=False, num_workers=0)
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        probs_chunks.append(probs.cpu().numpy())
        labels_chunks.append(labels.numpy())
    return np.concatenate(probs_chunks), np.concatenate(labels_chunks)


def evaluate_fold(exp_id: str, fold: int, tta: bool, device: str) -> dict:
    run_dir = _fold_run_dir(exp_id, fold)
    config_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "best.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = yaml.safe_load(config_path.read_text())
    dataset_cfg = cfg["dataset"]
    mean = dataset_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    image_size = int(dataset_cfg.get("image_size", 224))
    resize_to = 256  # per project_structure.md §2.3 (matches train.py)
    split_protocol = dataset_cfg.get("split_protocol", "hyperkvasir_official_5fold")
    batch_size = int(cfg.get("training", {}).get("batch_size", 64))

    fold_manifest = _ROOT / "data" / "splits" / split_protocol / f"fold_{fold}.csv"
    if not fold_manifest.exists():
        raise FileNotFoundError(f"Fold manifest not found: {fold_manifest}")
    test_rows = [r for r in read_manifest_csv(fold_manifest) if str(r.get("split")) == "test"]
    if not test_rows:
        raise ValueError(f"No test rows in {fold_manifest}")

    model = _build_model_from_config(cfg)
    load_checkpoint(model, ckpt_path)
    model.to(device).eval()

    views = make_views(tta, image_size, resize_to, mean, std)
    view_probs: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for name, transform in views:
        probs, lbls = _view_probabilities(model, test_rows, transform, device, batch_size)
        view_probs.append(probs)
        if labels is None:
            labels = lbls
        elif not np.array_equal(labels, lbls):
            raise RuntimeError("Label order changed across views — shuffle leaked in.")

    mean_probs = aggregate_softmax(view_probs)
    preds = mean_probs.argmax(axis=1).astype(np.int64)
    assert labels is not None
    metrics = compute_metrics(preds, labels)

    suffix = "tta" if tta else "base"
    np.savez(
        run_dir / f"predictions_{suffix}.npz",
        preds=preds,
        labels=labels.astype(np.int64),
        probs=mean_probs.astype(np.float32),
    )
    out = {
        "experiment": exp_id,
        "fold": fold,
        "tta": tta,
        "views": [name for name, _ in views],
        "n_test": int(len(labels)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "zero_support_classes": metrics["zero_support_classes"],
    }
    (run_dir / f"metrics_{suffix}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | experiment: {args.experiment} | tta: {args.tta}")

    for fold in args.folds:
        out = evaluate_fold(args.experiment, fold, args.tta, device)
        print(
            f"  fold {fold} [{'+'.join(out['views'])}] "
            f"n={out['n_test']}  acc={out['accuracy']:.4f}  macro_f1={out['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
