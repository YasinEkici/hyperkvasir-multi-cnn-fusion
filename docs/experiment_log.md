# Experiment Log

Record successful runs and environment notes here.

## 2026-05-22 - Week 1 Steps 1-3 Environment Setup

- Ran `uv sync`; environment resolved and checked successfully.
- Initial verification showed CPU-only torch/torchvision builds.
- Updated project dependency configuration so `torch` and `torchvision` resolve
  from the PyTorch CUDA 13.2 wheel index for RTX 5080 support.
- Re-ran `uv lock`, `uv sync`, and environment verification.
- Final verified versions:
  - Python: `3.11.9`
  - PyTorch: `2.12.0+cu132`
  - torchvision: `0.27.0+cu132`
  - CUDA runtime: `13.2`
  - CUDA available: `True`
  - GPU: `NVIDIA GeForce RTX 5080 Laptop GPU`
- Ran `uv run pytest tests/`; result: `6 passed`.

## 2026-05-23 - Week 1 Steps 4-6 Data Pipeline and Official Splits

- Updated the Week 1 split protocol to use the official HyperKvasir 5-fold
  split source under `data/splits/official/`.
- Ran `uv run python scripts/prepare_data.py --dataset hyperkvasir --config configs/dataset/hyperkvasir_23class_official.yaml`.
- Generated `data/splits/hyperkvasir_manifest.csv`.
- Verified HyperKvasir manifest sample count: `10662`.
- Verified class count: `23`.
- Ran `uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_official.yaml`.
- Generated official fold manifests:
  - `data/splits/hyperkvasir_official_5fold/fold_0.csv`
  - `data/splits/hyperkvasir_official_5fold/fold_1.csv`
  - `data/splits/hyperkvasir_official_5fold/fold_2.csv`
  - `data/splits/hyperkvasir_official_5fold/fold_3.csv`
  - `data/splits/hyperkvasir_official_5fold/fold_4.csv`
- Verified each fold manifest has train/validation/test partitions with all
  `23` classes represented.
- Ran `uv run pytest tests/`; result: `9 passed`.

## 2026-05-23 - Week 1 Steps 10-11: Runtime Components and First Experiment

### Runtime components implemented

- `src/utils/paths.py` — `project_root()`, `results_dir(exp_id)`, `feature_cache_dir()`, `splits_dir()`
- `src/utils/logging.py` — `get_logger()`, `log_epoch()`, `save_metrics()`
- `src/utils/checkpointing.py` — `save_checkpoint()`, `load_checkpoint()`
- `src/evaluation/metrics.py` — `compute_metrics()` using scikit-learn; `zero_division=0` throughout
- `src/training/losses.py` — `build_loss()` supporting `cross_entropy` and `ce_label_smooth`
- `src/training/optimizers.py` — `build_optimizer()` (AdamW head-only) + `build_adamw_with_llrd()` skeleton for Week 2
- `src/training/schedulers.py` — `build_scheduler()` with cosine + linear warmup
- `src/training/trainer.py` — `Trainer` class: `fit()`, `_train_epoch()`, `evaluate()`, `predict()`; early stopping on `val_macro_f1`; mixed precision via `torch.amp`
- `scripts/train.py` — full CLI; auto-builds feature cache if missing; `FrozenHeadModel` (BranchProjection + fusion + MLPClassifier over cached features); WeightedRandomSampler; saves `config.yaml`, `metrics.json`, `predictions.npz`, `best.pt`, `last.pt`
- `configs/experiment_matrix.yaml` — added `01_single_resnet50_frozen_official` entry

### Experiment: `01_single_resnet50_frozen_official` (fold 0)

- Command: `uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 01_single_resnet50_frozen_official`
- Backbone: ResNet50 (frozen, pretrained ImageNet) → 2048-dim features cached to `results/feature_cache/fold_0_resnet50_features.pt`
- Head: BranchProjection(2048→512) + MLPClassifier([256], 23 classes, dropout=0.3)
- Optimizer: AdamW lr=1e-3, wd=1e-4
- Scheduler: cosine with 5-epoch warmup
- Loss: CrossEntropy + label_smoothing=0.1
- WeightedRandomSampler: enabled (class_imbalance_handling=weighted_sampler)
- Early stopping at epoch 16 (patience=8)
- **Results (test set, fold 0):**
  - Accuracy: 0.8402
  - Macro F1: 0.5588
  - Macro Precision: 0.5675
  - Macro Recall: 0.5638
  - Zero-support classes: none
  - No NaN values in any metric
- Outputs saved to `results/runs/01_single_resnet50_frozen_official/`

### Test suite

- Ran `uv run pytest tests/ -v`; result: `22 passed`.
