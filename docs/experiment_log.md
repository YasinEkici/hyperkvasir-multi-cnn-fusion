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

## 2026-05-24 — Week 2 Steps 1–3: Frozen Ablation Experiments (Stage 0 + Stage 1)

### Infrastructure changes

- `configs/experiment_matrix.yaml` — added experiments `02` through `09` (official split, fold 0).
- `tests/test_frozen_head.py` (new) — shape tests for `FrozenHeadModel` with single, pair, and triple backbone inputs for both `concat` and `weighted` fusion types.
- `uv run pytest tests/ -v`; result: `32 passed`.

### Feature caches built

- `fold_0_mobilenetv2_features.pt` — built during experiment 02 run (~90 s on RTX 5080).
- `fold_0_efficientnetb0_features.pt` — built during experiment 03 run (~90 s on RTX 5080, weights auto-downloaded from pytorch.org).

### Experiments (test set, fold 0, official 5-fold split)

| ID | Method | Fusion | Accuracy | Macro F1 | Notes |
|---|---|---|---:|---:|---|
| `01_single_resnet50_frozen_official` | ResNet50 | none | 0.8402 | 0.5588 | Week 1 baseline |
| `02_single_mobilenetv2_frozen_official` | MobileNetV2 | none | 0.8483 | 0.5502 | early stop ep 21 |
| `03_single_efficientnetb0_frozen_official` | EfficientNetB0 | none | 0.8591 | 0.5586 | early stop ep 16 |
| `04_triple_concat_frozen_official` | R+M+E | concat | 0.8567 | 0.5630 | Stage 0 smoke |
| `05_pair_r_m_concat_frozen_official` | R+M | concat | 0.8247 | 0.5438 | early stop ep 14 |
| `06_pair_r_e_concat_frozen_official` | R+E | concat | 0.8388 | 0.5464 | early stop ep 11 |
| `07_pair_m_e_concat_frozen_official` | M+E | concat | 0.8728 | 0.5758 | best frozen pair |
| `08_triple_concat_frozen_official` | R+M+E | concat | 0.8567 | 0.5630 | Stage 1 (= 04) |
| `09_triple_weighted_frozen_official` | R+M+E | weighted | 0.8549 | 0.5609 | early stop ep 12 |

Zero-support classes: none in any run. No NaN values in any run.

### Observations

- EfficientNetB0 is the best single backbone (acc=0.8591, F1=0.5586), beating ResNet50 and MobileNetV2.
- The best pair is M+E concat (acc=0.8728, F1=0.5758), suggesting ResNet50 features may add noise for pair fusion in the frozen setting.
- Triple concat (0.5630) does not improve over the best pair (0.5758) — adding ResNet50 features to M+E hurts in the frozen setting.
- Weighted fusion (0.5609) performs similarly to triple concat (0.5630); learnable weights do not yet give an advantage over concat in the frozen regime.
