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

## 2026-05-24 — Week 2 Steps 4–5: Augmentation and EMA

### Step 4 — Augmentation (src/data/augmentation.py)

- `apply_rand_augment(img, N, M)`: wraps `torchvision.transforms.RandAugment(num_ops=N, magnitude=M)`.
- `apply_cutmix(images, labels, alpha)`: samples λ ~ Beta(α,α), clips cut box to image boundary, recomputes `lam` from actual clipped area (not raw Beta sample). Returns `(mixed, label_a, label_b, lam)`.
- `tests/test_augmentation.py` (new): 18 tests covering both functions including edge cases (N=0, B=1, parametrised alpha). All pass.
- Full suite: 50 passed.

### Step 5 — EMA (src/training/ema.py)

- `EMA.__init__(model, decay, start_epoch)`: clones all named parameters into shadow dict; validates decay in [0, 1).
- `EMA.update(model, epoch)`: no-op when `epoch < start_epoch`; otherwise `shadow = decay * shadow + (1 - decay) * param`.
- `EMA.apply(model)`: stores live-weight backup (clones), then copies shadow weights into model. Raises if called twice without restore.
- `EMA.restore(model)`: copies backup back into model; clears backup. Raises if no prior apply.
- Shadow uses `named_parameters()` — BN buffers (running_mean, running_var) are excluded, consistent with frozen-BN policy.
- `tests/test_ema.py` (new): 15 tests covering init, formula correctness, convergence, apply/restore byte-identity, double-apply guard, restore-without-apply guard, start_epoch boundary. All pass.
- Full suite: 65 passed.

## 2026-05-24 — Week 2 Step 6: LLRD optimizer

### Step 6 — build_adamw_with_llrd (src/training/optimizers.py)

- Replaced the flat two-group skeleton with per-backbone, per-block parameter groups.
- Block indices verified against project_structure.md §2.1–2.2:
  - ResNet50: depth 0=layer4[7], 1=layer3[6], 2=layer2[5], 3=layer1[4], 4=stem[0-3]
  - MobileNetV2: depth 0=features[15:19], 1=features[13:15], 2=features[1:13], 3=features[0]
  - EfficientNetB0: depth 0=features[6:9], 1=features[4:6], 2=features[1:4], 3=features[0]
- LR per block: `backbone_lr * (llrd_decay ** depth)`, depth=0 adjacent to head.
- Head group (projections, fusion, classifier) identified by excluding backbone param IDs.
- Empty groups (frozen blocks with no requires_grad params) are omitted.
- `tests/test_llrd.py` (new): 26 tests covering no-double-count, full-coverage,
  no-empty-groups, group-count-by-unfreeze-blocks, deepest-block-LR=backbone_lr,
  monotone LR decay toward stem, head-LR correctness, triple-backbone no overlap.
- Full suite: 91 passed.

## 2026-05-24 — Week 2 Step 7: Fine-tune training path

### Step 7 — Fine-tune code path (scripts/train.py + src/training/trainer.py)

- `train.py`: branches in `main()` on `unfreeze_blocks > 0`.
  - Fine-tune path: `_make_image_loaders` builds train/val/test DataLoaders from
    fold manifest rows. Train: Resize(256)→RandomCrop(224)→RandomHorizontalFlip→
    RandAugment→ToTensor→Normalize. Val/test: Resize(256)→CenterCrop(224)→ToTensor→
    Normalize. WeightedRandomSampler enabled for train when config says so.
  - `_PairDataset` wrapper strips the row-dict from HyperKvasirImageDataset (which
    returns 3-tuples) so Trainer receives the expected 2-tuples.
  - Optimizer: `build_adamw_with_llrd` when `optimizer.backbone_lr` is set in config.
  - EMA: instantiated when `ema.enabled = true`; shadow weights gated on start_epoch.
  - CutMix: passed as `cutmix_prob` / `cutmix_alpha` to Trainer.
  - Frozen path (unfreeze_blocks=0) completely unchanged.
- `trainer.py`: added optional `ema`, `cutmix_alpha`, `cutmix_prob` params (all
  default to None/0 → frozen path behavior unchanged).
  - `_compute_loss()`: applies CutMix when triggered, else standard CE.
  - `fit()`: EMA update after train epoch; apply before eval, restore after; save
    `ema.pt` (shadow dict) when new best found. EMA NOT applied before start_epoch.
    `best.pt` = EMA weights; `last.pt` = live weights.
- `tests/test_finetune_trainer.py` (new): 13 smoke tests. All pass.
- Full suite: 104 passed.

## 2026-05-24–25 — Week 2 Step 9: Fine-tune experiments + EMA bug fixes

### EMA bugs found and fixed

Two bugs were discovered during first execution of experiment 10:

**Bug 1 — Device mismatch in `EMA.update()`:**
- Shadow tensors cloned on CPU in `EMA.__init__()`. `Trainer.fit()` calls `model.to(device)`,
  moving model to CUDA. The in-place `mul_().add_()` in `update()` then failed:
  `RuntimeError: Expected all tensors to be on the same device (cuda:0 and cpu)`.
- Fix: lazy device migration in `update()` — if shadow device ≠ param device, move shadow
  in-place before the formula.

**Bug 2 — Stale random-init shadow on first EMA apply (warm-start missing):**
- `EMA.__init__()` clones model params at construction time (before training). With
  `start_epoch=5` and `decay=0.999`, after 1 formula update at epoch 6:
  shadow ≈ 0.999 × (random init) + 0.001 × (trained) — still ~99.9% random init.
  Applying this shadow collapsed val_f1 from 0.61 → 0.015, val_acc → 0.048 (≈ 1/23 random).
  Early stopping then triggered 8 epochs later, discarding a valid partially-trained model.
- Fix: `EMA.reset_shadow(model)` added; called once in `Trainer.fit()` at the epoch EMA
  first becomes active, before `update()`. This sets shadow = current trained weights so
  the formula runs from a meaningful starting point.

### Experiments (test set, fold 0, official 5-fold split)

| ID | Method | Fusion | Mode | Accuracy | Macro F1 | Stop ep |
|---|---|---|---|---:|---:|---:|
| `10_triple_concat_finetune_wide_official` | R+M+E | concat | finetune-3blk | **0.8784** | **0.5796** | 14 |
| `11_triple_weighted_finetune_wide_official` | R+M+E | weighted | finetune-3blk | 0.8685 | 0.5751 | 12 |
| `12_single_resnet50_finetune_wide_official` | R | — | finetune-3blk | 0.8501 | 0.5748 | 12 |

Zero-support classes: none in any run. No NaN values in any run.

### Observations

- Fine-tune triple concat (10) is the new best overall (acc=0.8784, F1=0.5796), beating
  the best frozen config (07 M+E: 0.8728, 0.5758).
- Concat still beats weighted in the fine-tune setting (0.5796 vs. 0.5751). The expected
  advantage of learnable branch weights did not appear on fold 0; 5-fold CV will clarify.
- Single ResNet50 fine-tune (12): large F1 gain over frozen single (0.5748 vs. 0.5588).
  End-to-end gradients recover discriminative features that frozen extraction misses.
- All three runs early-stopped at 12–14 epochs. Short convergence consistent with
  small dataset + strong augmentation (CutMix + RandAugment).

### Post-experiment scripts

- `uv run python scripts/generate_report_tables.py` → 12 rows written to
  `results/tables/ablation_table.csv` and `ablation_table.md`.
- `uv run python scripts/plot_results.py --confusion-matrix` → confusion matrices for
  all 12 experiments; per-class F1 chart for best (exp 10, F1=0.5796) saved to
  `results/figures/per_class_f1_best.png`.
- `docs/results_progress.md` fine-tune section populated (Step 9 acceptance gate passed).
- Full suite: 104 passed.
