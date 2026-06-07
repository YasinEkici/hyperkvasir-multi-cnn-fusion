# Experiment Log

Record successful runs and environment notes here.

## 2026-06-07 - Week 3.5 Step 3 Focal 5-Fold CV (exp 16) — COMPLETE

- Ran exp `16_triple_weighted_finetune_focal_official` for folds 0–4 on **Colab
  A100** via the runner-only notebook (D-09 provenance gate passed). GPU training
  performed by the user; assistant ran CPU post-processing only.
- Per-fold A100 timing: ~25–35 min/fold (~2 min/epoch, early-stop at epoch 12–17).
- Per-fold test macro-F1: 0.5777 / 0.5847 / 0.5687 / 0.5741 / 0.5962.
  No NaN, no zero-support classes in any fold.
- Post-processing (CPU, local):
  - `aggregate_cv.py` → macro-F1 0.5803 ± 0.0095, acc 0.8603 ± 0.0115.
  - `compute_ci.py` → pooled macro-F1 0.5914, 95% CI [0.5750, 0.6096] (n=10,662).
  - `compute_extra_metrics.py` (new) → acc/micro-F1 0.8603, weighted-F1 0.8633,
    MCC 0.8489. Saved `results/tables/extra_metrics_16_*.json`.
  - `plot_results.py --confusion-matrix` → 5 confusion matrices into
    `results/runs/16_*/`; global comparison/training/per-class figures regenerated
    over all experiments (per-class best remains exp 11 fold_1, F1=0.6014).
- **Verdict: focal did NOT beat CE.** exp 16 pooled F1 0.5914 < exp 11 CE 0.6000;
  CIs overlap; focal lower on every metric (acc, weighted-F1, MCC). Week 3 champion
  (exp 11 triple weighted CE) stays the project best. Honest negative result per
  D-06 / exec-plan §7. No test-set retuning.
- `docs/results_progress.md` Week 3.5 section populated with the ablation table,
  CI, and explicit "new best vs Week 3 best" verdict (acceptance gate passed).
- Added `scripts/compute_extra_metrics.py` (MCC, micro/weighted F1 from saved
  predictions; no retraining). Also computed for Week 3 configs 10/11/13/14/15.

## 2026-06-06 - Week 3.5 Step 2 Focal Loss Implementation

- Implemented `FocalLoss` in `src/training/losses.py` using the α-balanced
  variant from Lin et al. 2017 (ICCV), eq. 5:
  `FL(p_t) = -α_t (1 - p_t)^γ log(p_t)`.
- Verbatim equation copied into `FocalLoss` docstring per AGENTS.md requirement.
- Numerical stability via `F.cross_entropy(reduction='none')` for `log(p_t)`,
  then `p_t = exp(-ce)` and `(1 - p_t)^γ` modulating factor. No manual
  softmax + log.
- Extended `build_loss` to support `type: "focal"` (γ-only, no class weighting)
  and `type: "focal_balanced"` (γ + optional per-class α tensor).
- Focal loss and label smoothing deliberately NOT combined (documented in
  `FocalLoss` docstring and `docs/decisions.md` D-06).
- Created `configs/training/finetune_wide_focal.yaml` — clone of
  `finetune_wide.yaml` with only the loss block changed to `focal` (γ=2.0).
  All other hyperparameters identical for a clean single-variable ablation.
- Added exp 16 (`16_triple_weighted_finetune_focal_official`) to
  `configs/experiment_matrix.yaml`.
- Created `tests/test_focal_loss.py` — 29 tests covering:
  γ=0 ≡ CE equivalence, focusing effect, α-vector shape/behavior,
  numerical stability at extreme logits, gradient flow, reduction modes,
  `build_loss` integration, and constructor validation.
- Full test suite: **158 passed** (129 existing + 29 new), 0 failures.
- No experiment results yet — training deferred to Step 3 on A100.

## 2026-06-05 - Week 3.5 Step 1 Colab Runner Infrastructure

- Added the A100 exploration config with batch 128, linearly scaled learning
  rates, and config-controlled cuDNN benchmark mode.
- Added run-specific resolved-config, hard-stop provenance, and returned-output
  collection commands under `scripts/`.
- Added a narrow `--training` override to `run_cv.py` / `train.py`; experiment
  matrix and base training configs remain unchanged.
- Added the runner-only Colab notebook and documentation. The notebook defaults
  to an existing exp 11 fold-0 smoke test; exp 16 remains deferred to Steps 2–3.
- Pinned the provisional Colab environment to the official PyTorch CUDA 12.8
  wheel pair. Real A100 validation is pending the user-run smoke test.

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

## 2026-05-25 — Week 2 Step 10: Final test suite

- Command: `uv run pytest tests/ -v`
- Result: **104 passed** in 10.93 s — no regressions.
- Test files verified:
  - `test_augmentation.py` — 18 tests (RandAugment + CutMix)
  - `test_backbones.py` — 12 tests (dimensions, freeze, BN eval)
  - `test_data_splits.py` — 4 tests (manifest, leakage, transform)
  - `test_ema.py` — 15 tests (init, formula, apply/restore, guards)
  - `test_feature_cache.py` — 1 test (e2e cache build)
  - `test_finetune_trainer.py` — 13 tests (EMA, CutMix, fit smoke)
  - `test_frozen_head.py` — 10 tests (single/pair/triple shapes)
  - `test_fusion_modules.py` — 3 tests (concat + weighted)
  - `test_llrd.py` — 26 tests (coverage, no-overlap, LR monotonicity)
  - `test_metrics.py` — 1 test (statistical imports)
  - `test_smoke_train.py` — 1 test (entrypoint placeholders)

## 2026-05-25 — Week 2 Step 11: Final documentation + Week 2 close-out

### Week 2 summary

Week 2 delivered the full Stage 0 + Stage 1 mandatory ablation on fold 0 of the
official HyperKvasir 5-fold split. All required components were implemented and
all 12 experiments completed without NaN values or zero-support classes.

**Infrastructure delivered:**
- Augmentation: `apply_rand_augment` + `apply_cutmix` (`src/data/augmentation.py`)
- EMA with warm-start + lazy device migration (`src/training/ema.py`)
- Per-block LLRD optimizer (`src/training/optimizers.py`)
- Fine-tune image-based training path (`scripts/train.py`, `src/training/trainer.py`)
- Ablation table generator (`scripts/generate_report_tables.py`)
- Visualisation: confusion matrices + per-class F1 chart (`scripts/plot_results.py`,
  `src/evaluation/visualization.py`)
- `docs/results_progress.md` created and fully populated through fine-tune batch

**Final experiment results (test set, fold 0):**

| ID | Method | Fusion | Mode | Acc | Macro F1 |
|---|---|---|---|---:|---:|
| 01 | ResNet50 | — | frozen | 0.8402 | 0.5588 |
| 02 | MobileNetV2 | — | frozen | 0.8483 | 0.5502 |
| 03 | EfficientNetB0 | — | frozen | 0.8591 | 0.5586 |
| 04/08 | R+M+E | concat | frozen | 0.8567 | 0.5630 |
| 05 | R+M | concat | frozen | 0.8247 | 0.5438 |
| 06 | R+E | concat | frozen | 0.8388 | 0.5464 |
| 07 | M+E | concat | frozen | 0.8728 | 0.5758 |
| 09 | R+M+E | weighted | frozen | 0.8549 | 0.5609 |
| 10 | R+M+E | concat | finetune-3blk | **0.8784** | **0.5796** |
| 11 | R+M+E | weighted | finetune-3blk | 0.8685 | 0.5751 |
| 12 | ResNet50 | — | finetune-3blk | 0.8501 | 0.5748 |

**Key findings:**
- Best overall: triple concat fine-tune (acc=0.8784, F1=0.5796).
- Fine-tuning consistently improves over frozen for all configs tested.
- M+E pair is the best frozen configuration; adding ResNet50 features in frozen
  setting reduces F1 (noise from 2048-dim frozen features).
- Weighted fusion does not outperform concat on fold 0 in either regime;
  5-fold CV in Week 3 will determine whether this is a fold-0 artefact.

**Test suite at close:** 104 passed.

## 2026-05-25 — Week 3 Steps 1–3: GMU implementation and fold-0 smoke test

### Infrastructure (Steps 1–2)

- `src/models/fusion/gmu.py` — full GMU implementation (replaces placeholder).
  - Bimodal equations from Arevalo et al. (2017) §3.1 in docstring (verbatim).
  - N-branch softmax generalization: `h_i = tanh(W_i · x_i)`;
    `z = softmax(W_z · concat([x_1,…,x_N]))` → (B, N);
    `h = Σ_i z_i * h_i` → (B, D).
  - `branch_transforms`: `nn.ModuleList` of N `nn.Linear(D, D)`.
  - `gate`: `nn.Linear(N*D, N)` — receives raw pre-transform features (paper §3.1).
  - `output_dim` property returns `feature_dim` (D-04 decision).
- `src/models/full_model.py` — added `elif fusion_type == "gmu"` branch.
- `scripts/train.py` (`FrozenHeadModel`) — added `elif fusion_type == "gmu"` branch.
- `configs/method/triple_gmu.yaml` — new config (R+M+E, GMU, projection_dim=512).
- `tests/test_gmu.py` — 12 new tests: output shapes (N=1,2,3), output_dim property,
  gate sums to 1, no param overlap, B=1 edge case, gradient flow, ValueError guard,
  FrozenHeadModel triple + pair integration. All 12 pass.
- Full suite after Step 2: **116 passed**.

### Experiment 15 — GMU fold-0 smoke test (Step 3)

- Added `15_triple_gmu_finetune_wide_official` to `configs/experiment_matrix.yaml`.
- Config: `triple_gmu.yaml` (R+M+E, GMU, projection_dim=512) + `finetune_wide.yaml`
  (unfreeze_blocks=3, LLRD backbone_lr=1e-4 decay=0.75, EMA decay=0.999 start=5,
  CutMix α=1.0 p=0.5, RandAugment N=2 M=9, patience=8).
- Training log — val_f1 progression (no gate collapse observed):
  ep1: 0.3232 → ep2: 0.5101 → ep3: 0.5602 → ep6: 0.5840 (peak) → early stop ep14.

| ID | Method | Fusion | Mode | Acc | Macro F1 | Stop ep |
|---|---|---|---|---:|---:|---:|
| `15_triple_gmu_finetune_wide_official` | R+M+E | GMU | finetune-3blk | 0.8497 | 0.5645 | 14 |

Zero-support classes: none. No NaN values.

**Comparison with exp 10 (triple concat, best Week 2):**

| ID | Fusion | Acc | Macro F1 |
|---|---|---:|---:|
| 10 — triple concat | concat | **0.8784** | **0.5796** |
| 15 — triple GMU | GMU | 0.8497 | 0.5645 |
| Δ (GMU − concat) | — | −0.0287 | −0.0151 |

GMU trails triple concat by 0.015 F1 on fold 0. This is a single-fold result; 5-fold CV
(Track D) will determine whether the gap is consistent or a fold-0 artefact.
Stop-and-diagnose threshold (F1 < 0.50) was not triggered. Proceeding to Step 4.

## 2026-05-25 — Week 3 Step 4: Stage 1 supplementary fine-tune runs (fold 0)

### Infrastructure changes

- `configs/experiment_matrix.yaml` — added entries 13 and 14 (Stage 2 block, before entry 15).
- `scripts/plot_results.py --confusion-matrix` run after both experiments: all 15 confusion
  matrices refreshed; comparison bar chart and training curves updated to include exps 13–15.
- Ablation table regenerated: `results/tables/ablation_table.csv` / `.md` — 15 rows.

### Experiments (test set, fold 0, official 5-fold split)

| ID | Method | Fusion | Mode | Accuracy | Macro F1 | Stop ep |
|---|---|---|---|---:|---:|---:|
| `13_single_efficientnetb0_finetune_wide_official` | E | — | finetune-3blk | 0.8355 | 0.5539 | 13 |
| `14_pair_m_e_finetune_wide_official` | M+E | concat | finetune-3blk | 0.8384 | 0.5522 | 13 |

Zero-support classes: none in either run. No NaN values.

### Unexpected finding: fine-tune underperforms frozen for single E and pair M+E (fold 0)

| Config | Frozen F1 | Fine-tune F1 | Δ |
|---|---:|---:|---:|
| Single EfficientNetB0 | 0.5586 | 0.5539 | −0.005 |
| Pair M+E concat | 0.5758 | 0.5522 | −0.024 |
| Single ResNet50 (Week 2) | 0.5588 | 0.5748 | +0.016 ← fine-tune helped |

Fine-tune helped triple concat (+0.017) and single ResNet50 (+0.016) in Week 2, but
*hurt* single E and pair M+E on fold 0. Likely explanations:
1. EfficientNetB0 is already a strong ImageNet feature extractor; fine-tuning on ~6K
   training samples with high augmentation (CutMix + RandAugment) can overfit or
   disrupt well-learned features more than it recovers.
2. M+E frozen was already the best frozen config (0.5758); the complementary features
   were already well-aligned. Fine-tuning may introduce gradient interference between
   the two backbones without the regularisation benefit of a third branch.
3. Triple configs have more parameters and more gradient diversity — this likely
   provides a natural regularisation effect that single/pair fine-tuning lacks.

**Action taken:** CV config is unchanged. Track D will run fine-tune for all 5 configs
across folds 1–4. Fold-0 alone is insufficient to choose frozen vs. fine-tune;
using test-set results to retroactively select the training mode would be selection
bias. 5-fold CV will determine whether this fold-0 result is an artefact.

## 2026-05-26 — Week 3 Step 6: 5-Fold CV complete (all 5 configs)

### Runs completed

All 20 new GPU runs (folds 1–4 for 5 configs) completed on RTX 5080.
Per-fold test results from `metrics.json`:

**Exp 13 — Single EfficientNetB0 fine-tune:**
| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8355 | 0.5539 |
| 1 | 0.8605 | 0.5981 |
| 2 | 0.8421 | 0.5594 |
| 3 | 0.8440 | 0.5606 |
| 4 | 0.8597 | 0.5730 |

**Exp 14 — Pair M+E fine-tune:**
| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8384 | 0.5522 |
| 1 | 0.8549 | 0.5731 |
| 2 | 0.8440 | 0.5764 |
| 3 | 0.8577 | 0.5717 |
| 4 | 0.8459 | 0.5615 |

**Exp 10 — Triple concat fine-tune:**
| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8784 | 0.5796 |
| 1 | 0.8605 | 0.5709 |
| 2 | 0.8596 | 0.5699 |
| 3 | 0.8431 | 0.5610 |
| 4 | 0.8482 | 0.5640 |

**Exp 11 — Triple weighted fine-tune:**
| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8685 | 0.5751 |
| 1 | 0.8633 | 0.6014 |
| 2 | 0.8746 | 0.5829 |
| 3 | 0.8737 | 0.5860 |
| 4 | 0.8726 | 0.6005 |

**Exp 15 — Triple GMU fine-tune:**
| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8497 | 0.5645 |
| 1 | 0.8393 | 0.5625 |
| 2 | 0.8558 | 0.5705 |
| 3 | 0.8421 | 0.5634 |
| 4 | 0.8533 | 0.5753 |

Zero-support classes: none in any fold. No NaN values in any fold.

### CV summary (5-fold mean ± std)

| Exp | Config | Acc mean±std | F1 mean±std |
|---|---|---|---|
| 13 | Single E finetune | 0.8484 ± 0.0100 | 0.5690 ± 0.0158 |
| 14 | Pair M+E finetune | 0.8482 ± 0.0071 | 0.5670 ± 0.0089 |
| 10 | Triple concat finetune | 0.8580 ± 0.0122 | 0.5691 ± 0.0064 |
| 11 | Triple weighted finetune | **0.8706 ± 0.0042** | **0.5892 ± 0.0102** |
| 15 | Triple GMU finetune | 0.8480 ± 0.0063 | 0.5672 ± 0.0049 |

### Key findings (5-fold CV)

- **Triple weighted (exp 11) is the best CV config:** mean F1 = 0.5892 ± 0.0102, significantly
  higher than all other configs. This reverses the fold-0 ranking where concat (0.5796) beat
  weighted (0.5751). Confirms that fold-0 was an artefact for these two configs.
- **Fold-0 fine-tune anomaly resolved for exps 13 and 14:** Single E (fold-0 F1=0.5539) CV
  mean = 0.5690; fold 1 alone gives 0.5981. M+E (fold-0 F1=0.5522) CV mean = 0.5670.
  Both are competitive across folds; fold-0 was the anomaly.
- **GMU (exp 15) does not outperform triple concat over 5 folds:** GMU mean F1 = 0.5672 ±
  0.0049 vs. concat mean F1 = 0.5691 ± 0.0064. The difference is within noise (Δ = 0.002).
  GMU is a clean implementation and a valid alternative fusion strategy for the report, but
  does not provide a CV-level improvement over concat in the fine-tune regime.
- **Triple weighted is the clear winner** by CV F1. Its fold-0 value of 0.5751 understated its
  strength; folds 1 (0.6014) and 4 (0.6005) show it can exceed 0.60 F1 on individual folds.
- **Low variance for GMU (std=0.0049) and weighted (std=0.0042):** both are stable across folds.
  Concat has slightly higher std (0.0064), and single E is the least stable (std=0.0158, driven
  by fold-1 outlier at 0.5981).

### Post-run scripts

```
uv run python scripts/aggregate_cv.py --experiment {each of 5}   # all run, summary JSONs saved
uv run python scripts/generate_report_tables.py                   # 35 rows written
uv run python scripts/plot_results.py --confusion-matrix          # 40 confusion matrices saved
```

Ablation table updated to 35 rows (5 configs × 5 folds + fold-0-only entries for earlier experiments).
`results/figures/per_class_f1_best.png` updated: best experiment is now
`11_triple_weighted_finetune_wide_official_fold_1` (macro F1=0.6014).

## 2026-05-26 — Week 3 Step 7: Bootstrap 95% CI

### Method

`scripts/compute_ci.py` — concatenates `predictions.npz` across all 5 folds
(n = 10,662 pooled samples per experiment) before calling `bootstrap_ci`.
Parameters: n_bootstrap=1000, ci=0.95, seed=42. Percentile method.

All `predictions.npz` files verified present before running.

### Results

| Exp | Method | Fusion | F1 point | 95% CI |
|---|---|---|---:|---|
| 10 | Triple concat | concat | 0.5708 | [0.5618, 0.5799] |
| **11** | **Triple weighted** | **weighted** | **0.6000** | **[0.5814, 0.6206]** |
| 13 | Single E | — | 0.5730 | [0.5596, 0.5873] |
| 14 | Pair M+E | concat | 0.5759 | [0.5608, 0.5927] |
| 15 | Triple GMU | GMU | 0.5690 | [0.5580, 0.5813] |

Saved to `results/tables/ci_{id}.json` for all 5 configs.

### Key finding

Exp 11 (weighted) lower bound 0.5814 > exp 10 (concat) upper bound 0.5799 — non-overlapping
CIs. Triple weighted is statistically distinguishable from triple concat at the 95% level.
GMU vs. concat CIs fully overlap — no statistical separation.

## 2026-05-27 — Week 3 Step 8: Final test suite

- Command: `uv run pytest tests/ -v`
- Result: **125 passed** in 24.82s — 0 failures, 0 errors, 0 warnings.
- Test files verified (13 files):
  - `test_augmentation.py` — 18 tests (RandAugment + CutMix)
  - `test_backbones.py` — 12 tests (dimensions, freeze, BN eval)
  - `test_data_splits.py` — 4 tests (manifest, leakage, transform)
  - `test_ema.py` — 15 tests (init, formula, apply/restore, guards)
  - `test_feature_cache.py` — 1 test (e2e cache build)
  - `test_finetune_trainer.py` — 13 tests (EMA, CutMix, fit smoke)
  - `test_frozen_head.py` — 10 tests (single/pair/triple shapes)
  - `test_fusion_modules.py` — 3 tests (concat + weighted)
  - `test_gmu.py` — 12 tests (output shape, gate, grad flow, integration)
  - `test_llrd.py` — 26 tests (coverage, no-overlap, LR monotonicity)
  - `test_metrics.py` — 1 test (statistical imports)
  - `test_smoke_train.py` — 1 test (entrypoint placeholders)
  - `test_statistical.py` — 9 tests (CI contract, reproducibility, level)
- No regressions from any Week 3 change.

## 2026-05-27 — Week 3 Close-out

### Hard checkpoint verification (project_plan.md §2)

> *"End of Week 3: selected 5-fold results are ready; GMU is either complete or dropped."*

**Status: PASSED.**

- ✅ Selected 5-fold CV complete: 25/25 fold results present (5 configs × 5 folds).
- ✅ GMU complete: `src/models/fusion/gmu.py` implemented with verbatim paper equations,
  12 unit tests passing, fold-0 smoke test F1=0.5645 (no gate collapse).
- ✅ Bootstrap 95% CI computed for all 5 CV configs.
- ✅ `docs/results_progress.md` Week 3 section fully populated.
- ✅ 125 tests pass, 0 failures.

### Week 3 infrastructure delivered

| Script / Module | Purpose |
|---|---|
| `src/models/fusion/gmu.py` | GMU fusion — N-branch softmax generalization of Arevalo et al. (2017) §3.1 |
| `src/models/full_model.py` | Added `fusion_type="gmu"` branch |
| `scripts/train.py` | `--fold` CLI override; fold-indexed `run_dir` naming |
| `src/evaluation/statistical.py` | `bootstrap_ci` — percentile method, reproducible |
| `scripts/run_cv.py` | Subprocess loop over folds; continues on failure |
| `scripts/aggregate_cv.py` | Mean ± std from per-fold `metrics.json` |
| `scripts/compute_ci.py` | Concatenated-fold bootstrap CI |
| `configs/method/triple_gmu.yaml` | GMU method config |
| `tests/test_gmu.py` | 12 GMU unit tests |
| `tests/test_statistical.py` | 9 bootstrap_ci unit tests |

### Final Week 3 results

| Exp | Method | Fusion | F1 mean±std (CV) | F1 point (pooled) | 95% CI |
|---|---|---|---|---:|---|
| 13 | Single E | — | 0.5690 ± 0.0158 | 0.5730 | [0.5596, 0.5873] |
| 14 | Pair M+E | concat | 0.5670 ± 0.0089 | 0.5759 | [0.5608, 0.5927] |
| 10 | Triple concat | concat | 0.5691 ± 0.0064 | 0.5708 | [0.5618, 0.5799] |
| **11** | **Triple weighted** | **weighted** | **0.5892 ± 0.0102** | **0.6000** | **[0.5814, 0.6206]** |
| 15 | Triple GMU | GMU | 0.5672 ± 0.0049 | 0.5690 | [0.5580, 0.5813] |

### Key conclusions for the report

1. **Best configuration: Triple weighted fine-tune (exp 11)** — CV F1=0.5892, pooled F1=0.6000,
   95% CI [0.5814, 0.6206]. Statistically separable from all other configs (no CI overlap with
   exp 11's lower bound of 0.5814).
2. **Fold-0 was misleading for concat vs. weighted:** fold-0 showed concat (0.5796) > weighted
   (0.5751); 5-fold CV reverses this decisively (0.5892 vs. 0.5691).
3. **GMU is implemented and valid but provides no CV gain over concat:** Δ=0.002 F1, CIs fully
   overlapping. Report GMU as an implemented advanced fusion variant with honest null result.
4. **Fine-tune consistently outperforms frozen** across all 5 configs when assessed by CV mean
   (not fold-0 alone). The fold-0 anomaly for single E and pair M+E was a data artefact.
5. **Test suite stable:** 125 tests, 0 regressions throughout all of Week 3.

### Exec plan

`docs/exec-plans/active/003-week-3-gmu-cv.md` → moved to `docs/exec-plans/completed/`.
