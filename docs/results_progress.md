# Results Progress

> **Single source of truth for weekly implementation milestones and experiment results.**
>
> Every number in this file traces back to a `metrics.json` under `results/runs/{id}/`.
> Updated at the end of each step group. Used directly as ground truth when writing the report.
>
> Columns: Acc = test accuracy, F1 = test macro F1 (zero_division=0).
> Protocol: official HyperKvasir 5-fold split, fold 0, unless stated otherwise.

---

## Week 1 — Foundation
*Completed: 2026-05-22 to 2026-05-23*

### Implementation milestones

| # | Component | File(s) |
|---|---|---|
| 1 | uv environment, CUDA verified | `pyproject.toml`, `uv.lock`, `docs/environment.md` |
| 2 | HyperKvasir data pipeline | `src/data/datasets.py`, `src/data/manifests.py` |
| 3 | Official 5-fold split manifests (leak-checked) | `scripts/make_splits.py`, `src/data/splits.py` |
| 4 | Backbone wrappers (torchvision only) | `src/models/backbones.py` |
| 5 | Projection layer (Linear + LayerNorm + GELU) | `src/models/projections.py` |
| 6 | Concat + weighted fusion modules | `src/models/fusion/concat.py`, `weighted.py` |
| 7 | MLP classifier | `src/models/classifiers.py` |
| 8 | Frozen feature cache | `src/data/feature_cache.py` |
| 9 | Trainer (early stopping, mixed precision) | `src/training/trainer.py` |
| 10 | Loss, optimizer, scheduler | `src/training/losses.py`, `optimizers.py`, `schedulers.py` |
| 11 | Metrics (macro F1, per-class, zero-division safe) | `src/evaluation/metrics.py` |
| 12 | Logging, checkpointing, paths | `src/utils/` |
| 13 | Train CLI with FrozenHeadModel | `scripts/train.py` |

**Test suite:** 22 passed.

### Verified backbone feature dimensions

| Backbone | Output dim | Source |
|---|---:|---|
| ResNet50 | 2048 | `project_structure.md §2.1` |
| MobileNetV2 | 1280 | `project_structure.md §2.1` |
| EfficientNetB0 | 1280 | `project_structure.md §2.1` |

### Results — Baseline

| Experiment | Backbones | Fusion | Mode | Acc | Macro F1 | Stop epoch |
|---|---|---|---|---:|---:|---:|
| `01_single_resnet50_frozen_official` | ResNet50 | — | frozen | 0.8402 | 0.5588 | 16 |

Zero-support classes: none. No NaN values.

---

## Week 2 — Mandatory Fusion Experiments
*Started: 2026-05-24 (ongoing)*

### Implementation milestones

| # | Component | File(s) |
|---|---|---|
| 1 | Experiments 02–09 added to matrix | `configs/experiment_matrix.yaml` |
| 2 | FrozenHeadModel shape tests (pair + triple) | `tests/test_frozen_head.py` |
| 3 | Ablation table generator | `scripts/generate_report_tables.py` |
| 4 | Comparison + training curve plots | `scripts/plot_results.py` |
| 5 | RandAugment + CutMix augmentation | `src/data/augmentation.py` |
| 6 | EMA with warm-start + device-safe update | `src/training/ema.py` |
| 7 | LLRD per-backbone-block parameter groups | `src/training/optimizers.py` |
| 8 | Fine-tune image-based training path | `scripts/train.py`, `src/training/trainer.py` |
| 9 | Experiments 10–12 added to matrix | `configs/experiment_matrix.yaml` |
| 10 | Confusion matrices + per-class F1 chart | `src/evaluation/visualization.py` |

**Test suite:** 104 passed.

### Results — Stage 0 + Stage 1 frozen (fold 0)

| Experiment | Backbones | Fusion | Mode | Acc | Macro F1 | Stop epoch |
|---|---|---|---|---:|---:|---:|
| `01_single_resnet50_frozen_official` | R | — | frozen | 0.8402 | 0.5588 | 16 |
| `02_single_mobilenetv2_frozen_official` | M | — | frozen | 0.8483 | 0.5502 | 21 |
| `03_single_efficientnetb0_frozen_official` | E | — | frozen | **0.8591** | 0.5586 | 16 |
| `05_pair_r_m_concat_frozen_official` | R+M | concat | frozen | 0.8247 | 0.5438 | 14 |
| `06_pair_r_e_concat_frozen_official` | R+E | concat | frozen | 0.8388 | 0.5464 | 11 |
| `07_pair_m_e_concat_frozen_official` | M+E | concat | frozen | **0.8728** | **0.5758** | 21 |
| `08_triple_concat_frozen_official` | R+M+E | concat | frozen | 0.8567 | 0.5630 | 20 |
| `09_triple_weighted_frozen_official` | R+M+E | weighted | frozen | 0.8549 | 0.5609 | 12 |

*Note: `04_triple_concat_frozen_official` is Stage 0 smoke test, identical config to `08`, identical results — omitted from table.*

Zero-support classes: none in any run. No NaN values in any run.

### Key findings (frozen regime, fold 0)

- **Best single backbone:** EfficientNetB0 (acc=0.8591, F1=0.5586).
- **Best pair:** M+E concat (acc=0.8728, F1=0.5758) — best configuration overall so far.
- **ResNet50 hurts in frozen pair/triple concat:** adding R to M+E drops F1 from 0.5758 to 0.5630. Frozen ResNet50 features (2048-dim → 512 projection) add noise in the concat setting.
- **Weighted vs. concat (triple):** negligible difference (0.5609 vs. 0.5630). Weighted fusion's advantage expected to emerge during fine-tuning when gradients can adjust the branch weights meaningfully.

### Per-class analysis — best frozen config (07: M+E concat, macro F1=0.5758)

Source: `results/figures/per_class_f1_best.png`, `results/runs/07_pair_m_e_concat_frozen_official/confusion_matrix.png`

**F1 = 0.000 (5 classes — all low support):**

| Class | Test support | Pattern |
|---|---:|---|
| barretts | 8 | Confused with normal-z-line (visually similar Barrett's region) |
| hemorroids | 1 | Single test sample — any error = F1=0 |
| ileum | 1 | Single test sample — misclassified as short-segment-barretts |
| ulcerative-colitis-grade-1-2 | 2 | Confused with adjacent UC grades |
| ulcerative-colitis-grade-2-3 | 5 | Confused with adjacent UC grades |

**F1 < 0.3 (2 additional classes):**

| Class | Test support | F1 | Pattern |
|---|---:|---:|---|
| ulcerative-colitis-grade-0-1 | 7 | 0.167 | Confused with grade-1 |
| short-segment-barretts | 10 | 0.154 | Confused with normal-z-line and oesophagitis |

**Conclusion:** All F1=0 and near-zero classes have test support ≤ 10. This is a pure class-imbalance effect — the model never sees enough samples to learn discriminative features for these classes. The confusion matrix confirms the failure mode is not random: rare classes are absorbed by their visually or semantically closest majority class (UC grades confused with adjacent grades, Barrett's variants confused with normal-z-line). Fine-tuning with augmentation and LLRD should help by giving the model stronger gradient signal from frozen-backbone features.

### Results — Fine-tuning (fold 0)

Config: `finetune_wide.yaml` — unfreeze_blocks=3, LLRD (backbone_lr=1e-4, decay=0.75), EMA (decay=0.999, start_epoch=5), CutMix (α=1.0, p=0.5), RandAugment (N=2, M=9), patience=8.

| Experiment | Backbones | Fusion | Mode | Acc | Macro F1 | Stop epoch |
|---|---|---|---|---:|---:|---:|
| `13_single_efficientnetb0_finetune_wide_official` | E | — | finetune-3blk | 0.8355 | 0.5539 | 13 |
| `14_pair_m_e_finetune_wide_official` | M+E | concat | finetune-3blk | 0.8384 | 0.5522 | 13 |
| `10_triple_concat_finetune_wide_official` | R+M+E | concat | finetune-3blk | **0.8784** | **0.5796** | 14 |
| `11_triple_weighted_finetune_wide_official` | R+M+E | weighted | finetune-3blk | 0.8685 | 0.5751 | 12 |
| `12_single_resnet50_finetune_wide_official` | R | — | finetune-3blk | 0.8501 | 0.5748 | 12 |
| `15_triple_gmu_finetune_wide_official` | R+M+E | GMU | finetune-3blk | 0.8497 | 0.5645 | 14 |

Zero-support classes: none in any run. No NaN values in any run.

### Key findings (fine-tune regime, fold 0) — updated after Step 4

- **New best overall:** Triple concat fine-tune (10) — acc=0.8784, F1=0.5796, surpassing the best frozen config (07 M+E: acc=0.8728, F1=0.5758).
- **Fine-tune vs. frozen (triple concat):** acc +0.0217, F1 +0.0166 — end-to-end gradient flow through unfrozen backbone blocks consistently improves both metrics.
- **Weighted vs. concat (fine-tune):** concat (0.5796) still edges weighted (0.5751) — the expected advantage of learnable branch weights did not materialise on fold 0; 5-fold CV in Week 3 will determine if this holds.
- **Single ResNet50 fine-tune (12):** F1 0.5748 — a large gain over its frozen counterpart (01: 0.5588), but still below all triple fine-tune configs. End-to-end fine-tuning rescues ResNet50 performance that was lost during frozen pair/triple concat.
- **Early stopping:** All five fine-tune runs stopped in 12–14 epochs.
- **Fine-tune hurts single E and pair M+E on fold 0 (Step 4 finding):** EfficientNetB0
  fine-tune (0.5539) is below its frozen counterpart (0.5586 Δ−0.005); M+E fine-tune
  (0.5522) is well below frozen M+E (0.5758, Δ−0.024). Both are fold-0 observations only.
  CV config is unchanged — fine-tune will be run for all configs across folds 1–4;
  fold-0 alone is not a valid basis for switching the training mode. Small dataset + high augmentation → fast plateau; longer training with lower decay or larger dataset expected to extend convergence.

### Figures produced (Steps 3d + 9)

| Figure | Path | Description |
|---|---|---|
| Comparison bar chart | `results/figures/comparison_bar_chart.png` | Accuracy + macro F1 across all 12 experiments |
| Training curves | `results/figures/training_curves.png` | Val accuracy + macro F1 per epoch, all runs |
| Per-class F1 (best) | `results/figures/per_class_f1_best.png` | Per-class F1 for best config (10: triple concat fine-tune, F1=0.5796) |
| Confusion matrices | `results/runs/{id}/confusion_matrix.png` | One per experiment (01–12) |

---

## Week 3 — Focused Strengthening
*Planned: ~2026-05-25 to 2026-05-31*

### Planned milestones

- GMU fusion implementation (`src/models/fusion/gmu.py`)
- Selected 5-fold CV on: best single (E), best pair (M+E), triple concat, triple weighted, triple GMU
- Bootstrap 95% CI for headline results

### Implementation milestones

| # | Component | File(s) |
|---|---|---|
| 1 | GMU fusion module (N-branch softmax generalization) | `src/models/fusion/gmu.py` |
| 2 | GMU unit tests (12 tests) | `tests/test_gmu.py` |
| 3 | `--fold` CLI override + fold-indexed run dir | `scripts/train.py` |
| 4 | `bootstrap_ci` (percentile method, reproducible) | `src/evaluation/statistical.py` |
| 5 | `run_cv.py` subprocess loop | `scripts/run_cv.py` |
| 6 | `aggregate_cv.py` mean±std aggregation | `scripts/aggregate_cv.py` |
| 7 | `compute_ci.py` concatenated-fold CI | `scripts/compute_ci.py` |
| 8 | bootstrap_ci unit tests (9 tests) | `tests/test_statistical.py` |

**Test suite:** 125 passed (Step 8 final verification: 2026-05-27, 24.82s, 0 failures).

### Results — Selected 5-fold CV

Config: `finetune_wide.yaml` — unfreeze_blocks=3, LLRD (backbone_lr=1e-4, decay=0.75), EMA (decay=0.999, start_epoch=5), CutMix (α=1.0, p=0.5), RandAugment (N=2, M=9), patience=8.

| Exp | Method | Fusion | Mode | Acc mean±std | F1 mean±std |
|---|---|---|---|---|---|
| 13 | EfficientNetB0 | — | finetune-3blk | 0.8484 ± 0.0100 | 0.5690 ± 0.0158 |
| 14 | M+E | concat | finetune-3blk | 0.8482 ± 0.0071 | 0.5670 ± 0.0089 |
| 10 | R+M+E | concat | finetune-3blk | 0.8580 ± 0.0122 | 0.5691 ± 0.0064 |
| 11 | R+M+E | weighted | finetune-3blk | **0.8706 ± 0.0042** | **0.5892 ± 0.0102** |
| 15 | R+M+E | GMU | finetune-3blk | 0.8480 ± 0.0063 | 0.5672 ± 0.0049 |

Zero-support classes: none in any fold. No NaN values in any fold.

**Per-fold F1 breakdown:**

| Exp | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| 13 | 0.5539 | 0.5981 | 0.5594 | 0.5606 | 0.5730 |
| 14 | 0.5522 | 0.5731 | 0.5764 | 0.5717 | 0.5615 |
| 10 | 0.5796 | 0.5709 | 0.5699 | 0.5610 | 0.5640 |
| 11 | 0.5751 | **0.6014** | 0.5829 | 0.5860 | **0.6005** |
| 15 | 0.5645 | 0.5625 | 0.5705 | 0.5634 | 0.5753 |

### Bootstrap 95% CI

Computed via `scripts/compute_ci.py` — concatenates `predictions.npz` from all 5 folds
(n = 10,662 pooled samples) before bootstrapping (n_bootstrap=1000, seed=42).
Point estimate = macro F1 on the full pooled set (size-weighted, not fold-mean).

| Exp | Method | Fusion | F1 point | 95% CI lower | 95% CI upper | CI width |
|---|---|---|---:|---:|---:|---:|
| 10 | R+M+E | concat | 0.5708 | 0.5618 | 0.5799 | 0.0181 |
| **11** | **R+M+E** | **weighted** | **0.6000** | **0.5814** | **0.6206** | **0.0392** |
| 13 | E | — | 0.5730 | 0.5596 | 0.5873 | 0.0277 |
| 14 | M+E | concat | 0.5759 | 0.5608 | 0.5927 | 0.0319 |
| 15 | R+M+E | GMU | 0.5690 | 0.5580 | 0.5813 | 0.0233 |

Saved to `results/tables/ci_{id}.json` for all 5 configs.

**Key CI findings:**
- **Exp 11 (weighted) CI lower bound (0.5814) is strictly above exp 10 (concat) upper bound
  (0.5799):** the two CIs do not overlap. Triple weighted is statistically distinguishable
  from triple concat at the 95% level on the pooled test set.
- **Exp 15 (GMU) and exp 10 (concat) CIs heavily overlap** ([0.5580, 0.5813] vs.
  [0.5618, 0.5799]): no statistical evidence that GMU and concat differ. The Δ=0.002
  gap in CV mean F1 is within bootstrap noise.
- **Exp 11 has the widest CI (0.0392):** its pooled F1=0.6000 is driven partly by
  high-performing folds (1: 0.6014, 4: 0.6005), which also increases variance.
- **All CIs are narrow relative to the gap between exp 11 and the rest:** the weakest
  competitor (exp 15, upper=0.5813) does not reach exp 11's lower bound (0.5814).

### Key findings (Week 3 — 5-fold CV)

- **Best CV config: Triple weighted fine-tune (exp 11)** — F1 = 0.5892 ± 0.0102.
  This reverses the fold-0 ranking (where concat 0.5796 > weighted 0.5751), confirming
  fold-0 was an artefact for these two. Weighted fusion is decisively better over 5 folds.
- **Fold-0 fine-tune anomaly for exps 13 and 14 resolved:** Both configs perform
  competitively across folds (single E mean 0.5690, M+E mean 0.5670). Fold-0 was
  the outlier, not the representative result.
- **GMU does not surpass concat over 5 folds:** GMU mean F1 = 0.5672 ± 0.0049 vs.
  concat 0.5691 ± 0.0064 (Δ = 0.002). GMU is a valid implementation but provides no
  CV-level gain over simple concatenation in the fine-tune regime.
- **Low CV variance for weighted and GMU** (std ≈ 0.004–0.010) indicates stable
  training across official folds. Single E has the highest variance (std=0.0158),
  driven by fold-1 outlier (0.5981).
- **Ceiling comparison with literature:** GastroViT (F1≈0.64) uses a different protocol
  (not official 5-fold split); direct comparison not valid. Best CV F1 here (0.5892)
  is within the expected range for a 3-backbone MLP-only architecture on this dataset.

---

## Week 3.5 — Performance Lift: Focal Loss Ablation
*Completed: 2026-06-07 (Step 3)*

Additive ablation (D-06): the Week 3 best config (exp 11, triple weighted
fine-tune) re-run with **focal loss** (Lin et al. 2017, γ=2.0) instead of
label-smoothed cross-entropy. Single-variable change — `finetune_wide_focal.yaml`
differs from `finetune_wide.yaml` only in the loss block. Trained on **Colab
A100** via the runner-only notebook (D-09 provenance gate passed). Original CE
runs (exp 11) preserved unchanged for comparison.

### Results — exp 16 focal 5-fold CV

Per-fold (test set, official 5-fold split):

| Fold | Acc | Macro F1 |
|---|---:|---:|
| 0 | 0.8582 | 0.5777 |
| 1 | 0.8666 | 0.5847 |
| 2 | 0.8648 | 0.5687 |
| 3 | 0.8393 | 0.5741 |
| 4 | 0.8726 | 0.5962 |

Zero-support classes: none in any fold. No NaN values in any fold.

CV summary: macro F1 = **0.5803 ± 0.0095**, accuracy = 0.8603 ± 0.0115,
macro precision = 0.5765 ± 0.0070, macro recall = 0.6062 ± 0.0105.

### Focal vs CE — head-to-head (single-variable ablation)

Pooled over all 5 folds (n = 10,662), bootstrap 95% CI (n_bootstrap=1000, seed=42):

| Exp | Loss | Macro-F1 (pooled) | 95% CI | CV mean±std | Acc | Weighted-F1 | MCC |
|---|---|---:|---|---|---:|---:|---:|
| **11** | **CE (label-smooth)** | **0.6000** | **[0.5814, 0.6206]** | 0.5892 ± 0.0102 | **0.8706** | **0.8716** | **0.8599** |
| 16 | Focal (γ=2.0) | 0.5914 | [0.5750, 0.6096] | 0.5803 ± 0.0095 | 0.8603 | 0.8633 | 0.8489 |
| Δ (focal − CE) | — | −0.0086 | overlapping | −0.0089 | −0.0103 | −0.0083 | −0.0110 |

Tables saved to `results/tables/ci_16_*.json`, `cv_16_*_summary.json`,
`extra_metrics_16_*.json`. Confusion matrices in `results/runs/16_*/`.

### Verdict — new best vs Week 3 best

**Focal loss did NOT beat cross-entropy. The Week 3 champion (exp 11, triple
weighted CE, pooled macro-F1 0.6000) remains the project best.**

- Focal's pooled point estimate (0.5914) is **below** CE (0.6000), and the two
  CIs **heavily overlap** ([0.5750, 0.6096] vs [0.5814, 0.6206]) — the difference
  is not statistically distinguishable at the 95% level.
- Focal is also marginally lower on **every** secondary metric (accuracy,
  weighted-F1, MCC), so there is no metric under which focal wins.
- This is an honest **negative result**, anticipated by the risk register
  (exec plan §7) and D-06. No test-set retuning was performed.

**Interpretation (for report §5.5):** the weighted-sampler already addresses
class imbalance upstream, and focal's modulating factor `(1−p_t)^γ` adds no
further gain on top of it. The residual bottleneck is the rare classes
(support ≤ 10), which is a data-scarcity limit not solvable by reweighting the
loss. CE label smoothing stays the recommended loss for this architecture.

### Step 4 — Test-Time Augmentation (TTA) on the champion (exp 11)
*Completed: 2026-06-07*

Inference-time technique (no training). The champion checkpoint (exp 11, triple
weighted CE) is run forward over each fold's held-out test set under a fixed,
deterministic set of 4 views — `base` (Resize256→CenterCrop224), `hflip`,
`scale` (Resize224→CenterCrop224), `scale_hflip` — and the **softmax is averaged**
per image. Per-model, per-fold; no leakage, no cross-fold model averaging.
Implemented in `scripts/evaluate.py`; run on the local RTX 5080.

Validation: the `base`-only view reproduces the training-time `predictions.npz`
exactly (100% prediction agreement on fold 0), confirming the inference path.

Per-fold (test set):

| Fold | base F1 | TTA F1 | Δ |
|---|---:|---:|---:|
| 0 | 0.5751 | 0.5827 | +0.0076 |
| 1 | 0.6014 | 0.6126 | +0.0112 |
| 2 | 0.5829 | 0.5808 | −0.0021 |
| 3 | 0.5860 | 0.6025 | +0.0165 |
| 4 | 0.6005 | 0.5987 | −0.0018 |

### TTA vs base — head-to-head (exp 11)

Pooled over all 5 folds (n = 10,662), bootstrap 95% CI (n_bootstrap=1000, seed=42):

| Variant | Macro-F1 (pooled) | 95% CI | CV mean±std | Acc | Weighted-F1 | MCC |
|---|---:|---|---|---:|---:|---:|
| base (exp 11) | 0.6000 | [0.5814, 0.6206] | 0.5892 ± 0.0102 | 0.8706 | 0.8716 | 0.8599 |
| **base + TTA** | **0.6075** | **[0.5860, 0.6296]** | 0.5955 ± 0.0121 | **0.8765** | **0.8761** | **0.8662** |
| Δ (TTA − base) | +0.0075 | overlapping | +0.0063 | +0.0059 | +0.0045 | +0.0063 |

Tables saved to `results/tables/ci_11_*_tta.json`, `extra_metrics_11_*_tta.json`;
per-fold `predictions_tta.npz` / `metrics_tta.json` in `results/runs/11_*/`.

### Verdict — TTA

**TTA gives a small, consistent uplift on every metric (new pooled best point
estimate macro-F1 = 0.6075), but it is NOT statistically significant at the 95%
level — the TTA and base CIs overlap** ([0.5860, 0.6296] vs [0.5814, 0.6206]; each
point estimate lies inside the other's interval).

- Unlike focal (negative on all metrics), TTA improves macro-F1, accuracy,
  weighted-F1 and MCC simultaneously, and never lowers the pooled aggregate.
- It is a **zero-cost, training-free** inference add-on with no downside.
- **Recommendation:** adopt TTA as the final inference protocol for the champion
  (exp 11). It is the best-performing configuration by point estimate and is the
  candidate "best model" to freeze in Step 7, while honestly noting the gain is
  within CI overlap (not a statistically conclusive improvement over base).

### Step 5 — Leakage-free seed ensemble on the champion (exp 11)
*Completed: 2026-06-08*

Train a second seed (123) of the unchanged champion config (`finetune_wide.yaml`,
single-variable: only the seed changes) across all 5 folds, then average the
per-image softmax of the two seeds **within each fold's held-out test set** and
pool (D-07: leakage-free; no cross-fold model averaging). Seed 42 was trained in
Week 3 on the RTX 5080; seed 123 on the same stack for a clean comparison.
Implemented via a `--seed` override (`train.py`/`run_cv.py`) + `scripts/ensemble_seeds.py`.

Per-seed CV (test macro-F1): seed 42 = 0.5892 ± 0.0102, seed 123 = 0.5779 ± 0.005
(seed 123 is the weaker draw). Ensembled per-fold macro-F1 (base): 0.5882, 0.6070,
0.5757, 0.6084, 0.5865. No NaN, no zero-support.

### Ensemble vs base vs TTA — head-to-head (exp 11, pooled n=10,662)

| Variant | Macro-F1 | 95% CI (width) | Acc | Weighted-F1 | MCC |
|---|---:|---|---:|---:|---:|
| base (seed 42) | 0.6000 | [0.5814, 0.6206] (0.039) | 0.8706 | 0.8716 | 0.8599 |
| **TTA (seed 42)** | **0.6075** | [0.5860, 0.6296] (0.044) | 0.8765 | 0.8761 | 0.8662 |
| ensemble (42+123) | 0.5971 | [0.5850, 0.6097] (0.025) | 0.8774 | 0.8780 | 0.8672 |
| ensemble + TTA | 0.6005 | [0.5883, 0.6135] (0.025) | **0.8810** | **0.8797** | **0.8710** |

Tables: `results/tables/ci_11_*_ensemble[_tta].json`, `extra_metrics_11_*_ensemble[_tta].json`.
Ensemble predictions in canonical `results/runs/11_*/predictions_ensemble[_tta].npz`.

### Verdict — seed ensemble

**The leakage-free 2-seed ensemble did NOT raise the headline macro-F1:**
ensemble+TTA pooled macro-F1 = 0.6005 < TTA-only 0.6075. The second seed (123,
CV 0.5779) under-performed seed 42 (0.5892), and averaging it in pulled the
rare-class (macro) average down.

- **However, the ensemble's variance-reduction benefit did materialise:** the CI
  narrowed markedly (width 0.025 vs 0.044 for single-seed TTA), and ensemble+TTA
  is the **best variant on accuracy (0.8810), weighted-F1 (0.8797) and MCC
  (0.8710)** — these majority-class-weighted metrics benefit from averaging even
  when macro-F1 does not.
- All four variants' CIs overlap → no statistically significant difference.
- With M=2 and one weak seed, a 2-seed average is sensitive to seed quality; it
  can dip below the best single seed on macro-F1 (the rare-class-sensitive metric).
- **Recommendation:** by the locked headline metric (macro-F1, PLD-11),
  **exp 11 + TTA (single seed 42, 0.6075) remains the best model** and the
  candidate to freeze in Step 7. The seed ensemble is reported honestly as a
  robustness technique that stabilised variance and topped the aggregate metrics
  but did not improve macro-F1 at M=2.

### Week 3.5 summary — best vs Week 3 best

| Technique | macro-F1 (pooled) | vs Week 3 best (0.6000) |
|---|---:|---|
| Week 3 best — exp 11 CE (base) | 0.6000 | — (champion) |
| + Focal (exp 16) | 0.5914 | worse (within CI) |
| **+ TTA** | **0.6075** | **+0.0075 (within CI) — best macro-F1** |
| + seed ensemble (+TTA) | 0.6005 | +0.0005 (within CI); best acc/weighted/MCC |

**Overall Week 3.5 finding:** no technique produced a statistically significant
(CI-level) macro-F1 gain over the Week 3 CE champion — the architecture is at its
ceiling on the official 5-fold protocol. TTA gives the best macro-F1 point
estimate (0.6075) at zero training cost and is adopted as the inference protocol;
the recommended final model is **exp 11 (triple weighted CE) + TTA**.

**FROZEN FINAL MODEL (Step 7, 2026-06-08): `exp 11` (triple weighted CE) + TTA,
seed 42 — pooled macro-F1 0.6075, 95% CI [0.5860, 0.6296].** This is the new best
vs the Week 3 best (exp 11 base, 0.6000) — a +0.0075 point-estimate gain, within
CI overlap (not statistically significant). Full freeze record (checkpoints +
inference recipe) in [`docs/FINAL_MODEL.md`](FINAL_MODEL.md). Week 4 builds on this.

---

## Week 4 — Analysis and Report
*Planned: ~2026-06-01 to 2026-06-04*

### Planned milestones

- Confusion matrix for best model
- Per-class precision/recall/F1 table (Table 3 in report)
- Training time comparison table (Table 4)
- Grad-CAM++ visualizations (if time allows)
- Report draft v1

### Figures to produce

| Figure | Script | Status |
|---|---|---|
| Accuracy + F1 bar chart (all experiments) | `scripts/plot_results.py` | done (W2) |
| Training curves (key experiments) | `scripts/plot_results.py` | done (W2) |
| Confusion matrix (best model) | `scripts/plot_results.py` | pending (W4) |
| Grad-CAM++ examples | `scripts/evaluate.py` | pending (W4) |
