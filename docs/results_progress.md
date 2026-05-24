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

**Test suite:** 32 passed.

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

### Figures produced (Step 3d)

| Figure | Path | Description |
|---|---|---|
| Comparison bar chart | `results/figures/comparison_bar_chart.png` | Accuracy + macro F1 across all experiments |
| Training curves | `results/figures/training_curves.png` | Val accuracy + macro F1 per epoch, all runs |
| Per-class F1 (best) | `results/figures/per_class_f1_best.png` | Per-class F1 for best frozen config (07) |
| Confusion matrices | `results/runs/{id}/confusion_matrix.png` | One per completed experiment (01–09) |

### Results — Fine-tuning (fold 0) — *pending Steps 4–9*

| Experiment | Backbones | Fusion | Mode | Acc | Macro F1 | Stop epoch |
|---|---|---|---|---:|---:|---:|
| `10_triple_concat_finetune_wide_official` | R+M+E | concat | finetune-wide | — | — | — |
| `11_triple_weighted_finetune_wide_official` | R+M+E | weighted | finetune-wide | — | — | — |
| `12_single_resnet50_finetune_wide_official` | R | — | finetune-wide | — | — | — |

---

## Week 3 — Focused Strengthening
*Planned: ~2026-05-25 to 2026-05-31*

### Planned milestones

- GMU fusion implementation (`src/models/fusion/gmu.py`)
- Selected 5-fold CV on: best single (E), best pair (M+E), triple concat, triple weighted, triple GMU
- Bootstrap 95% CI for headline results

### Results — Selected 5-fold CV — *pending*

| Method | Fusion | Mode | Acc mean±std | F1 mean±std |
|---|---|---|---|---|
| EfficientNetB0 | — | best from W2 | — | — |
| M+E | concat | best from W2 | — | — |
| R+M+E | concat | best from W2 | — | — |
| R+M+E | weighted | best from W2 | — | — |
| R+M+E | GMU | best from W2 | — | — |

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
