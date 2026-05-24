# 002 — Week 2: Mandatory Fusion Experiments

> **Status legend:** ✅ Done · ⬜ Pending
>
> Steps marked ✅ were completed on 2026-05-24. Pending steps begin at Step 3d.

---

## 1. Goal

Complete the **Stage 0 smoke tests** and **Stage 1 mandatory one-fold ablation** from
`project_plan.md` on **fold 0** using the official HyperKvasir 5-fold split protocol.

By the end of this plan the project must have:

- All 8 frozen single/pair/triple configurations run and saved.
- Fine-tune wide variants for at least the triple concat, triple weighted, and single
  ResNet50 configurations.
- An ablation table (Table 1) and result figures updated after every experiment batch.
- `docs/results_progress.md` fully populated through the end of Week 2.
- `uv run pytest tests/` passing throughout.

This is the Week 3 hard checkpoint prerequisite: "mandatory concat/weighted
experiments work on one fold."

---

## 2. Inputs

- `AGENTS.md`
- `project_structure.md` — especially §2 (verified backbone facts) and §6 (module contracts)
- `project_plan.md` — Stage 0, Stage 1 tables; locked design decisions PLD-04 through PLD-18
- `docs/decisions.md` — official split protocol decision (2026-05-23)
- `docs/results_progress.md` — weekly ground-truth results tracker (created 2026-05-24)
- `docs/experiment_log.md` — narrative run log
- `docs/known_issues.md` — KI-002 (macro F1 / imbalance)
- `configs/experiment_matrix.yaml` — entries 01–09 present; 10–12 to be added
- `configs/method/` — all 8 method configs present
- `configs/training/frozen_extraction.yaml`, `finetune_narrow.yaml`, `finetune_wide.yaml`
- `src/training/trainer.py` — implemented
- `src/models/fusion/concat.py`, `weighted.py` — implemented
- `src/models/fusion/gmu.py` — placeholder, do NOT implement this week
- `src/data/augmentation.py` — placeholder, needs implementation (Step 4)
- `src/training/ema.py` — placeholder, needs implementation (Step 5)
- `src/training/optimizers.py` — `build_adamw_with_llrd` is a skeleton (Step 6)
- `scripts/train.py` — frozen path implemented; fine-tune path missing (Step 7)
- `scripts/generate_report_tables.py` — implemented ✅
- `scripts/plot_results.py` — implemented ✅
- Fold manifests: `data/splits/hyperkvasir_official_5fold/fold_0.csv` (and folds 1–4)
- Feature caches built: `results/feature_cache/fold_0_{resnet50,mobilenetv2,efficientnetb0}_features.pt`

---

## 3. Scope

### Track A — Frozen experiments ✅ Complete

- ✅ Experiment matrix entries 02–09 added.
- ✅ `FrozenHeadModel` verified for pair and triple inputs (concat + weighted).
- ✅ `tests/test_frozen_head.py` added; 32 tests pass.
- ✅ All 8 frozen experiments run on fold 0.

### Track B — Fine-tuning path ⬜ Pending

- Implement `src/data/augmentation.py`: `apply_rand_augment()` and `apply_cutmix()`.
- Implement `src/training/ema.py`: `EMA` class with `update()`, `apply()`, `restore()`.
- Complete `build_adamw_with_llrd()` with proper per-backbone layer groups.
- Add fine-tuning code path to `scripts/train.py` (image-based, `MultiCNNFusionClassifier`).
- Add tests for EMA and LLRD.
- Run fine-tune wide for experiments 10, 11, 12.
- After experiments: run `plot_results.py` and update `docs/results_progress.md`.

### Track C — Observability ✅ Complete

- ✅ `scripts/generate_report_tables.py` — produces `results/tables/ablation_table.csv` and `.md`.
- ✅ `scripts/plot_results.py` — produces comparison bar chart, training curves, confusion matrices, per-class F1 chart.
- ✅ `docs/results_progress.md` — created with Week 1 and Week 2 frozen results, figures subsection, per-class analysis.
- ✅ **Step 3d** — confusion matrix per experiment + per-class F1 chart for best config (07: M+E concat).

---

## 4. Out of Scope

- GMU, AFF, LMF fusion implementations (Week 3).
- Running experiments on folds 1–4 (Stage 2 — Week 3).
- Bootstrap confidence intervals (Week 3).
- Grad-CAM++ visualization (Week 4).
- McNemar test (Week 4).
- MLflow wiring (deferred — JSON backend sufficient for this project scope).
- UMAP feature-space visualization (stretch).
- DeepWeeds or any secondary dataset.
- Cross-protocol (Effimix, GastroViT) reproduction.
- Non-MLP classifiers.
- Changing the official split protocol or any locked design decision (PLD-*).

---

## 5. Files Expected to Be Modified

### Track A ✅ Done
- `configs/experiment_matrix.yaml` — entries 02–09 added ✅
- `tests/test_frozen_head.py` — created ✅
- `docs/experiment_log.md` — updated ✅
- `docs/results_progress.md` — Week 2 frozen section populated ✅

### Track B ⬜ Partially complete
- `src/data/augmentation.py` — implement RandAugment and CutMix ✅
- `tests/test_augmentation.py` — 18 tests, all pass ✅
- `src/training/ema.py` — implement EMA class ✅
- `tests/test_ema.py` — 15 tests, all pass ✅
- `src/training/optimizers.py` — complete `build_adamw_with_llrd` with per-block groups ✅
- `tests/test_llrd.py` — 26 tests, all pass ✅
- `src/training/optimizers.py` — complete `build_adamw_with_llrd` per-layer grouping
- `scripts/train.py` — add fine-tune image-based training path
- `configs/experiment_matrix.yaml` — add entries 10–12
- `tests/test_augmentation.py` — new
- `tests/test_ema.py` — new
- `docs/experiment_log.md` — after fine-tune runs
- `docs/results_progress.md` — after fine-tune runs (Step 9 acceptance gate)

### Track C ⬜ Step 3d pending
- `scripts/plot_results.py` — extend with confusion matrix generation
- `src/evaluation/visualization.py` — implement confusion matrix and per-class chart helpers
- `results/figures/` — confusion matrices per experiment + per-class F1 chart

### Documentation (all tracks)
- `docs/results_progress.md` — after every experiment batch (mandatory acceptance gate)
- `docs/experiment_log.md` — after every run group
- `docs/known_issues.md` — for any new failure or instability
- `docs/decisions.md` — only if a locked decision or split policy changes

---

## 6. Step-by-Step Implementation Plan

### ✅ Step 1 — Add frozen experiment entries to experiment_matrix.yaml
*Done 2026-05-24. Entries 02–09 added using `hyperkvasir_23class_official.yaml`, fold 0.*

### ✅ Step 2 — Verify and test multi-backbone FrozenHeadModel
*Done 2026-05-24. `tests/test_frozen_head.py` created with 10 shape tests covering
single, pair, and triple backbone inputs for both concat and weighted fusion.
All 32 tests pass.*

### ✅ Step 3 — Run all Stage 0 + Stage 1 frozen experiments
*Done 2026-05-24. Experiments 02–09 all completed on fold 0 with no NaN values
and no zero-support classes. Results:*

| ID | Backbones | Fusion | Acc | Macro F1 |
|---|---|---|---:|---:|
| 02 | MobileNetV2 | — | 0.8483 | 0.5502 |
| 03 | EfficientNetB0 | — | 0.8591 | 0.5586 |
| 04 | R+M+E | concat | 0.8567 | 0.5630 |
| 05 | R+M | concat | 0.8247 | 0.5438 |
| 06 | R+E | concat | 0.8388 | 0.5464 |
| 07 | M+E | concat | **0.8728** | **0.5758** |
| 08 | R+M+E | concat | 0.8567 | 0.5630 |
| 09 | R+M+E | weighted | 0.8549 | 0.5609 |

### ✅ Step 3a — Implement generate_report_tables.py
*Done 2026-05-24 (ahead of original Step 10 placement). Produces
`results/tables/ablation_table.csv` and `ablation_table.md` from all
`results/runs/*/metrics.json` files.*

### ✅ Step 3b — Implement plot_results.py
*Done 2026-05-24 (unplanned, added to address observability gap). Produces:*
- *`results/figures/comparison_bar_chart.png` — accuracy + macro F1 across all experiments.*
- *`results/figures/training_curves.png` — val accuracy and macro F1 per epoch, all runs.*

### ✅ Step 3c — Create docs/results_progress.md
*Done 2026-05-24 (unplanned, added to address missing weekly ground-truth tracker).
Populated with Week 1 milestones + results and Week 2 frozen results + key findings.*

### ✅ Step 3d — Complete observability: confusion matrix + per-class chart

Extend `src/evaluation/visualization.py` with two functions:

- `plot_confusion_matrix(preds, labels, class_names, output_path)` — normalised
  confusion matrix heatmap saved as PNG.
- `plot_per_class_f1(per_class_metrics, class_names, output_path)` — horizontal bar
  chart of per-class F1 scores sorted descending; highlights classes with F1 < 0.3.

Extend `scripts/plot_results.py` with a `--confusion-matrix` flag that:
- Loads `predictions.npz` from each run dir.
- Calls `plot_confusion_matrix()` and saves to `results/runs/{id}/confusion_matrix.png`.
- For the best experiment by macro F1, also calls `plot_per_class_f1()` and saves to
  `results/figures/per_class_f1_best.png`.

Run for all currently completed experiments (01–09):

```bash
uv run python scripts/plot_results.py --confusion-matrix
```

After this step, update `docs/results_progress.md` with:
- A "Figures" subsection listing available plots and their paths.
- Per-class breakdown note in the Week 2 "Key findings" section: the 3–5 classes
  with the lowest F1 scores (from the per-class chart of the best frozen experiment),
  with a brief note on whether low F1 correlates with low support or class ambiguity.

### ✅ Step 4 — Implement augmentation (src/data/augmentation.py)
*Done 2026-05-24. `apply_rand_augment` wraps `torchvision.transforms.RandAugment`.
`apply_cutmix` samples λ ~ Beta(α,α), clips box to image boundary, recomputes lam
from actual clipped area. Returns (mixed, label_a, label_b, lam). 18/18 tests pass,
full suite 50 passed.*


### ✅ Step 5 — Implement EMA (src/training/ema.py)
*Done 2026-05-24. `EMA` class with `__init__(model, decay, start_epoch)`,
`update(model, epoch)` (no-op when epoch < start_epoch), `apply(model)`
(stores live-weight backup before swapping shadow in), `restore(model)`
(restores backup; raises if called without prior apply). Shadow weights
stored as `dict[str, Tensor]` via `named_parameters()` — BN buffers excluded.
15/15 tests pass, full suite 65 passed.*

### ✅ Step 6 — Complete build_adamw_with_llrd (src/training/optimizers.py)
*Done 2026-05-24. Per-block LLRD groups implemented for all three backbones using
verified indices from project_structure.md §2.1–2.2. Empty groups skipped. Head
params (projections, fusion, classifier) collected via id-tracking of backbone params.
26/26 tests pass (no-double-count, full-coverage, no-empty-groups, group-count-by-unfreeze,
deepest-block-LR, monotone-decay, head-LR, triple-backbone). Full suite: 91 passed.*

### ✅ Step 7 — Add fine-tuning code path to scripts/train.py
*Done 2026-05-24. Branches in main() on unfreeze_blocks > 0. Fine-tune path:
MultiCNNFusionClassifier, _make_image_loaders (Resize→RandomCrop/CenterCrop→
RandAugment→ToTensor→Normalize), WeightedRandomSampler, build_adamw_with_llrd
when backbone_lr set, EMA gated on ema.enabled (apply/restore guarded against
pre-start_epoch use), CutMix in _compute_loss helper (gated on cutmix_prob > 0).
best.pt = EMA weights, last.pt = live weights, ema.pt = shadow dict. Frozen path
untouched. 13/13 trainer smoke tests pass, full suite 104 passed.*

### ✅ Step 8 — Add fine-tune experiment entries to experiment_matrix.yaml
*Done 2026-05-24. Experiments 10, 11, 12 added under "Stage 1: Fine-tune wide" comment.
All 9 config paths verified to exist. YAML parses cleanly (16 total entries). 104 tests pass.*


### ✅ Step 9 — Run fine-tune experiments + update observability
*Done 2026-05-24–25. All three fine-tune experiments completed. No VRAM issues (RTX 5080
handled batch_size=64 with 3 unfrozen backbones). Two EMA bugs found and fixed during
this step: (1) shadow tensors initialised on CPU but model moved to CUDA inside Trainer —
fixed with lazy device migration in `EMA.update()`; (2) shadow held random-init weights
at first activation (decay=0.999 over 5 epochs → <0.5% signal) — fixed with `reset_shadow()`
warm-start called once in `fit()` at the epoch EMA first becomes active.*

*Results (test set, fold 0):*

| ID | Method | Fusion | Mode | Acc | Macro F1 |
|---|---|---|---|---:|---:|
| 10 | R+M+E | concat | finetune-3blk | **0.8784** | **0.5796** |
| 11 | R+M+E | weighted | finetune-3blk | 0.8685 | 0.5751 |
| 12 | R | — | finetune-3blk | 0.8501 | 0.5748 |

*`generate_report_tables.py` → 12 rows written. `plot_results.py --confusion-matrix` →
confusion matrices for all 12 experiments + per-class F1 chart for best (exp 10).
`docs/results_progress.md` fine-tune section populated — acceptance gate passed.*

### ⬜ Step 10 — Final test suite

```bash
uv run pytest tests/ -v
```

All tests must pass before marking this plan complete.

### ⬜ Step 11 — Final documentation

- Append dated entry to `docs/experiment_log.md` covering all of Week 2.
- Update `docs/known_issues.md` for any unresolved issue from fine-tuning.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| ~~FrozenHeadModel wrong shape for pair configs~~ | ✅ Resolved — test_frozen_head.py |
| ~~Feature cache build time for M/E backbones~~ | ✅ Resolved — caches built |
| Triple fine-tune exceeds 16 GB VRAM | Reduce batch_size to 32; document |
| CutMix loss formula wrong | Test: mixed loss must lie between label_a and label_b pure losses |
| EMA apply/restore order corrupts checkpoint | Test: weights after restore must be byte-identical to pre-apply |
| LLRD leaves some parameters ungrouped | Test: every requires_grad param in exactly one group |
| Fine-tune macro F1 worse than frozen on fold 0 | Expected; 5-fold CV in Week 3 averages out fold variance |
| Observability step skipped under time pressure | Step 3d and post-Step 9 plot update are acceptance gates — not optional |
| Agent adds a non-MLP classifier or changes PLD-* | Forbidden per `project_structure.md §9`; stop-and-ask |

---

## 8. Acceptance Criteria

**Track A ✅**
- `configs/experiment_matrix.yaml` contains entries 01–09.
- `tests/test_frozen_head.py` covers single/pair/triple inputs for concat and weighted.
- 32 tests pass.
- Experiments 02–09 complete; every `metrics.json` has no NaN values.

**Track C — Observability**
- `results/tables/ablation_table.md` exists and contains one row per completed experiment.
- `results/figures/comparison_bar_chart.png` and `training_curves.png` exist and reflect
  all completed experiments.
- `results/figures/` contains a `confusion_matrix.png` per completed experiment (after Step 3d).
- `results/figures/per_class_f1_best.png` exists for the highest macro-F1 experiment.
- `docs/results_progress.md` is updated after frozen batch and after fine-tune batch.

**Track B**
- `configs/experiment_matrix.yaml` contains entries 10–12.
- `test_augmentation.py` and `test_ema.py` pass.
- LLRD test: every requires_grad parameter in exactly one group.
- Experiments 10–12 complete with no NaN values.
- `docs/results_progress.md` fine-tune section populated.
- `uv run pytest tests/` passes (no regressions).
- No `.pt`, feature cache, or raw data file committed to git.
- No PLD-* decision changed without a `docs/decisions.md` entry.

---

## 9. Commands to Run

```bash
# --- Track A (DONE) ---
# uv run pytest tests/ -v                                          # 32 passed
# uv run python scripts/train.py ... --experiment 02_...          # done 02-09

# --- Track C: Step 3d (NEXT) ---
uv run python scripts/plot_results.py --confusion-matrix

# --- Track B: Steps 4-9 ---
# (implement augmentation, EMA, LLRD, train.py fine-tune path first)
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 10_triple_concat_finetune_wide_official
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 11_triple_weighted_finetune_wide_official
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 12_single_resnet50_finetune_wide_official

# After experiments 10-12:
uv run python scripts/generate_report_tables.py
uv run python scripts/plot_results.py --confusion-matrix

# --- Step 10: Final test suite ---
uv run pytest tests/ -v
```

---

## 10. Documentation Updates Required

**After Step 3d (observability completion):**
- Update `docs/results_progress.md`: add "Figures" subsection listing available plots.

**After Steps 4–7 (new code):**
- Append to `docs/experiment_log.md` noting new modules.

**After Step 9 (fine-tune batch) — mandatory gate:**
- Update `docs/results_progress.md`: populate fine-tune results table and key findings.
- Append dated entry to `docs/experiment_log.md`.

**After Step 10 (final test suite):**
- Append to `docs/experiment_log.md` with final test count and pass status.

**Any time a new failure or workaround appears:**
- Append to `docs/known_issues.md`.

**Only if a locked decision or split policy changes:**
- Append to `docs/decisions.md`.
