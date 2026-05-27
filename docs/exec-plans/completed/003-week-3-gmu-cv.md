# 003 — Week 3: GMU Implementation and 5-Fold Cross-Validation

> **Status legend:** ✅ Done · ⬜ Pending
>
> All steps begin at ⬜ Pending.

---

## 1. Goal

Complete Stage 2 of `project_plan.md` by implementing the GMU fusion module and
running selected 5-fold cross-validation on the best configurations from Stage 1.
Provide statistical rigor via bootstrap 95% CI.

By the end of this plan the project must have:

- `src/models/fusion/gmu.py` fully implemented with verbatim paper equations in
  docstring (AGENTS.md requirement).
- Fold-0 supplementary fine-tune runs for single EfficientNetB0 and pair M+E,
  establishing the true "best mode" for those configs before CV.
- 5-fold CV results for 5 selected configurations, reported as mean ± std.
- Bootstrap 95% CI for headline results (at minimum exp 10 and exp 15).
- `docs/results_progress.md` Week 3 section fully populated (hard acceptance gate).
- `uv run pytest tests/` passing throughout.

This satisfies the Week 3 hard checkpoint from `project_plan.md`:
*"Selected 5-fold results are ready; GMU is either complete or dropped."*

---

## 2. Inputs

- `AGENTS.md`
- `project_structure.md` — §2.1–2.2 (backbone verified facts), §6 (module contracts)
- `project_plan.md` — §3.3 Stage 2 (selected 5-fold CV minimum set), §3.2 Stage 1
- `docs/decisions.md` — official split protocol
- `docs/results_progress.md` — Week 2 ground-truth results
- `docs/experiment_log.md` — narrative run log
- `docs/known_issues.md` — KI-002 (imbalance), KI-003 (EMA bugs — resolved)
- `configs/experiment_matrix.yaml` — entries 01–12 present; 13–15 to be added
- `configs/training/finetune_wide.yaml` — training config reused for all fine-tune runs
- `configs/method/triple_concat.yaml`, `triple_weighted.yaml`, `pair_m_e_concat.yaml`,
  `single_efficientnetb0.yaml` — existing method configs
- `src/models/fusion/gmu.py` — placeholder (raises NotImplementedError)
- `src/evaluation/statistical.py` — placeholder (raises NotImplementedError)
- `references/methodology_fusion/gmu_arevalo_2017_iclr_workshop/paper.md` — §3.1 equations
- Feature caches: `results/feature_cache/fold_0_{resnet50,mobilenetv2,efficientnetb0}_features.pt`
- Fold manifests: `data/splits/hyperkvasir_official_5fold/fold_{0-4}.csv`
- Week 2 run outputs: `results/runs/{01-12}/metrics.json`, `predictions.npz`

---

## 3. Scope

### Track A — GMU Implementation
- Implement `src/models/fusion/gmu.py` with softmax-gate N-branch generalization of
  the bimodal GMU equations from Arevalo et al. (2017) §3.1.
- Wire into `src/models/full_model.py` (`MultiCNNFusionClassifier`) and
  `scripts/train.py` (`FrozenHeadModel`).
- Add `configs/method/triple_gmu.yaml`.
- Write `tests/test_gmu.py`.
- Add experiment 15 to matrix; run GMU fold-0 smoke test.

### Track B — Missing Stage 1 Fine-Tune Runs (fold 0)
- Add and run `13_single_efficientnetb0_finetune_wide_official` (fold 0).
- Add and run `14_pair_m_e_finetune_wide_official` (fold 0).
- Rationale (D-01): fine-tune consistently outperformed frozen in Week 2 (+0.016 F1 for
  single R50); running fine-tune for these two configs establishes a defensible
  "best mode" claim for both in the CV and in the report.

### Track C — CV Infrastructure
- Add `--fold` CLI override to `scripts/train.py`; fold-indexed run dir naming.
- Implement `src/evaluation/statistical.py::bootstrap_ci` (module contract in
  `project_structure.md §6`).
- Create `scripts/run_cv.py` — loops over folds for a given experiment ID.
- Create `scripts/aggregate_cv.py` — computes mean ± std from per-fold metrics.
- Create `scripts/compute_ci.py` — concatenates fold predictions, calls bootstrap_ci.
- Write `tests/test_statistical.py`.

### Track D — 5-Fold CV Runs (20 new runs)
Fold 0 already exists for exps 10, 11 (Week 2) and will be produced in Tracks A–B
for exps 13, 14, 15. Track D runs folds 1–4 for all 5 selected configs.

Selected configs (project_plan.md §3.3 minimum selected set):

| Exp ID | Config | Fold-0 F1 (known/expected) | Source |
|---|---|---:|---|
| 13 | Single EfficientNetB0 fine-tune | ~0.58–0.60 | Track B |
| 14 | Pair M+E fine-tune | ~0.59–0.62 | Track B |
| 10 | Triple concat fine-tune | 0.5796 | Week 2 |
| 11 | Triple weighted fine-tune | 0.5751 | Week 2 |
| 15 | Triple GMU fine-tune | ? | Track A |

### Track E — Bootstrap CI
- Implement `scripts/compute_ci.py`.
- Run CI for exps 10 (triple concat — current best) and 15 (GMU — primary new method).
- Use concatenated predictions across all 5 folds (D-05 decision: larger N → narrower CI).

### Track F — Documentation
- `docs/results_progress.md` Week 3 section after each track.
- `docs/experiment_log.md` dated entries after each step group.
- `docs/known_issues.md` if new issue found.

---

## 4. Out of Scope

- AFF and LMF fusion implementations (`project_plan.md §3.4` — stretch).
- McNemar test (`project_structure.md §6` — signature present, not required this week).
- Grad-CAM++ visualization (Week 4).
- DeepWeeds or any secondary dataset.
- Cross-protocol reproduction (Effimix, GastroViT).
- Report writing (Week 4).
- Running experiments on folds beyond 0–4.
- Non-MLP classifiers (PLD-06).
- Changing any locked design decision (PLD-*).
- UMAP / silhouette score (stretch).
- Updating BN running statistics during fine-tuning (ablation not planned).

---

## 5. Files Expected to Be Modified

### Track A
| File | Change |
|---|---|
| `src/models/fusion/gmu.py` | Implement (replace placeholder) |
| `src/models/full_model.py` | Add `elif fusion_type == "gmu"` branch |
| `scripts/train.py` | Add GMU branch in FrozenHeadModel; `--fold` override (shared with Track C) |
| `configs/method/triple_gmu.yaml` | New |
| `configs/experiment_matrix.yaml` | Add entries 13, 14, 15 |
| `tests/test_gmu.py` | New |

### Track C
| File | Change |
|---|---|
| `scripts/train.py` | `--fold` CLI override + fold-indexed run_dir |
| `src/evaluation/statistical.py` | Implement `bootstrap_ci` |
| `scripts/run_cv.py` | New |
| `scripts/aggregate_cv.py` | New |
| `scripts/compute_ci.py` | New |
| `tests/test_statistical.py` | New |

### Documentation (all tracks)
| File | When |
|---|---|
| `docs/results_progress.md` | After each experiment batch (mandatory acceptance gate) |
| `docs/experiment_log.md` | After each step group |
| `docs/known_issues.md` | Only if new issue found |
| `docs/decisions.md` | Only if a locked decision or split policy changes |

---

## 6. Step-by-Step Implementation Plan

### ✅ Step 1 — Implement GMU fusion module

Read `references/methodology_fusion/gmu_arevalo_2017_iclr_workshop/paper.md` §3.1.
Copy bimodal equations verbatim into the module docstring (AGENTS.md requirement).

**Bimodal paper equations (§3.1, verbatim):**
```
h_v = tanh(W_v · x_v)
h_t = tanh(W_t · x_t)
z   = σ(W_z · [x_v, x_t])
h   = z * h_v + (1−z) * h_t
Θ   = {W_v, W_t, W_z}
```

**N-branch softmax generalization (approved D-03, D-04):**
```
h_i = tanh(W_i · x_i)                              for i = 1..N
z   = softmax(W_z · concat([x_1, ..., x_N]))       → (B, N)
h   = Σ_i  z_i * h_i                               → (B, feature_dim)
```

Implementation contract:
- `W_i`: `nn.Linear(feature_dim, feature_dim, bias=False)` per branch, stored in
  `nn.ModuleList`. Output `h_i` shape: (B, D).
- `W_z`: `nn.Linear(N * feature_dim, N, bias=True)`. Gate logits shape: (B, N).
- `z = F.softmax(self.gate(torch.cat(features, dim=-1)), dim=-1)` → (B, N).
- `h = (z.unsqueeze(-1) * torch.stack(h_list, dim=1)).sum(dim=1)` → (B, D).
- `output_dim` property returns `self.feature_dim` (D-04 decision; matches paper).
- Interface matches `concat.py` and `weighted.py`:
  `__init__(num_branches, feature_dim, **kwargs)` / `forward(list[Tensor]) -> Tensor`.

Wire into both fusion-type switches:
- `src/models/full_model.py` — in `MultiCNNFusionClassifier.__init__()`.
- `scripts/train.py` — in `FrozenHeadModel.__init__()`.

Add `configs/method/triple_gmu.yaml`:
```yaml
backbone_names: [resnet50, mobilenetv2, efficientnetb0]
projection_dim: 512
fusion_type: gmu
mlp_hidden: [256]
dropout: 0.3
```

### ✅ Step 2 — Write GMU tests and verify full suite

`tests/test_gmu.py` — required tests:
- Output shape (B, feature_dim) for N=1, 2, 3 branches.
- `output_dim` property equals `feature_dim`.
- Gate vector z sums to 1 along branch dimension (softmax invariant).
- No parameter overlap between branch transforms (`W_i`) and gate (`W_z`).
- `forward` does not crash with B=1 (single-sample batch).
- Gradient flows: all parameters have non-zero grad after backward pass.
- `FrozenHeadModel` and `MultiCNNFusionClassifier` instantiate cleanly with
  `fusion_type="gmu"`.

```bash
uv run pytest tests/test_gmu.py -v
uv run pytest tests/ -v          # full suite — no regressions
```

### ✅ Step 3 — Add GMU experiment entry and run fold 0 smoke test

Add to `configs/experiment_matrix.yaml`:
```yaml
# --- Stage 2: Week 3 additions ---
- id: "15_triple_gmu_finetune_wide_official"
  dataset: "configs/dataset/hyperkvasir_23class_official.yaml"
  method:  "configs/method/triple_gmu.yaml"
  training: "configs/training/finetune_wide.yaml"
  fold: 0
```

Run:
```bash
uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 15_triple_gmu_finetune_wide_official
```

**Acceptance gate:**
- `results/runs/15_.../metrics.json` exists with no NaN values.
- Zero-support class count is zero.
- Result documented alongside exp 10 (triple concat fine-tune, F1=0.5796).

**Stop-and-diagnose threshold:** If GMU fold-0 macro F1 < 0.50 (catastrophic collapse),
stop before CV. Likely causes: gate saturation (check z entropy), missing `reset_shadow()`
call for GMU params (EMA contract), learning rate mismatch. Fix before proceeding.

Do NOT use test-set result to retune GMU hyperparameters.

### ✅ Step 4 — Missing Stage 1 fine-tune runs (fold 0)

Add to `configs/experiment_matrix.yaml`:
```yaml
- id: "13_single_efficientnetb0_finetune_wide_official"
  dataset: "configs/dataset/hyperkvasir_23class_official.yaml"
  method:  "configs/method/single_efficientnetb0.yaml"
  training: "configs/training/finetune_wide.yaml"
  fold: 0

- id: "14_pair_m_e_finetune_wide_official"
  dataset: "configs/dataset/hyperkvasir_23class_official.yaml"
  method:  "configs/method/pair_m_e_concat.yaml"
  training: "configs/training/finetune_wide.yaml"
  fold: 0
```

Run:
```bash
uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 13_single_efficientnetb0_finetune_wide_official

uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 14_pair_m_e_finetune_wide_official
```

After both complete:
```bash
uv run python scripts/generate_report_tables.py   # ablation table now 15 rows
uv run python scripts/plot_results.py --confusion-matrix  # refresh all figures incl. exps 13, 14, 15
```

Update `docs/results_progress.md` with fold-0 results for experiments 13 and 14.

### ✅ Step 5 — CV infrastructure

**5a — `--fold` CLI override in `scripts/train.py`**

Add optional `--fold` argument to `argparse`. When provided:
- Overrides `exp["fold"]` from the YAML config.
- Appends `_fold_{k}` to `run_dir` so fold-specific results do not overwrite
  the canonical fold-0 run directory.

Example: `--experiment 10_triple_concat_finetune_wide_official --fold 1`
→ `results/runs/10_triple_concat_finetune_wide_official_fold_1/`

**5b — `scripts/run_cv.py`**

CLI: `--experiment {id} --folds 0 1 2 3 4 [--config path]`

For each fold: calls `scripts/train.py` as a subprocess with `--fold k`.
Logs per-fold completion status. If one fold fails, continues remaining folds
and reports which fold(s) failed at end (no silent swallowing of errors).

**5c — `scripts/aggregate_cv.py`**

CLI: `--experiment {id} --folds 0 1 2 3 4`

Reads `metrics.json` from:
- `results/runs/{id}/` for fold 0 (canonical path, backward compatible).
- `results/runs/{id}_fold_{k}/` for k ≥ 1.

Computes mean ± std for accuracy, macro_f1, macro_precision, macro_recall.
Writes to `results/tables/cv_{id}_summary.json`.

**5d — `src/evaluation/statistical.py` — implement `bootstrap_ci`**

```python
def bootstrap_ci(
    metric_fn: Callable,
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high).

    Resamples (preds, labels) with replacement n_bootstrap times.
    point_estimate = metric_fn(preds, labels) on original data.
    ci_low, ci_high = percentiles ((1-ci)/2, 1-(1-ci)/2) of bootstrap distribution.
    """
```

`tests/test_statistical.py` — required tests:
- CI bounds bracket the point estimate: `ci_low ≤ point ≤ ci_high`.
- CI width > 0.
- Deterministic: same result with same seed.
- Works with macro_f1 as `metric_fn` on synthetic preds/labels.
- `n_bootstrap` samples are drawn (shape check).

**5e — `scripts/compute_ci.py`**

CLI: `--experiment {id} --folds 0 1 2 3 4 [--n-bootstrap 1000]`

Concatenates `predictions.npz` (preds + labels arrays) from all specified fold
run directories. Calls `statistical.bootstrap_ci` with macro_f1 as `metric_fn`.
Prints and saves `results/tables/ci_{id}.json`:
```json
{"point_estimate": 0.5796, "ci_low": 0.558, "ci_high": 0.601, "n_bootstrap": 1000, "ci": 0.95}
```

```bash
uv run pytest tests/test_statistical.py -v
uv run pytest tests/ -v          # full suite after Track C
```

### ✅ Step 6 — 5-fold CV runs (20 new runs)

Fold 0 is already complete for each config (from Steps 3–4 and Week 2).
Track D runs folds 1–4 for all 5 selected configs.

```bash
uv run python scripts/run_cv.py \
  --experiment 13_single_efficientnetb0_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 14_pair_m_e_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 10_triple_concat_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 15_triple_gmu_finetune_wide_official --folds 1 2 3 4
```

**Estimated GPU time:** 20 runs × ~20 min = ~7 hours.
**Compute strategy:** Run overnight locally (RTX 5080 — 16 GB VRAM sufficient, verified
in Week 2). Switch to Colab only if A100 is confirmed; do not use if T4 assigned
(see memory: `project_compute_strategy.md`).

After all runs complete:
```bash
uv run python scripts/aggregate_cv.py --experiment 13_single_efficientnetb0_finetune_wide_official
uv run python scripts/aggregate_cv.py --experiment 14_pair_m_e_finetune_wide_official
uv run python scripts/aggregate_cv.py --experiment 10_triple_concat_finetune_wide_official
uv run python scripts/aggregate_cv.py --experiment 11_triple_weighted_finetune_wide_official
uv run python scripts/aggregate_cv.py --experiment 15_triple_gmu_finetune_wide_official
uv run python scripts/generate_report_tables.py
```

### ✅ Step 7 — Bootstrap CI

```bash
uv run python scripts/compute_ci.py \
  --experiment 10_triple_concat_finetune_wide_official --folds 0 1 2 3 4

uv run python scripts/compute_ci.py \
  --experiment 15_triple_gmu_finetune_wide_official --folds 0 1 2 3 4
```

CI results saved to `results/tables/ci_{id}.json`.
CI for all other configs optional but recommended for completeness.

### ✅ Step 8 — Final test suite

```bash
uv run pytest tests/ -v
```

All tests must pass before marking this plan complete.

### ✅ Step 9 — Final documentation

Populate `docs/results_progress.md` Week 3 section with:
- 5-fold CV results table (mean ± std for all 5 configs).
- Bootstrap 95% CI for exps 10 and 15.
- Key findings: does GMU outperform triple concat over 5 folds? Does fold-0 ranking
  hold across all folds? Does fine-tune M+E pair outperform frozen?
- GastroViT (F1=0.64) comparison with protocol note (their split ≠ official 5-fold).

Append dated entry to `docs/experiment_log.md` covering all of Week 3.
Update `docs/known_issues.md` if any new failure or workaround discovered.
Move this exec plan to `docs/exec-plans/completed/` when all steps are ✅.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GMU fold-0 F1 < 0.50 (gate collapse) | Low | Check z entropy during training; verify gradient flow through W_i and W_z; check EMA is tracking GMU params via named_parameters() |
| GMU no better than triple concat across all folds | Medium | Honest finding; report as is; Week 3 hard checkpoint only requires GMU to be "complete or dropped with reason" |
| EMA device mismatch on GMU gate params | Very low | EMA shadow covers all named_parameters() — GMU gate params included automatically; same fix as KI-003 already in place |
| Feature cache missing for folds 1–4 | None (fine-tune) | Fine-tune runs use image DataLoaders; no feature cache needed for CV runs |
| run_cv.py crashes mid-fold and loses partial results | Low | Per-fold subprocess isolation; completed folds already saved; aggregate_cv.py handles partial results |
| aggregate_cv.py path mismatch (fold-0 canonical vs fold-indexed dir) | Medium | Document convention in aggregate_cv.py; test with dry-run on existing Week 2 results before full CV |
| Bootstrap CI too wide (small per-fold test set ~2K) | Medium | Concatenate all 5 folds (~10K total) before bootstrap — D-05 decision already addresses this |
| Week 3 GPU time budget exceeded | Medium | CV runs unattended overnight; GMU + Track B runs are priority; bootstrap is CPU-only and fast |
| Non-MLP classifier or PLD-* change | Very low | Forbidden per project_structure.md §9; stop-and-ask per AGENTS.md |

---

## 8. Acceptance Criteria

**Track A — GMU**
- `src/models/fusion/gmu.py` docstring contains verbatim bimodal equations from
  Arevalo et al. (2017) §3.1.
- `tests/test_gmu.py` passes: output shape, gate sum = 1, `output_dim = feature_dim`,
  gradient flow, no param overlap, clean instantiation via full_model and FrozenHeadModel.
- `15_triple_gmu_finetune_wide_official` fold-0 `metrics.json` has no NaN values.
- GMU fold-0 result documented and compared against exp 10 (F1=0.5796).

**Track B — Missing Stage 1 runs**
- `13_single_efficientnetb0_finetune_wide_official` fold-0 complete, no NaN.
- `14_pair_m_e_finetune_wide_official` fold-0 complete, no NaN.
- Ablation table updated to 15 rows.

**Track C — Infrastructure**
- `--fold` override in `train.py`; fold-indexed `run_dir` naming correct.
- `run_cv.py` runs multiple folds; per-fold failure reported without aborting remaining.
- `aggregate_cv.py` produces `cv_{id}_summary.json` with mean ± std.
- `bootstrap_ci` in `statistical.py` passes all unit tests; CI bounds bracket
  point estimate for any valid metric_fn.

**Track D — CV runs**
- All 25 fold results present (5 configs × 5 folds, fold-0 from earlier steps).
- No NaN values in any fold's `metrics.json`.
- `aggregate_cv.py` summary JSON produced for all 5 configs.

**Track E — Bootstrap CI**
- CI computed and saved as `results/tables/ci_{id}.json` for at least exps 10 and 15.
- CI documented in `docs/results_progress.md` Week 3 section.

**All tracks**
- `uv run pytest tests/` passes after every step group (no regressions).
- No `.pt`, feature cache, or raw data file committed to git.
- No PLD-* decision changed without a `docs/decisions.md` entry.
- `docs/results_progress.md` Week 3 section fully populated (mandatory gate for Step 9).

---

## 9. Commands to Run

```bash
# ── Track A: GMU implementation ───────────────────────────────────────────────
uv run pytest tests/test_gmu.py -v
uv run pytest tests/ -v                              # after Step 2

# ── Track A: GMU fold-0 smoke test ────────────────────────────────────────────
uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 15_triple_gmu_finetune_wide_official

# ── Track B: Missing Stage 1 fine-tune runs ───────────────────────────────────
uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 13_single_efficientnetb0_finetune_wide_official

uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 14_pair_m_e_finetune_wide_official

uv run python scripts/generate_report_tables.py     # ablation table → 15 rows

# ── Track C: Infrastructure tests ─────────────────────────────────────────────
uv run pytest tests/test_statistical.py -v
uv run pytest tests/ -v

# ── Track D: 5-fold CV (20 new runs, folds 1–4 only) ─────────────────────────
uv run python scripts/run_cv.py \
  --experiment 13_single_efficientnetb0_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 14_pair_m_e_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 10_triple_concat_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official --folds 1 2 3 4

uv run python scripts/run_cv.py \
  --experiment 15_triple_gmu_finetune_wide_official --folds 1 2 3 4

# ── Track D: Aggregate CV results ─────────────────────────────────────────────
uv run python scripts/aggregate_cv.py \
  --experiment 13_single_efficientnetb0_finetune_wide_official
uv run python scripts/aggregate_cv.py \
  --experiment 14_pair_m_e_finetune_wide_official
uv run python scripts/aggregate_cv.py \
  --experiment 10_triple_concat_finetune_wide_official
uv run python scripts/aggregate_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official
uv run python scripts/aggregate_cv.py \
  --experiment 15_triple_gmu_finetune_wide_official

uv run python scripts/generate_report_tables.py

# ── Track E: Bootstrap CI ─────────────────────────────────────────────────────
uv run python scripts/compute_ci.py \
  --experiment 10_triple_concat_finetune_wide_official --folds 0 1 2 3 4

uv run python scripts/compute_ci.py \
  --experiment 15_triple_gmu_finetune_wide_official --folds 0 1 2 3 4

# ── Step 8: Final test suite ───────────────────────────────────────────────────
uv run pytest tests/ -v

# ── Optional: update figures ──────────────────────────────────────────────────
uv run python scripts/plot_results.py --training-curves
```

---

## 10. Documentation Updates Required

**After Step 3 (GMU fold-0 smoke test):**
- Append to `docs/experiment_log.md`: GMU implementation notes + fold-0 result vs exp 10.

**After Step 4 (exps 13, 14 fold-0):**
- Append to `docs/experiment_log.md`: Stage 1 supplementary fine-tune runs.
- Update `docs/results_progress.md`: add fold-0 rows for exps 13 and 14.

**After Track C (infrastructure complete):**
- Append to `docs/experiment_log.md`: new scripts summary (run_cv, aggregate_cv,
  compute_ci, bootstrap_ci).

**After Track D (CV runs complete) — mandatory acceptance gate:**
- Update `docs/results_progress.md` Week 3 section: 5-fold CV table (mean ± std).
- Append dated entry to `docs/experiment_log.md`.

**After Track E (CI):**
- Update `docs/results_progress.md` Week 3 section: bootstrap 95% CI for exps 10 and 15.

**After Step 9 (final documentation):**
- `docs/results_progress.md` Week 3 section fully closed.
- Append final entry to `docs/experiment_log.md` covering all of Week 3.
- Update `docs/known_issues.md` if any new failure or workaround found.
- Move this plan to `docs/exec-plans/completed/003-week-3-gmu-cv.md`.
