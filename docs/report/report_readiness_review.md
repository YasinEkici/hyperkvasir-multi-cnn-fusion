# Report Readiness Review — Multi-CNN Feature Fusion (HyperKvasir 23-class)

> Inspection-only review (no training, no model changes, no LaTeX yet).
> Grounded in: `AGENTS.md`, `project_plan.md`, `docs/school_assignment.md`,
> `docs/decisions.md` (D-06…D-09), `docs/experiment_log.md`,
> `docs/results_progress.md`, `references/INDEX.md`,
> `docs/report/report_comparison_framework.md`,
> `docs/report/AGENT_REPORT_RULES.md`, and the contents of `results/tables/`,
> `results/figures/`, `results/runs/`.
>
> **Report language: Turkish** (mandated by `AGENT_REPORT_RULES.md` §intro).
> Every number below traces to a `metrics.json` / `*.json` / `.png` artifact —
> none are invented.

---

## 1. Report sections we should write

Merged from assignment §5, `project_plan.md` §10, and the comparison framework §6:

1. **Giriş (Introduction)** — task, dataset challenge, why feature fusion, the three
   comparison axes, MLP-only statement + the "3-classifier was an instructor-confirmed
   error" sentence.
2. **Yöntem (Method)** — R/M/E backbones, feature extraction, projection (512, LN+GELU),
   concat + weighted fusion (verbatim equations), GMU (advanced), MLP, frozen vs.
   fine-tune (last 3 blocks, BN frozen). Focal loss **only if results land**.
3. **Deneyler (Experiments / Setup)** — dataset (10,662 images, 23 classes, imbalanced),
   official 5-fold protocol, hyperparameters (`finetune_wide.yaml`), metrics, reproducibility.
4. **Sonuçlar (Results)** — single / pair / triple, concat vs. weighted, frozen vs.
   fine-tune, 5-fold mean±std, bootstrap CI, per-class table, confusion matrix; a
   *separate* contextual literature subsection.
5. **Tartışma (Discussion — most critical, 25%)** — which backbone, whether fusion helped,
   best combination, strengths/weaknesses, rare-class failure, transfer-learning verdict +
   timing, honest literature positioning.
6. **Sonuç (Conclusion)** — summary, limitations, future work.

---

## 2. Sections we can write **now** (evidence is complete)

| Section | Why ready |
|---|---|
| Introduction | All framing is decided (PLDs, assignment, decisions). No results dependency. |
| Method | All implemented methods are final: backbones, projection, concat/weighted/GMU, MLP, frozen/fine-tune. Equations exist in `src/` docstrings + reference files. |
| Experiments / Setup | Protocol (official 5-fold), dataset counts (10,662 / 23), hyperparameters (`finetune_wide.yaml`), provenance (D-09) all documented. |
| Results — internal | Stage-1 ablation, 5-fold CV mean±std, bootstrap CI, extra metrics (micro-F1/MCC/weighted-F1) all on disk (see §4). |
| Discussion — internal | Backbone/fusion/transfer findings + rare-class analysis fully supported by logs and figures. |
| Conclusion | Derivable from the above; no new results needed. |

**Bottom line:** the *mandatory* project (concat + weighted + GMU, 5-fold CV, CI) is
**fully reportable today**. Nothing core is blocked.

---

## 3. Sections / items that must **wait** (or be omitted)

| Item | Status | Action |
|---|---|---|
| Focal loss (exp 16) results (`Table 5: CE vs focal`) | **No run artifacts exist** locally (`results/runs/16_*` is empty); framework doc says "running — placeholder" | Mark `TODO` / omit. **Do not report focal numbers.** |
| TTA, seed ensemble, recipe-v2 | Not started; explicitly out of scope this phase | Omit; may appear only as "future work". |
| Grad-CAM++ figure | Not produced (stretch, Week 4) | Omit or mark optional. |
| UMAP / silhouette | Not produced (stretch) | Omit. |
| Cross-protocol reproduction (Effimix/GastroViT splits) | Not reproduced | Keep **contextual-only**, never head-to-head (PLD-17). |

---

## 4. Existing result tables and figures (available now)

**Tables (`results/tables/`):**
- `ablation_table.csv` / `.md` — 35 rows (15 experiments incl. all 5 folds; Acc, macro F1/P/R, stop epoch). → **Table 1 + raw for Table 2**.
- `cv_{10,11,13,14,15}_*_summary.json` — 5-fold mean±std per config. → **Table 2**.
- `ci_{10,11,13,14,15}_*.json` — bootstrap 95% CI (pooled n=10,662). → **Table 2b**.
- `extra_metrics_{10,11,13,14,15}_*.json` — micro-F1, macro P/R/F1, weighted-F1, **MCC** (pooled). *(untracked / new)* → supporting metrics column.

**Figures (`results/figures/`):**
- `comparison_bar_chart.png` — Acc + macro F1 across experiments.
- `training_curves.png` — val Acc/F1 per epoch.
- `per_class_f1_best.png` — per-class F1 (note: generated for exp 11 **fold-1**, F1=0.6014, the best single fold — confirm which view we want for the report).

**Confusion matrices (`results/runs/{id}/confusion_matrix.png`):** present for **all** runs
01–15 including every fold of the 5 CV configs — so the best-model (exp 11) confusion matrix is available.

**Per-run raw artifacts:** `metrics.json`, `predictions.npz`, `epoch_log.jsonl`, `config.yaml`
present for all CV runs → per-class metrics and training-time are **derivable without retraining**.

---

## 5. Important artifacts that are **missing** (and how to obtain — no retraining needed)

| Missing artifact | Needed for | How to produce (no training) |
|---|---|---|
| **Per-class P/R/F1 + support table** for best model (exp 11) | Table 3 (§5.5 core) | Derive from existing `predictions.npz` via a metrics script. **Not yet a table file.** |
| **Training-time / epochs table** (frozen vs. fine-tune) | Table 4 (§5.5 "how long it took") | Aggregate from existing `epoch_log.jsonl` per run. Confirm wall-clock fields are logged. |
| **Architecture diagram** | Method figure | Author manually (e.g. draw.io/TikZ); not a results artifact. |
| **Reference `paper.md` verification** | Every citation (AGENTS.md §"Working with references") | Confirm each cited file (Borgli 2020, Effimix 2022, GastroViT 2025, Ahmed 2023, GMU, focal, etc.) actually exists under `references/` before citing. Do **not** assume. |
| **Focal (exp 16) results** | Table 5 | Out of scope now — `TODO`. |

---

## 6. Claims that are **safe** to make (traceable)

- **Triple weighted fine-tune (exp 11) is the best configuration**: CV macro-F1
  **0.5892 ± 0.0102**, pooled F1 **0.6000**, 95% CI **[0.5814, 0.6206]**.
- **Weighted is statistically separable from concat** on the pooled set: exp 11 CI lower
  bound (0.5814) > exp 10 CI upper bound (0.5799), non-overlapping.
- **Fold-0 ranking was misleading**: fold-0 had concat (0.5796) > weighted (0.5751); 5-fold
  CV reverses this — must be stated as a methodological lesson.
- **GMU is implemented (verbatim Arevalo equations) but gives no CV gain over concat**
  (Δ ≈ 0.002, fully overlapping CIs) — report as an honest neutral/null result.
- **EfficientNetB0 is the best single frozen backbone** (F1 0.5586); **M+E is the best frozen
  pair** (F1 0.5758).
- **Fine-tuning helps on aggregate** (CV mean) even though it hurt single-E / pair-M+E on
  fold-0 only — report the fold-0 anomaly honestly.
- **Accuracy ≫ macro-F1 gap is real and expected** (e.g. exp 11: acc 0.8706 vs macro-F1 0.60;
  weighted-F1 0.8716, MCC 0.8599) → imbalance evidence, consistent with Borgli's macro≈0.62 anchor.
- **Rare classes collapse (F1=0) due to support ≤ 10**, not random error (confusion matrix shows
  semantically-adjacent absorption).
- **Official 5-fold + bootstrap CI is methodologically stronger** than single-split baselines.

All of the above trace to `docs/results_progress.md`, `docs/experiment_log.md`,
`results/tables/*`, and `results/runs/*` artifacts.

---

## 7. Claims that must **not** be made (per `AGENT_REPORT_RULES.md` §7 + constraints)

- ❌ State-of-the-art / best-on-HyperKvasir.
- ❌ Direct superiority over Effimix (0.97) / GastroViT (0.64) — **different protocols**; contextual only.
- ❌ Any focal / TTA / seed-ensemble / recipe-v2 result (not produced / out of scope).
- ❌ Accuracy alone proving success.
- ❌ Rare classes "solved" or model "clinically usable".
- ❌ Treating protocol-mismatched literature values as direct baselines.
- ❌ Claiming GMU improved fusion (it did not, over CV).
- ❌ Any number not traceable to a repository artifact — use `TODO` instead.

---

## 8. Recommended next step — LaTeX report skeleton

**Do not generate it yet** — propose first, then build on approval. Recommended plan:

1. **One LaTeX skeleton file** (e.g. `docs/report/main.tex` + a `references.bib`), Turkish,
   with the six sections from §1 as empty `\section{}` stubs and `% TODO` markers where
   results plug in. No prose yet beyond section scaffolding.
2. **Table/figure placeholders** wired to the artifacts in §4 (Table 1 ablation, Table 2 CV,
   Table 2b CI, Table 3 per-class [pending §5], Table 4 timing [pending §5], confusion-matrix
   figure, comparison bar chart, training curves).
3. **Before any prose**, produce the two missing *derived* tables (per-class, training-time)
   from existing `predictions.npz` / `epoch_log.jsonl` — **no retraining**, with a small read-only
   script — so Results/Discussion can be written against real numbers.
4. **Verify reference files** exist under `references/` and build `references.bib` only from
   confirmed entries.

**Decisions confirmed (2026-06-07):**
- **Per-class table (Table 3): pooled across all 5 folds** of exp 11 (n=10,662) — more
  representative for the official 5-fold protocol, avoids noisy single-fold rare-class reads.
- **Derived tables (per-class, training-time): do NOT generate yet.** This step is review-only;
  they are documented as missing (§5) and will be produced later under a separate, explicit
  read-only prompt *after* the skeleton/source map exists. No training, no model changes,
  no experimental outputs in this phase.

**Still to confirm before scaffolding LaTeX:**
- Output format/template: standard LaTeX `article` + `references.bib`, or a specific instructor
  `.cls`/template you will provide?

---

### Summary

The **mandatory project is report-ready**: all required experiments (single/pair/triple,
concat + weighted, GMU, 5-fold CV, bootstrap CI) are complete with traceable artifacts.
Only **derived** tables (per-class, training-time) and a hand-drawn architecture diagram are
missing — none require retraining. **Focal/TTA/ensemble remain out and must be marked `TODO`
or omitted.** Next step is a Turkish LaTeX skeleton + two read-only derived tables, pending
your answers above.
