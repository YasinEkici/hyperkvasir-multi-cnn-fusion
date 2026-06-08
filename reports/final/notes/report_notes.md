# Report Notes — reports/final/

Working notes for the LaTeX report. The report is **Turkish** (mandated by
`docs/report/AGENT_REPORT_RULES.md`). This file is scratch space, not graded.

## Status: DRAFT IN PROGRESS (updated 2026-06-08)
Sections 01–03 drafted and **updated to the final state** (Week 3.5 Steps 4/5/7
done: TTA, seed ensemble, freeze). Section 06 (Conclusion) drafted. Sections
04 (Results) and 05 (Discussion) are still placeholders — written under a
separate prompt. The abstract is written **last**. No results invented; no SOTA
claims. focal / TTA / seed ensemble are now COMPLETED experiments (documented in
`docs/FINAL_MODEL.md` + `docs/decisions.md` D-06/D-07), not TODOs.

## Build
- Engine: pdfLaTeX (Overleaf default). Class: `article`.
- `\TODO{...}` renders red in the PDF and is grep-able.
- Skeleton compiles clean: all `\includegraphics` / table `\input` are commented
  out until assets are copied into `figures/` and `tables/`.

### Packages beyond the task's listed set (justified)
The task listed: graphicx, booktabs, tabularx, float, caption, subcaption,
amsmath, hyperref, geometry. Added because a **Turkish** report cannot compile/
render correctly without them:
- `inputenc` (utf8), `fontenc` (T1), `babel` (turkish) — encoding + hyphenation.
- `xcolor` — only to make the `\TODO` macro visible.
If you prefer the strict set, switch to XeLaTeX/LuaLaTeX + fontspec, or write the
report in English. Flag if you want this changed.

## Evidence map (source of truth → report slot)
| Report slot | Source artifact | Status |
|---|---|---|
| Table 1 (ablation) | `results/tables/ablation_table.md` | exists |
| Table 2 (CV mean±std) | `results/tables/cv_*_summary.json` | exists |
| Table 2b (bootstrap CI) | `results/tables/ci_*.json` | exists |
| Supporting metrics (micro-F1/weighted-F1/MCC), base/TTA/ensemble | `results/tables/extra_metrics_*{,_tta,_ensemble}.json` | exists (ensure git-tracked before submit) |
| Bootstrap CI incl. TTA/ensemble/focal | `results/tables/ci_*{,_tta,_ensemble}.json`, `ci_16_*` | exists |
| Table 3 (per-class P/R/F1, frozen exp 11 + TTA, 5-fold pooled) | `results/tables/per_class_frozen_tta.{md,csv}` | **PRODUCED** |
| Table 4 (training time/epochs) | `results/tables/training_time.{md,csv}` | **PRODUCED** (wall-clock APPROX) |
| Fig: comparison bar | `results/figures/comparison_bar_chart.png` | exists |
| Fig: training curves | `results/figures/training_curves.png` | exists |
| Fig: per-class F1 | `results/figures/per_class_f1_best.png` | exists |
| Fig: confusion matrix (frozen best + TTA) | `results/runs/11_*/confusion_matrix*.png` (+ TODO: TTA variant per exec-plan A1) | exists / verify TTA view |
| Fig: architecture diagram | — | **NOT produced (manual)** |

## Decisions carried from readiness review (2026-06-07)
- Per-class table (Table 3) = **exp 11, pooled across all 5 folds** (n=10,662).
- Derived tables (Table 3, Table 4) **not generated yet** — separate explicit
  read-only step after the skeleton.
- No instructor template exists → `article` class confirmed.

## Headline result (frozen final model — `docs/FINAL_MODEL.md`, traceable)
- **Final = exp 11 (R+M+E weighted fine-tune, CE) + TTA, seed 42.**
- Pooled (n=10,662): macro-F1 **0.6075**, 95% CI **[0.5860, 0.6296]**;
  acc/micro-F1 0.8765, weighted-F1 0.8761, MCC 0.8662, macro P/R 0.6162/0.6158.
- Week 3 CE champion (base exp 11) macro-F1 0.6000 → +TTA 0.6075 (within CI).
- Focal exp 16 macro-F1 0.5914, CI [0.5750, 0.6096]; seed ensemble = no macro-F1
  gain. No Week 3.5 technique beat the CE champion at the 95% CI level.

## Must-NOT (AGENT_REPORT_RULES.md §7)
- No SOTA / no direct superiority over Effimix (0.97) or GastroViT (0.92 acc /
  0.64 F1) — different protocols → contextual only; never mix GastroViT 16-class
  with our 23-class.
- focal/TTA/ensemble ARE done, but their gains are **within CI** — state honestly,
  do not overclaim significance.
- macro-F1 is headline; accuracy supporting only.

## Citation backing (verified 2026-06-08)
- GMU cite = `arevalo2017gmu` (ICLR Workshop, has paper.md). The 2020 NCAA entry
  has NO paper.md → not cited.
- Backbone cites backed by manual stubs under
  `references/methodology_backbones/{resnet,mobilenetv2,efficientnet}*/paper.md`
  (foundational architectures, not auto-extracted). Documented exception to the
  "tool-extracted only" convention.
- NOT cited (no paper.md): randaugment (named in text only), mcnemar, bilal.

## Open follow-ups
1. Ensure `results/tables/*` (extra_metrics, per_class_frozen_tta, training_time,
   ci_*_tta/ensemble) are git-tracked before submission.
2. Sections 04/05 (Results/Discussion) + abstract still to write.
3. Confirm author names + student numbers in `main.tex`; draw architecture diagram.
