# Final Model — Frozen for Week 4

> Frozen 2026-06-08 at Week 3.5 close-out (exec plan 003.5, Step 7).
> Week 4 (confusion matrix, per-class table, Grad-CAM++) must build on THIS model
> and inference recipe — not on a stale checkpoint.

## Decision

**Final best model: `exp 11` (triple weighted fine-tune, CE) + Test-Time
Augmentation (TTA), single seed 42.**

### Why this over the seed ensemble

| Criterion | exp 11 + TTA | exp 11 + ensemble + TTA |
|---|---|---|
| Macro-F1 (pooled, **PLD-11 headline**) | **0.6075** ✅ | 0.6005 |
| 95% CI | [0.5860, 0.6296] | [0.5883, 0.6135] (higher lower-bound by 0.0023) |
| Acc / weighted-F1 / MCC | 0.8765 / 0.8761 / 0.8662 | 0.8810 / 0.8797 / 0.8710 ✅ |
| Checkpoints needed | 1 per fold (clean) ✅ | 2 per fold (seed 42 + 123) |
| Week-4 Grad-CAM++ / interpretability | single model, unambiguous ✅ | ensemble — attribution ambiguous |

- The **locked headline metric (macro-F1, PLD-11)** favors TTA (0.6075 > 0.6005):
  the weaker second seed (123) lowered the ensemble's rare-class average.
- The ensemble's only edge is a **+0.0023 higher CI lower-bound** — negligible and
  fully within CI overlap.
- **Week-4 practicality is decisive:** a single model gives clean, unambiguous
  Grad-CAM++ and per-class analysis; a 2-seed ensemble would need both checkpoints
  and makes interpretability ambiguous.
- Honest caveat: TTA's +0.0075 over base is **within CI overlap** — not a
  statistically significant gain. TTA is adopted because it is the best
  point-estimate at zero training cost and never lowers the aggregate.

## Frozen artifact specification

| Field | Value |
|---|---|
| Experiment id | `11_triple_weighted_finetune_wide_official` |
| Method | ResNet50 + MobileNetV2 + EfficientNetB0 → 512 proj → **weighted fusion** → MLP[256] |
| Training config | `configs/training/finetune_wide.yaml` (CE + label smoothing 0.1, unfreeze 3 blocks, LLRD, EMA, CutMix, RandAugment) |
| Seed | 42 |
| Protocol | Official HyperKvasir 5-fold; per-fold held-out test, pooled (n=10,662) |
| Hardware (training) | RTX 5080 / cu132 (Week 3) |

### Checkpoints (per fold, EMA best weights — gitignored, referenced by path)

```
fold 0: results/runs/11_triple_weighted_finetune_wide_official/best.pt
fold 1: results/runs/11_triple_weighted_finetune_wide_official_fold_1/best.pt
fold 2: results/runs/11_triple_weighted_finetune_wide_official_fold_2/best.pt
fold 3: results/runs/11_triple_weighted_finetune_wide_official_fold_3/best.pt
fold 4: results/runs/11_triple_weighted_finetune_wide_official_fold_4/best.pt
```

### Inference recipe (TTA) — must be reproduced exactly

Deterministic 4-view TTA, softmax averaged per image (`scripts/evaluate.py`):
1. `base`  — Resize(256) → CenterCrop(224)
2. `hflip` — base + horizontal flip (p=1.0)
3. `scale` — Resize(224) → CenterCrop(224)
4. `scale_hflip` — scale + horizontal flip (p=1.0)

ImageNet normalization (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]); BN in eval mode.

```bash
uv run python scripts/evaluate.py \
  --experiment 11_triple_weighted_finetune_wide_official \
  --folds 0 1 2 3 4 --tta --device cuda
```

## Frozen metrics (pooled, n=10,662)

| Metric | Value | Source |
|---|---|---|
| Macro-F1 | **0.6075**, 95% CI [0.5860, 0.6296] | `results/tables/ci_11_*_tta.json` |
| Accuracy / micro-F1 | 0.8765 | `results/tables/extra_metrics_11_*_tta.json` |
| Weighted-F1 | 0.8761 | same |
| MCC | 0.8662 | same |
| Macro P / R | 0.6162 / 0.6158 | same |

Per-fold test macro-F1 (TTA): 0.5827, 0.6126, 0.5808, 0.6025, 0.5987.
Predictions: `results/runs/11_*/predictions_tta.npz`.

## Context for the report

No Week 3.5 technique (focal, TTA, seed ensemble) beat the Week 3 CE champion
(0.6000) at the 95% CI level — all CIs overlap. The architecture is at its ceiling
on the official 5-fold protocol; the residual gap is driven by rare classes
(support ≤ 10), a data-scarcity limit not closed by loss/inference/ensemble tweaks.
The frozen model improves the Week 3 best macro-F1 point estimate from 0.6000 to
0.6075 at zero additional training cost.
