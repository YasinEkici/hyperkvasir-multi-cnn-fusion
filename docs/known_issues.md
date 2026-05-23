# Known Issues

Track reproducibility, dataset, and runtime issues here.

## KI-001: PyTorch AMP deprecation warnings (resolved)

- **Symptom:** `FutureWarning: torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` are deprecated.
- **Affected file:** `src/training/trainer.py`
- **Resolution:** Updated to `torch.amp.GradScaler("cuda")` and `torch.amp.autocast("cuda")` (2026-05-23).

## KI-002: Macro F1 lower than accuracy on imbalanced classes

- **Symptom:** `01_single_resnet50_frozen_official` fold 0 reports accuracy=0.84 but macro_f1=0.56.
- **Cause:** HyperKvasir has severe class imbalance (e.g., hemorrhoids=6 images, ileum=9 images). Macro F1 averages per-class F1 equally, penalizing poor minority-class performance even when overall accuracy is high.
- **Mitigation in place:** `WeightedRandomSampler` is enabled for training. Label smoothing (0.1) is applied.
- **Expected improvement:** Multi-backbone fusion and fine-tuning (Week 2) should improve minority-class recall.
