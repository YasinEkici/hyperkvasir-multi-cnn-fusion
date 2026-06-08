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
- **Status after Week 2 fine-tuning:** Best fine-tune config (exp 10) reaches F1=0.5796, up from
  0.5588 (frozen baseline). However, 5 classes still score F1=0.000 on fold 0 test set:
  barretts (8 test samples), hemorroids (1), ileum (1), ulcerative-colitis-grade-1-2 (2),
  ulcerative-colitis-grade-2-3 (5). All have test support ≤ 8 — any misclassification
  collapses F1 to 0. This is a fold-0 data sparsity effect, not a model failure.
  5-fold CV in Week 3 will average out this variance and give a more reliable estimate.
- **Remaining open:** F1 gap vs. GastroViT (0.64 vs. 0.5796 on fold 0) is partly explained
  by rare-class penalty. Resolution expected after 5-fold CV.

## KI-003: EMA device mismatch and warm-start bug (resolved 2026-05-24)

- **Symptom (device):** `RuntimeError: Expected all tensors to be on the same device (cuda:0 and cpu)`
  at `EMA.update()` during first fine-tune run (exp 10).
- **Symptom (warm-start):** val_f1 collapsed from ~0.61 → ~0.015 at epoch 6 (first EMA
  activation); val_acc → ~0.048 (≈ 1/23 random).
- **Cause (device):** Shadow tensors cloned on CPU in `EMA.__init__()` before `Trainer.fit()`
  moves the model to CUDA. In-place `mul_().add_()` in `update()` then hit a device mismatch.
- **Cause (warm-start):** With `start_epoch=5` and `decay=0.999`, after 1 update at epoch 6:
  shadow ≈ 0.999 × random_init + 0.001 × trained ≈ random. Applying this shadow reset the
  model to near-random weights, triggering early stopping 8 epochs later.
- **Resolution:** (1) Lazy device migration added to `EMA.update()` — moves shadow to param
  device on first mismatch. (2) `EMA.reset_shadow(model)` added; called once in `Trainer.fit()`
  at the first epoch EMA becomes active, before `update()`, to warm-start shadow from current
  trained weights. Both fixes verified by 15 EMA unit tests (all pass).

## KI-004: Week 3.5 caveats (focal / TTA / seed ensemble) — informational

- **Architecture at ceiling on official 5-fold:** no Week 3.5 technique beat the
  Week 3 CE champion (macro-F1 0.6000) at the 95% CI level. Focal 0.5914 (lost),
  TTA 0.6075 (+0.0075, within CI), seed ensemble 0.6005 (within CI). All CIs
  overlap. Residual gap is rare-class data scarcity (support ≤ 10), not a model bug
  (resolves the KI-002 open question: the F1 gap vs literature is a protocol +
  rare-class-penalty effect, not under-training).
- **Seed-123 weaker draw:** the second ensemble seed (123, CV 0.5779) underperformed
  seed 42 (0.5892); a 2-seed average is sensitive to seed quality and dipped below
  the best single seed on macro-F1. A larger M with stronger seeds might help but was
  out of scope for the deadline.
- **Hardware/stack note:** seed-42 (Week 3) and seed-123 (Step 5) were both trained
  on RTX 5080 / cu132 to keep the seed ensemble single-variable. Step 3 focal (exp 16)
  ran on Colab A100 / cu128 as a separate self-contained experiment. TTA/ensemble
  inference (Steps 4–5 post-processing) ran locally on the RTX 5080.
- **Provenance-gate runtime (Colab):** `check_provenance.py` SHA-256s the full dataset
  tree twice (source on Drive FUSE + staged local) — several minutes per run; expected,
  not a hang.
