# Decisions

Project decisions are locked in `project_structure.md` and `project_plan.md`.

## 2026-05-23 - HyperKvasir Split Protocol

- Use the official HyperKvasir 5-fold split file as the primary Week 1 split
  protocol.
- Store the official split source under `data/splits/official/`.
- Keep any locally generated stratified split support as a fallback/internal
  utility only, not as the main reporting protocol.
- Materialize train/validation/test manifests deterministically from official
  folds: for fold `k`, use official fold `k` as test, official fold
  `(k + 1) % 5` as validation, and the remaining folds as training.
- The local raw dataset class folder names are aligned to the official class
  names, so no class-name mapping is required in the data pipeline.

## 2026-06-05 - D-08: A100 Exploration Scaling

- Colab A100 exploration runs use `configs/training/finetune_wide_a100.yaml`.
- Relative to `finetune_wide.yaml`, batch size scales from 64 to 128, head LR
  from `1e-3` to `2e-3`, and backbone LR from `1e-4` to `2e-4`.
- Exploration mode sets `deterministic: false` and `cudnn_benchmark: true`, as
  permitted by PLD-18. All other scientific settings remain unchanged.
- This is an additive runtime configuration. The base config is not mutated.
- If validation macro-F1 collapses during the smoke test, batch 64 / original
  learning rates remain the documented fallback; no test-set retuning is allowed.

## 2026-06-05 - D-09: Colab Runner-Only and Provenance Gate

- Colab notebooks contain orchestration only. Scientific logic remains in
  `src/`, `scripts/`, and `configs/`.
- Colab stages the approved Drive dataset to local `/content` storage before
  training. Drive-mounted image reads are not used during training.
- Every reportable run requires a git commit SHA, immutable per-run
  `resolved_config.yaml`, `runtime.json`, deterministic dataset SHA256,
  per-class counts, split-manifest identity, label scheme, approved source, and
  explicit returned-output location.
- `scripts/check_provenance.py` exits non-zero on any source/staged hash,
  per-class count, manifest, class-count, or git-SHA mismatch.
- The runner asserts that the CUDA device name contains `A100`; T4 and CPU
  sessions abort before training.
- Runs missing any required provenance record are provisional/debug-only and
  cannot supply headline results.

## 2026-06-06 - D-06: Focal Loss Added as Ablation

- Focal loss (Lin et al. 2017, ICCV) is added as an additive ablation on top of
  the Week 3 best config (exp 11, triple weighted fine-tune).
- This is NOT a PLD change. `project_plan.md` §1 ("focal loss … meaningful") and
  §9 risk register ("focal loss ablation") both anticipate it.
- Implementation: `FocalLoss` in `src/training/losses.py` using the α-balanced
  variant (paper eq. 5): `FL(p_t) = -α_t (1 - p_t)^γ log(p_t)`.
- `build_loss` extended to support `type: "focal"` (γ-only, no class weighting)
  and `type: "focal_balanced"` (γ + per-class α from inverse class frequency).
- Focal loss and label smoothing are deliberately NOT combined. Label smoothing
  modifies the target distribution, which conflicts with the focal modulating
  factor that relies on the true-class probability p_t. The ablation config
  (`finetune_wide_focal.yaml`) removes label smoothing to ensure a clean
  comparison.
- Default γ=2.0 (paper-recommended value).
- Exp 16 (`16_triple_weighted_finetune_focal_official`) added to experiment
  matrix. It uses `finetune_wide_focal.yaml` and differs from exp 11's
  `finetune_wide.yaml` only in the loss block — all other hyperparameters
  are identical for a single-variable ablation.
- Week 3 CE runs (exp 11) are preserved unchanged for comparison.

## 2026-06-08 - D-07: Leakage-free Ensembling Policy

- Only **leakage-free** ensembling is permitted on the official 5-fold protocol.
- **Seed ensemble (allowed):** train M seeds of the same config; average their
  per-image **softmax within each fold's own held-out test set**, then pool the
  5 folds so each sample appears exactly once (n=10,662). Implemented in
  `scripts/ensemble_seeds.py`; new seeds use a `--seed` override and write to
  `{exp}_seed{S}[_fold_{k}]` (canonical seed-42 runs untouched).
- **TTA (allowed):** per-model, per-image, deterministic test-time views
  (see Step 4) — averaged softmax, no leakage.
- **Naive cross-fold model averaging is FORBIDDEN:** averaging the 5 fold models
  over the full dataset leaks, because each sample was in 4 of the 5 folds'
  training sets. Never ensemble models across folds.
- Always average **softmax probabilities**, never argmax votes.
- Step 5 finding: a 2-seed ensemble reduced variance (narrower CI; best
  accuracy/weighted-F1/MCC) but did not raise macro-F1 (the second seed was the
  weaker draw). By macro-F1 (PLD-11), exp 11 + TTA remains the best model.
