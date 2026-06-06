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
