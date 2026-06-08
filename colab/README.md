# Colab A100 Runner

Colab is a GPU runner only. This directory contains orchestration, not dataset,
model, training, evaluation, metric, or plotting logic. Any missing operation
must be implemented as a repository command under `scripts/`.

## Approved Inputs

- Repository: `https://github.com/YasinEkici/hyperkvasir-multi-cnn-fusion.git`
- Branch: `week3.5/perf-colab`
- Suggested Drive root: `/content/drive/MyDrive/hyperkvasir-multi-cnn-fusion`
- Smoke experiment: `11_triple_weighted_finetune_wide_official`, fold 0
- Week 3.5 Step 3 experiment: `16_triple_weighted_finetune_focal_official`, folds 0-4
- Focal training config: `configs/training/finetune_wide_focal.yaml`
- A100 smoke config: `configs/training/finetune_wide_a100.yaml`

The Drive root must use this layout:

```text
hyperkvasir-multi-cnn-fusion/
|-- data/
|   `-- hyperkvasir/
|       `-- labeled-images/
`-- returned_outputs/
```

Split manifests come from the checked-out repository. The notebook copies the
Drive dataset to `/content/.../data/raw/hyperkvasir/labeled-images` before
training; training never reads images directly from Drive.

## Environment Rule

The local `pyproject.toml` is intentionally pinned to CUDA 13.2 for the RTX
5080. Colab must not run a normal `uv sync`, because that would import the local
CUDA choice. The runner creates Python 3.11 `.venv`, installs
`env/requirements-colab.txt`, and invokes commands with `uv run --no-sync`.

The Colab requirements use the official PyTorch CUDA 12.8 wheel pair
`torch==2.10.0` and `torchvision==0.25.0`. This pin remains provisional until
the real A100 smoke cell confirms the Colab driver, CUDA availability, and a
successful CUDA tensor operation.

## Hard Gates

The runner stops before training unless all conditions hold:

1. CUDA is available and device 0 contains `A100` in its name.
2. The checked-out commit matches the commit recorded after checkout.
3. The Drive source and `/content` staged dataset have the same deterministic
   SHA256 tree hash and per-class image counts.
4. Dataset counts match the official split manifest and contain 23 classes.
5. A per-run `resolved_config.yaml`, `runtime.json`, and provenance JSON exist.

Base files under `configs/` are never edited during a run.

## Private Repository Clone

If Colab fails with `fatal: could not read Username for 'https://github.com'`,
the repository is private or GitHub requires authentication. Use the notebook's
token prompt in cell 2. Create a fine-grained GitHub token with read-only
`Contents` access to this repository, paste it when prompted, and leave the
notebook cell source unchanged. Do not hard-code the token in the notebook.

## Commands

Step 1 smoke test, run by the user on Colab:

```bash
uv run --no-sync python scripts/run_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official \
  --folds 0 \
  --training configs/training/finetune_wide_a100.yaml \
  --device cuda
```

The focal-loss command for Step 3 is:

```bash
uv run --no-sync python scripts/run_cv.py \
  --experiment 16_triple_weighted_finetune_focal_official \
  --folds 0 1 2 3 4 \
  --training configs/training/finetune_wide_focal.yaml \
  --device cuda
```

Do not use `configs/training/finetune_wide_a100.yaml` for exp 16. That config
switches the loss back to CE + label smoothing and invalidates the focal-loss
ablation. Exp 16 must use `configs/training/finetune_wide_focal.yaml`, either
through the experiment matrix or via the explicit `--training` override above.

## Session Resume

Each fold is a separate subprocess and output directory. After a session drop,
inspect `results/runs/`, set `FOLDS` to only the unfinished folds, and rerun the
training cell. If you change `RUN_ID` for a resumed run, also rerun the resolved
config and provenance cells so returned artifacts stay traceable.

## Returned Artifacts

`scripts/collect_outputs.py` refuses to overwrite
`returned_outputs/<run_id>/`. It requires the control records plus, for every
requested fold: `config.yaml`, `best.pt`, `epoch_log.jsonl`, `metrics.json`, and
`predictions.npz`. PNG/SVG figures are copied when present.
