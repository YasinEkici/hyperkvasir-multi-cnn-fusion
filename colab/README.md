# Colab A100 Runner

Colab is a GPU runner only. This directory contains orchestration, not dataset,
model, training, evaluation, metric, or plotting logic. Any missing operation
must be implemented as a repository command under `scripts/`.

## Approved Inputs

- Repository: `https://github.com/YasinEkici/hyperkvasir-multi-cnn-fusion.git`
- Branch: `week3.5/perf-colab`
- Suggested Drive root: `/content/drive/MyDrive/hyperkvasir-multi-cnn-fusion`
- Smoke experiment: `11_triple_weighted_finetune_wide_official`, fold 0
- A100 training config: `configs/training/finetune_wide_a100.yaml`

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

## Commands

Step 1 smoke test, run by the user on Colab:

```bash
uv run --no-sync python scripts/run_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official \
  --folds 0 \
  --training configs/training/finetune_wide_a100.yaml \
  --device cuda
```

The later focal-loss command supplied for Step 3 is:

```bash
uv run --no-sync python scripts/run_cv.py \
  --experiment 16_triple_weighted_finetune_focal_official \
  --folds 0 1 2 3 4 \
  --training configs/training/finetune_wide_a100.yaml \
  --device cuda
```

Experiment 16 is intentionally not added in Step 1. It becomes runnable after
the focal-loss implementation and experiment-matrix entry in Steps 2–3.

## Session Resume

Each fold is a separate subprocess and output directory. After a session drop,
inspect `results/runs/`, set `FOLDS` to only the unfinished folds, and rerun the
training cell. Do not rerun a completed fold because training output directories
are not versioned.

## Returned Artifacts

`scripts/collect_outputs.py` refuses to overwrite
`returned_outputs/<run_id>/`. It requires the control records plus, for every
requested fold: `config.yaml`, `best.pt`, `epoch_log.jsonl`, `metrics.json`, and
`predictions.npz`. PNG/SVG figures are copied when present.
