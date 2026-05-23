# Multi-CNN Fusion for HyperKvasir Classification

Multi-CNN feature-fusion classifier scaffold for the HyperKvasir 23-class labeled subset.

## Setup

```bash
uv sync
uv run pytest tests/
```

Prepare data and run MVP entry points:

```bash
uv run python scripts/prepare_data.py --dataset hyperkvasir
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_own.yaml
uv run python scripts/extract_features.py --config configs/dataset/hyperkvasir_23class_own.yaml
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 01_single_resnet50_frozen_own
```

Colab fallback:

```bash
pip install -r env/requirements-colab.txt
```
