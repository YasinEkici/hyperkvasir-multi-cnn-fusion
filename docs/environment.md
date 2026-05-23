# Environment

Record local CUDA, PyTorch, torchvision, and hardware validation here.

## 2026-05-22 Local Environment Validation

- Command: `uv sync`
- Result: completed successfully.
- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- PyTorch: `2.12.0+cu132`
- torchvision: `0.27.0+cu132`
- PyTorch CUDA runtime: `13.2`
- CUDA available to PyTorch: `True`
- CUDA device count: `1`
- GPU: `NVIDIA GeForce RTX 5080 Laptop GPU`
- Dependency source note: `torch` and `torchvision` are pinned through uv to
  the PyTorch CUDA 13.2 wheel index:
  `https://download.pytorch.org/whl/cu132`.
- Verification command:
  `uv run python -c "import platform, sys, torch, torchvision; print('python', sys.version.replace(chr(10), ' ')); print('platform', platform.platform()); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('cuda_device_count', torch.cuda.device_count()); print('cuda_device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"`
