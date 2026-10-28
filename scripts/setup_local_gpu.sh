#!/usr/bin/env bash
# Install the local CUDA inference environment.
#
# Run this only after the NVIDIA driver is working:
#   nvidia-smi
#
# This script uses Python 3.12 and PyTorch's CUDA 12.6 wheels. The GTX 1060 is
# compute capability 6.1 (Pascal), which uses FP16 rather than BF16.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Install the NVIDIA driver first:"
  echo "  sudo apt update"
  echo "  sudo ubuntu-drivers install nvidia-driver-580"
  echo "  sudo reboot"
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi cannot talk to the GPU. Reboot and check the driver."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/"
  exit 1
fi

echo "Creating Python 3.12 environment in .venv-gpu"
uv venv --python 3.12 .venv-gpu

echo "Installing PyTorch 2.7.1 with CUDA 12.6"
uv pip install --python .venv-gpu/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

echo "Installing API and inference dependencies"
uv pip install --python .venv-gpu/bin/python \
  -r requirements-colab.txt -r requirements.txt

echo
echo "Verifying CUDA"
.venv-gpu/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GiB")
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

echo
echo "Done. Start the API with:"
echo "  ./scripts/run_local.sh"
