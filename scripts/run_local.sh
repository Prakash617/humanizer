#!/usr/bin/env bash
# Start the real API and browser UI on the local GPU.
#
# Open http://127.0.0.1:8000/ui after the model finishes loading.
# Change the port with: PORT=8001 ./scripts/run_local.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv-gpu/bin/python ]; then
  echo "Missing .venv-gpu. Run ./scripts/setup_local_gpu.sh first."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "The NVIDIA GPU is not available. Check nvidia-smi and reboot if needed."
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PORT="${PORT:-8000}"

echo "Starting AI Humanizer on http://127.0.0.1:${PORT}"
echo "UI:   http://127.0.0.1:${PORT}/ui"
echo "Docs: http://127.0.0.1:${PORT}/docs"
echo "The model loads before /health becomes ready. Press Ctrl+C to stop."

exec .venv-gpu/bin/uvicorn app.main:app \
  --host 127.0.0.1 \
  --port "${PORT}"
