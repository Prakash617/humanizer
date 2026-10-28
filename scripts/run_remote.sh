#!/usr/bin/env bash
# Start the API and browser UI using the remote OpenAI-compatible backend.
#
# Put the provider key in .env first:
#   INFERENCE_BACKEND=remote
#   REMOTE_API_KEY=...
#
# Then run:
#   ./scripts/run_remote.sh
#
# Open http://127.0.0.1:8000/ui. No local model is loaded, so startup is fast.
# Change the port with: PORT=8001 ./scripts/run_remote.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/uvicorn ]; then
  UVICORN=.venv/bin/uvicorn
  PYTHON=.venv/bin/python
elif [ -x .venv-gpu/bin/uvicorn ]; then
  UVICORN=.venv-gpu/bin/uvicorn
  PYTHON=.venv-gpu/bin/python
else
  echo "No virtual environment with uvicorn was found."
  echo "Create one and install requirements.txt first."
  exit 1
fi

# Check the key without printing it.
if ! "$PYTHON" - <<'PY'
from app.config import get_settings

raise SystemExit(0 if get_settings().remote_api_key else 1)
PY
then
  echo "REMOTE_API_KEY is missing. Add a fresh key to .env, not to source control."
  exit 1
fi

export INFERENCE_BACKEND=remote

PORT="${PORT:-8000}"

echo "Starting AI Humanizer with the remote backend on http://127.0.0.1:${PORT}"
echo "UI:   http://127.0.0.1:${PORT}/ui"
echo "Docs: http://127.0.0.1:${PORT}/docs"
echo "Press Ctrl+C to stop."

exec "$UVICORN" app.main:app \
  --host 127.0.0.1 \
  --port "${PORT}"
