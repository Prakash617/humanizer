# AI Humanizer

A beginner-friendly prototype that rewrites English text for better naturalness,
clarity, fluency, and readability while preserving meaning.

The project is developed on an Ubuntu laptop with VS Code. Inference can run in
three places:

- locally on the GTX 1060 with Qwen2.5-1.5B-Instruct
- in Google Colab on a GPU
- remotely through an OpenAI-compatible provider such as NVIDIA NIM

> **Status: complete prototype.**
> The local model, remote backend, Pydantic schemas, FastAPI endpoints, mocked test
> suite, gated integration tests, browser UI, and two Colab notebooks are included.
> The full API is demonstrated in `notebooks/colab_api_demo.ipynb`.

> **What this project is not.** It improves writing quality. It does not guarantee
> a particular AI-detector result and must not be presented as a detector-bypass
> tool. It is not a permanent public service.

## Contents

- [How it works](#how-it-works)
- [Model decision](#model-decision)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Run the tests](#run-the-tests)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Browser UI](#browser-ui)
- [Run the API locally](#run-the-api-locally)
- [Local GPU mode (GTX 1060 6 GB)](#local-gpu-mode-gtx-1060-6-gb)
- [Remote backend (NVIDIA NIM)](#remote-backend-nvidia-nim)
- [Colab notebook 1: model experiments](#colab-notebook-1-model-experiments)
- [Colab notebook 2: full API demo](#colab-notebook-2-full-api-demo)
- [Security and Colab lifetime](#security-and-colab-lifetime)
- [Troubleshooting](#troubleshooting)
- [Deployment roadmap](#deployment-roadmap)
- [Checkpoints](#checkpoints)
- [Limitations](#limitations)

## How it works

```text
Ubuntu laptop (VS Code)
        |
        |  FastAPI routes + /ui
        v
  app/main.py
        |
        +-- local:   app/humanizer.py -> Qwen on GTX 1060 or Colab GPU
        |
        +-- remote:  app/remote.py    -> NVIDIA NIM / OpenAI-compatible API
```

Responsibilities are deliberately separated:

| File | Responsibility |
|---|---|
| `app/config.py` | Settings from environment variables and `.env` |
| `app/humanizer.py` | Local model loading, prompt, chunking, generation, fallback |
| `app/remote.py` | Optional OpenAI-compatible remote inference |
| `app/schemas.py` | Pydantic request/response models and style enum |
| `app/main.py` | FastAPI app, lifespan, backend selection, routes, error mapping |

The API never imports a frontend. A future web or desktop frontend should call the
HTTP endpoints instead of importing Python model objects.

## Model decision

| Item | Finding |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| License | **Apache-2.0** |
| Parameters | 1.54B total (1.31B non-embedding) |
| Weight type | BF16 |
| Approximate weights | ~3.1 GB, plus KV cache and runtime overhead |
| Context | 32,768 tokens |
| Transformers | `>=4.37.0` required for Qwen2; Colab pins 5.21.0 |
| Generation defaults | temperature `0.7`, top-p `0.8`, top-k `20`, repetition penalty `1.1` |
| Fallback | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) |

The model is public and does not require a Hugging Face token. A free Colab T4 GPU
has enough memory. CPU inference works but is much slower.

Model card: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>

Fallback: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>

### Optional remote model

| Item | Finding |
|---|---|
| Provider | NVIDIA NIM |
| Model | `nvidia/nemotron-3-super-120b-a12b` |
| Size | 120B total, 12B active (MoE) |
| Context | Up to 1M tokens |
| Strength | Strong instruction following and long-context editing |
| Caveat | Hosted inference sends text to the provider and may cost money |

Other NIM text models such as `meta/llama-3.3-70b-instruct`,
`nemotron-3.5-lightning-30b-a3b`, or `openai/gpt-oss-20b` can be selected with
`REMOTE_MODEL`. The remote backend is optional; local inference remains the
default.

## Project structure

```text
humanizer/
├── app/
│   ├── __init__.py                    # Package version
│   ├── config.py                      # Environment/.env settings
│   ├── humanizer.py                   # Local model loading and rewriting
│   ├── remote.py                      # OpenAI-compatible remote inference
│   ├── main.py                        # FastAPI app, routes, backend selection
│   ├── schemas.py                     # Pydantic request/response models
│   └── ui.py                          # Built-in browser UI
├── tests/
│   ├── __init__.py
│   ├── test_api.py                    # Mocked API tests
│   ├── test_remote.py                 # Offline remote-backend tests
│   ├── test_integration_real_model.py # Real local model, opt-in only
│   └── test_integration_remote.py     # Real remote provider, opt-in only
├── notebooks/
│   ├── humanizer_experiments.ipynb    # Model and prompt experiments
│   └── colab_api_demo.ipynb           # Full API demo in Colab
├── scripts/
│   ├── setup_local_gpu.sh             # Python 3.12 + CUDA environment
│   ├── run_local.sh                   # Start the local GPU API and UI
│   └── run_remote.sh                  # Start the remote-backend API and UI
├── requirements.txt                   # Lightweight local/API dependencies
├── requirements-colab.txt             # Colab inference dependencies
├── pytest.ini                         # Test paths and markers
├── .env.example                       # Safe configuration template
├── .gitignore                         # Secrets, caches, weights, outputs
└── README.md
```

## Requirements

- Ubuntu 24.04 LTS (nearby versions work too)
- Python 3.11 or 3.12 (3.12 matches common Colab runtimes)
- Git
- Visual Studio Code
- Internet access for package installation

The lightweight `.venv` is enough for the remote backend and mocked tests. Local
GPU inference needs an NVIDIA driver and the separate `.venv-gpu` environment.
Colab remains an option when no local GPU is available. The remote backend uses
only the Python standard library for HTTP, so it adds no packages.

## Local setup

### 1. Verify the tools

```bash
python3 --version
git --version
code --version
```

If `code` is missing, open VS Code and run
**Shell Command: Install 'code' command in PATH** from the Command Palette.

Ubuntu 24.04 normally provides Python 3.12. If it is missing:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git
```

### 2. Enter the project and create a virtual environment

```bash
cd ~/Desktop/program/ai/humanizer
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should begin with `(.venv)`. Leave it later with `deactivate`.

### 3. Install lightweight dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Pydantic, configuration helpers, pytest, and
httpx2. It intentionally does **not** install PyTorch, Transformers, Accelerate, or
CUDA. Those are installed in `.venv-gpu` for local inference or in Colab.

### 4. Create local configuration

```bash
cp .env.example .env
```

The real `.env` is ignored by Git. Never put tokens, passwords, or private keys in
`.env.example` or any source file.

## Run the tests

From the project root, with the virtual environment active:

```bash
python -m pytest -q
```

Expected:

```text
34 passed, 2 skipped
```

The default tests use a fake model and a fake HTTP response. They never download
weights, import PyTorch, contact a provider, or need a GPU. They check:

- `GET /` metadata
- `GET /ui` serves the browser UI
- `GET /health` when the model is loaded or failed
- successful humanization and response schema
- empty, whitespace-only, and missing text rejection
- invalid style rejection
- oversized input rejection (`413`)
- inference failure (`500`)
- model-not-loaded behavior (`503`)
- optional `X-API-Key` protection (`401` then `200`)
- remote request construction, authentication, rate limits, timeouts, and malformed responses (`429`, `502`, `504`)

Two integration tests are skipped by default. The local real-model test needs
PyTorch, Transformers, and enough memory:

```bash
RUN_REAL_MODEL_TESTS=1 python -m pytest tests/test_integration_real_model.py -q -s
```

It downloads the real model and performs one request through `TestClient`. The
remote provider test is described in the NIM section above.

## Configuration

Settings are read from environment variables first, then `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `AI Humanizer API` | API title |
| `APP_VERSION` | `0.2.0` | Reported version |
| `LOG_LEVEL` | `INFO` | Log level |
| `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Primary model |
| `FALLBACK_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Used if the primary fails |
| `HF_TOKEN` | empty | Optional private/gated model token |
| `MAX_INPUT_CHARACTERS` | `8000` | Request validation limit |
| `MAX_INPUT_TOKENS` | `6000` | Safe model input limit |
| `MAX_CHUNK_TOKENS` | `1400` | Paragraph chunk size |
| `MAX_NEW_TOKENS` | `1024` | Generation budget per chunk |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.8` | Nucleus sampling |
| `TOP_K` | `20` | Top-k sampling |
| `REPETITION_PENALTY` | `1.1` | Repetition control |
| `API_KEY` | empty | When set, `POST /humanize` needs `X-API-Key` |
| `INFERENCE_BACKEND` | `local` | `local` or `remote` |
| `REMOTE_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible base URL |
| `REMOTE_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Hosted model ID |
| `REMOTE_API_KEY` | empty | Provider key; required when the backend is remote |
| `REMOTE_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `REMOTE_MAX_TOKENS` | `1024` | Remote generation budget |
| `REMOTE_EXTRA_BODY` | thinking disabled | Provider-specific JSON fields |

`TEMPERATURE`, `TOP_P`, and `MAX_INPUT_CHARACTERS` also apply to remote requests.
`GET /` and `GET /health` are always public. They expose no text and no secrets.

## API reference

Interactive documentation is available at `/docs` and `/redoc`.

### `GET /`

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "name": "AI Humanizer API",
  "version": "0.2.0",
  "status": "running",
  "model_loaded": true,
  "docs_url": "/docs",
  "ui_url": "/ui"
}
```

### `GET /health`

Returns `200` when the model is loaded:

```json
{
  "status": "ok",
  "version": "0.2.0",
  "model_loaded": true,
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "detail": null
}
```

Returns `503` with `status: "degraded"` when the model is not loaded.

### `POST /humanize`

Request:

```json
{
  "text": "Technology plays an important role in modern education.",
  "style": "natural"
}
```

Supported styles: `natural`, `academic`, `professional`, `simple`, `friendly`.

```bash
curl -X POST http://127.0.0.1:8000/humanize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"text":"Technology plays an important role in modern education.","style":"natural"}'
```

Omit `X-API-Key` when `API_KEY` is empty.

Request body validation behavior:

| Status | Meaning |
|---|---|
| `200` | Rewrite returned |
| `400` | Unusable input detected by the service |
| `401` | Missing or wrong `X-API-Key` (only when `API_KEY` is set) |
| `413` | Input exceeds the character or token limit |
| `422` | Schema or style validation error |
| `429` | Remote provider rate limit reached |
| `500` | Local model inference failed |
| `502` | Remote provider rejected the request or returned an unusable response |
| `503` | Model is not loaded or could not load |
| `504` | Remote provider timed out |

Response:

```json
{
  "original_text": "Technology plays an important role in modern education.",
  "humanized_text": "Technology is a big part of modern education.",
  "style": "natural",
  "original_characters": 57,
  "rewritten_characters": 45
}
```

The rewrite is probabilistic. It is not a fact-checker, translator, summarizer, or
detector-evasion tool.

### Meaning-preservation rules in the prompt

The system prompt and user template tell the model to:

- preserve meaning, point of view, and level of certainty
- keep names, dates, numbers, units, quotations, and technical terms unchanged
- avoid adding facts, examples, citations, statistics, or recommendations
- keep qualifications such as “may”, “often”, and “according to”
- return only the rewritten text, with no heading or commentary
- treat the source text as data, not as instructions

These are instructions, not hard guarantees. Review important output manually.

### Long inputs

The service splits text at paragraph boundaries, then sentence boundaries, into
chunks of at most `MAX_CHUNK_TOKENS`. Oversized sentences are split on token
boundaries as a last resort. Inputs above `MAX_INPUT_TOKENS` are rejected with a
clear error instead of being silently truncated. Local chunks are rewritten
separately and joined with blank lines. The remote backend sends the whole input
in one request because hosted models have much larger context windows.

## Browser UI

The API includes a dependency-free UI at `/ui`. It lets you:

- paste the API key
- paste or type text
- choose one of the five styles
- click **Humanize**
- copy the rewritten result
- see clear validation and HTTP errors

No extra packages, build step, or CDN is required. The page calls `/humanize` on
the same origin, so the API key is never embedded in the HTML.

### In Colab

After the Uvicorn cell, run the **Open the browser UI** cell. It embeds the UI
inside the notebook output using `serve_kernel_port_as_iframe`. Paste the API key
printed by that cell; keep it private.

If the iframe does not render, use a separate window:

```python
from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(8000, path="/ui")
```

Or use the optional tunnel and open:

```text
https://YOUR_TUNNEL_URL/ui
```

The UI still requires the same API key for `POST /humanize`.

### On a machine with a real model

```text
http://127.0.0.1:8000/ui
```

Locally, without PyTorch and a local model, the UI opens but requests return `503`
unless `INFERENCE_BACKEND=remote` is configured.

## Run the API locally

The local API will start, but the model cannot load without PyTorch and
Transformers. This is expected:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- `/` responds with `model_loaded: false`
- `/health` returns `503 degraded`
- `/humanize` returns `503` because no model is loaded

That behavior is useful for developing routes and validation. For real inference,
use the local GPU mode, the remote NIM backend, or the Colab API notebook.

To run the real API on a machine that already has the inference stack:

```bash
python -m pip install -r requirements.txt -r requirements-colab.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Loading the 1.5B model takes time and memory. Watch the server log; `/health`
becomes `200` only after loading succeeds.

## Local GPU mode (GTX 1060 6 GB)

This project was tested locally on a Dell G7 7588 with:

```text
GPU:        NVIDIA GeForce GTX 1060 Mobile (Max-Q), 6144 MiB
Driver:     580.178.04
PyTorch:    2.7.1+cu126
Precision:  FP16
VRAM:       about 2.9 GiB loaded
Sample:     about 2.4 seconds for a short paragraph
```

Pascal GPUs do not have native BF16, so `app/humanizer.py` selects FP16 on
compute capability 6.x. Newer Ampere and later cards use BF16.

### 1. Install the NVIDIA driver once

```bash
sudo apt update
sudo ubuntu-drivers install nvidia-driver-580
sudo reboot
```

After reboot, confirm the GPU is visible:

```bash
nvidia-smi
```

### 2. Create the GPU environment

```bash
cd ~/Desktop/program/ai/humanizer
./scripts/setup_local_gpu.sh
```

This creates `.venv-gpu` with Python 3.12, PyTorch CUDA 12.6, Transformers,
Accelerate, FastAPI, and the test dependencies. It does not change `.venv`.

### 3. Start the API and UI

```bash
./scripts/run_local.sh
```

Open:

```text
http://127.0.0.1:8000/ui
```

Use another port with:

```bash
PORT=8001 ./scripts/run_local.sh
```

The server binds to `127.0.0.1`, so it is local only. If `API_KEY` is empty in
`.env`, the UI does not need a key. If you set `API_KEY`, paste that key into the
UI.

### 4. Verify the GPU run

```bash
curl http://127.0.0.1:8000/health
nvidia-smi
```

`/health` should return `model_loaded: true`, and `nvidia-smi` should show roughly
3 GB of GPU memory in use.

### Memory notes

The laptop has 6.9 GB system RAM. Close heavy browser tabs and other applications
before loading the model. If the 1.5B model runs out of memory, switch to the
smaller fallback in `.env`:

```text
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

The GTX 1060 is a Pascal card. It works, but generation is slower than a modern
Colab GPU. Short paragraphs are fine; long documents take longer.

## Remote backend (NVIDIA NIM)

Remote mode is optional. It sends each request to an OpenAI-compatible endpoint and
does not load a local model. This is useful on a machine without a GPU, or when a
larger model produces better prose.

A larger model improves writing quality, but it still does **not** guarantee a
particular AI-detector result. The project remains a writing tool, not a
detector-bypass service.

### 1. Add a fresh key to `.env`

If a key was ever pasted into a chat, issue tracker, or commit, revoke it in the
NVIDIA console and create a new one. Do not reuse an exposed key.

```dotenv
INFERENCE_BACKEND=remote
REMOTE_BASE_URL=https://integrate.api.nvidia.com/v1
REMOTE_MODEL=nvidia/nemotron-3-super-120b-a12b
REMOTE_API_KEY=your-fresh-key

REMOTE_TIMEOUT_SECONDS=120
REMOTE_MAX_TOKENS=1024

# Nemotron reasons by default. Disable thinking for faster rewriting.
REMOTE_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}
```

Confirm the secret file is ignored:

```bash
git check-ignore .env
git status --short
```

`.env` must never appear in `git status`. Do not put the key in `.env.example`,
source files, shell history, screenshots, or chat.

### 2. Start the remote API and UI

```bash
./scripts/run_remote.sh
```

Open:

```text
http://127.0.0.1:8000/ui
```

Startup is fast because no weights are downloaded. The script checks that
`REMOTE_API_KEY` is present without printing it.

### 3. Verify

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/humanize \
  -H "Content-Type: application/json" \
  -d '{"text":"In 2024, Dr. Chen wrote 3 reports.","style":"professional"}'
```

`/health` should report the remote model ID. A bad key returns `502`; a provider
timeout returns `504`; a rate limit returns `429`.

### 4. Model choices

The default is a strong general-purpose choice:

| Model ID | Notes |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | 120B total / 12B active, long context, thinking disabled for editing |
| `meta/llama-3.3-70b-instruct` | Strong classic instruct model, no reasoning overhead |
| `nemotron-3.5-lightning-30b-a3b` | Faster and cheaper for short edits |
| `openai/gpt-oss-20b` | Smaller open model, lower cost |

Availability depends on your NVIDIA account and region. To list the exact IDs your
key can access without printing the key:

```bash
.venv/bin/python - <<'PY'
import json
import urllib.request

from app.config import get_settings

settings = get_settings()
request = urllib.request.Request(
    f"{settings.remote_base_url}/models",
    headers={"Authorization": f"Bearer {settings.remote_api_key}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)

for model in payload.get("data", []):
    print(model.get("id"))
PY
```

If a model ID returns `404`, choose one from that list.

### 5. Test the remote backend

Offline tests use a fake HTTP response and never contact the provider:

```bash
python -m pytest tests/test_remote.py -q
```

The real-provider test is opt-in and may incur charges:

```bash
RUN_REMOTE_TESTS=1 python -m pytest tests/test_integration_remote.py -q -s
```

### Privacy and cost

- Your text leaves the laptop and goes to the provider.
- The provider's retention and training policies apply.
- Hosted inference may cost money; check the provider's current pricing.
- Do not send confidential material unless the provider's terms allow it.

## Colab notebook 1: model experiments

`notebooks/humanizer_experiments.ipynb` explores the model and prompt before the
API. It:

1. checks the Colab Python and GPU
2. clones or updates the repository
3. installs the pinned inference requirements
4. verifies the model card, license, revision, and repository size
5. loads Qwen2.5-1.5B-Instruct with the official chat template
6. runs a basic generation test
7. defines and tests `humanize_text(text, style="natural")`
8. rewrites sample paragraphs
9. compares all five styles on one factual paragraph
10. computes automatic preservation/readability signals
11. provides a human review rubric
12. saves results without model weights

Use it when you want to inspect model behavior and tune the prompt. Use the API
notebook when you want to exercise the actual HTTP service.

## Colab notebook 2: full API demo

`notebooks/colab_api_demo.ipynb` is the end-to-end demonstration. It loads one
model, passes that same service object to the FastAPI app, tests the app in-process,
starts Uvicorn, and optionally exposes a temporary authenticated tunnel.

### Exact steps

1. Open the notebook directly in Colab:

   <https://colab.research.google.com/github/Prakash617/humanizer/blob/main/notebooks/colab_api_demo.ipynb>

   Or use **File > Open notebook > GitHub** and select
   `Prakash617/humanizer`, then `notebooks/colab_api_demo.ipynb`.
2. In Colab, choose **Runtime > Change runtime type > T4 GPU**.
3. Optional: add Colab Secrets (key icon in the left sidebar):
   - `HUMANIZER_API_KEY` — your API key. If absent, a random key is generated.
   - `HF_TOKEN` — only for private/gated models. Not needed for Qwen.
   - `NGROK_AUTHTOKEN` — only for the optional public tunnel.
4. Choose **Runtime > Run all**, or run the cells top to bottom. The clone cell is
   already set to:
   ```python
   REPO_URL = "https://github.com/Prakash617/humanizer.git"
   ```
5. Read the output of each cell. Do not skip the mocked test cell.

### What the notebook proves

| Cell | What it demonstrates |
|---|---|
| Runtime check | Python, GPU, RAM, and disk information |
| Environment | Hugging Face cache outside the repository |
| Clone | The laptop code runs from Git in Colab |
| Install | Colab-specific inference dependencies |
| Unit tests | 34 offline tests pass in the Colab runtime |
| Load model | Qwen loads once; the API key is created or read |
| TestClient | `/`, `/health`, `401` without key, `200` with key |
| Uvicorn | The same app serves real local HTTP on port 8000 |
| UI | Embeds `/ui` in the notebook for point-and-click use |
| Tunnel | Optional temporary public URL, still key-protected |
| Stop | Stops Uvicorn and the tunnel cleanly |
| Troubleshooting | Common Colab failures and fixes |
| Checkpoint | Final verification list |

### Expected result

- The offline suite reports `34 passed, 2 skipped`.
- The model loads once. The notebook prints the model ID, device, and dtype.
- `GET /` returns API metadata with `model_loaded: true` and `ui_url: /ui`.
- `GET /health` returns `200` and the loaded model ID.
- `POST /humanize` without `X-API-Key` returns `401`.
- `POST /humanize` with `X-API-Key` returns a rewritten paragraph.
- Uvicorn serves the same app at `http://127.0.0.1:8000` inside Colab.
- The UI cell embeds `/ui`; pasting the API key enables point-and-click rewriting.
- The optional ngrok cell prints a temporary HTTPS URL. The API key is still
  required.

Example real HTTP call from inside the notebook:

```python
import json
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8000/humanize",
    data=json.dumps({
        "text": "In 2024, the team released version 2.0 of the library.",
        "style": "professional",
    }).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    },
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.status)
    print(json.loads(response.read().decode("utf-8")))
```

### Why the model is loaded only once

`create_app(service)` accepts an injected service. In Colab:

```python
service = ModelHumanizer()
service.load()

api_app = create_app(service)
```

The app lifespan calls `service.load()` again, but loading is idempotent, so it
returns immediately. Running a plain `uvicorn app.main:app` would create a separate
process and load a second copy of the weights. On a T4 GPU that can cause
out-of-memory errors.

### Temporary path to the outside world (optional)

The tunnel cell uses ngrok only if `NGROK_AUTHTOKEN` is configured. A Colab
`output` or `serve_kernel_port_as_thread` URL is usually easier for a quick demo,
but those mechanisms change often and are not a security boundary.

Do not create a tunnel when `API_KEY` is empty. The notebook sets `API_KEY` before
importing the app, so `POST /humanize` requires the header by default.

## Security and Colab lifetime

- Bind Uvicorn to `127.0.0.1` unless you intentionally create a tunnel.
- Keep `API_KEY` set whenever a tunnel exists.
- Never commit `.env`, provider keys, API keys, ngrok tokens, or Hugging Face tokens.
- Treat any key pasted into chat, a log, or a screenshot as exposed and rotate it.
- Remote mode sends text to a third party. Review that provider's data policy before
  submitting private or confidential material.
- Colab deletes packages, model weights, the Git clone, the server, the tunnel, and
  the generated API key when the runtime stops.
- A Colab URL is not a production service. It has no SLA, rate limiting, billing
  controls, or durable storage.
- Do not expose an unauthenticated inference endpoint. GPU inference is expensive
  and can be abused.

If a secret is ever committed, rotate it immediately. Deleting it in a later commit
is not enough because it remains in Git history.

`.gitignore` excludes virtual environments, caches, `.env`, secrets, Hugging Face
caches, model weights (`.safetensors`, `.bin`, `.pt`, `.pth`, `.gguf`, and similar),
experiment outputs, logs, notebook checkpoints, and editor files.

## Troubleshooting

### `ensurepip is not available` or `No module named venv`

```bash
sudo apt install -y python3.12-venv
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
```

### `python3.12: command not found`

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

### `pip: command not found` or the wrong Python is used

```bash
which python
python -m pip --version
```

Use `python -m pip`, not bare `pip`.

### `ModuleNotFoundError: No module named 'app'`

Run pytest and Uvicorn from the project root, not from `app/` or `tests/`.

### Port 8000 is already in use

```bash
uvicorn app.main:app --reload --port 8001
```

In Colab, run the stop cell and restart the runtime.

### Local `/health` returns `503`

The local model is not loaded. Check the server log for a download, memory, or
driver error. Either install the local inference stack (`./scripts/setup_local_gpu.sh`),
use Colab, or switch to `INFERENCE_BACKEND=remote` with a valid
`REMOTE_API_KEY`.

### Remote `/humanize` returns `502`

- `The remote provider rejected the API key`: replace `REMOTE_API_KEY` with a fresh,
  valid key and restart the server.
- `HTTP 404`: the model ID is not available to your account. List the models your
  key can access, as shown in the NIM section.
- `HTTP 429`: the provider rate limit was reached. Wait and retry.
- `invalid JSON` or `did not contain choices`: the provider returned an unexpected
  response. Check the base URL and model ID.

### Remote `/humanize` returns `504`

The provider did not respond before `REMOTE_TIMEOUT_SECONDS`. Increase the timeout
for long inputs, or retry when the provider is less busy.

### `401 Invalid or missing API key`

Include the header:

```text
X-API-Key: YOUR_KEY
```

The key is created in the model-loading cell or read from the
`HUMANIZER_API_KEY` Colab Secret. It is masked in notebook output.

### Colab reports `CUDA out of memory`

1. Choose **Runtime > Restart runtime**.
2. Run the notebook from the top.
3. Do not run the experiments notebook and the API notebook at the same time.
4. Let the loader fall back to `Qwen/Qwen2.5-0.5B-Instruct`.

### Colab says CUDA is unavailable

- Confirm the runtime type is GPU.
- Do not reinstall an arbitrary PyTorch CUDA wheel. Restart and let Colab provide
  its matched build.
- Print `torch.__version__`, `torch.version.cuda`, and
  `torch.cuda.is_available()`.

### Model download fails in Colab

- Check internet access and free disk space.
- Confirm `REPO_URL` and the branch in the clone cell.
- Qwen is public. An invalid `HF_TOKEN` can still cause authentication errors;
  remove the secret and retry.
- Re-run the loading cell; Hugging Face may resume a partial download.

### `qwen2`, `apply_chat_template`, or Transformers import errors

Restart the runtime and reinstall the pinned Colab requirements. Qwen2 requires
Transformers 4.37+, and mixing versions in one runtime causes confusing errors.
The Colab pins are `transformers==5.21.0`, `accelerate==1.15.0`,
`jinja2==3.1.6`, and `textstat==0.7.13`.

### `torch.cuda.is_available() == False` after installing packages

A mismatched PyTorch wheel replaced Colab's CUDA build. Restart the runtime and do
not pin PyTorch in `requirements-colab.txt`. The notebook installs an unpinned
fallback only when PyTorch is missing.

### Uvicorn starts but requests fail

- Confirm the cell printed `Uvicorn started`.
- Check that port 8000 is free in the runtime.
- Run the stop cell, restart the runtime, and rerun from the top.

### The UI does not appear in Colab

- Run the Uvicorn cell first and wait for `Uvicorn started`.
- Rerun the UI cell after the server is ready.
- If the iframe is blocked, open a separate window:
  ```python
  from google.colab.output import serve_kernel_port_as_window
  serve_kernel_port_as_window(8000, path="/ui")
  ```
- Through a tunnel, open `TUNNEL_URL/ui` and paste the API key.

### ngrok shows an HTML warning page

For API requests add:

```text
ngrok-skip-browser-warning: true
```

This does not replace the API key.

### Long input is rejected

This is deliberate. Inputs above `MAX_INPUT_CHARACTERS` or `MAX_INPUT_TOKENS` are
rejected instead of silently truncated. Split the document into sections or raise
the limits carefully.

### Notebook cells fail after a runtime reset

Installed packages, the Git clone, model weights, the server, and the tunnel are
gone. Rerun the notebook from the top.

## Deployment roadmap

The prototype is intentionally local/temporary. A real deployment needs:

1. **Serving host** — an authenticated cloud service with a GPU, or a managed
   inference provider.
2. **Process model** — one model per worker, a queue for long requests, and
   concurrency limits. Do not spawn a model per request.
3. **Authentication** — user accounts or signed API keys stored in a secret
   manager, not in `.env` on a public server.
4. **Rate limits and quotas** — per-user request and token limits; GPU inference is
   expensive.
5. **Timeouts and retries** — streaming or asynchronous jobs for long documents.
6. **Observability** — structured logs, latency/error metrics, GPU memory alerts,
   and request IDs.
7. **Data policy** — retention rules, encryption, and a clear statement that
   submitted text may be processed by a third party.
8. **Frontend** — a small web app that calls the API; never expose the model
   directly.
9. **Evaluation** — automated meaning-preservation checks plus human review on a
   fixed test set.
10. **Safety** — abuse monitoring, content policy, and an explicit statement that
    the tool does not bypass AI detectors.

Do not deploy this prototype as an open, unauthenticated endpoint.

## Checkpoints

### Phase 1 — scaffold

- [x] Virtual environment and lightweight dependencies
- [x] FastAPI scaffold and `/docs`
- [x] `.gitignore` excludes `.venv/`, `.env`, weights, and outputs
- [x] Initial Git commit

### Phase 2 — model experiments

- [x] Model card reviewed: Apache-2.0, public, ~3.1 GB
- [x] Colab experiment notebook created and syntax-validated
- [x] Pinned Colab inference dependencies
- [x] Prompt, styles, chunking, and evaluation rubric documented
- [x] Results saved outside the repository

### Phase 3 — inference module

- [x] `ModelHumanizer` loads the primary model and retries the fallback
- [x] FP16 on Pascal/older CUDA GPUs, BF16 on Ampere+, FP32 on CPU
- [x] Chat template and generation settings
- [x] Paragraph/sentence/token chunking
- [x] Validation, domain errors, and safe output clearing
- [x] `humanize_text(text, style="natural")` interface

### Phase 4 — FastAPI

- [x] `GET /`, `GET /health`, `POST /humanize`
- [x] Pydantic schemas and style enum
- [x] Lifespan model loading with injected service support
- [x] Error mapping (`400`, `401`, `413`, `422`, `429`, `500`, `502`, `503`, `504`)
- [x] Optional `X-API-Key` protection
- [x] Built-in browser UI at `/ui`

### Phase 5 — tests and Colab demonstration

- [x] 34 offline tests pass locally: `34 passed, 2 skipped`
- [x] Real-model integration test gated by `RUN_REAL_MODEL_TESTS=1`
- [x] Colab API notebook created and syntax-validated
- [x] Notebook loads one model, uses `TestClient`, starts Uvicorn, optional tunnel
- [ ] Optional: run `notebooks/colab_api_demo.ipynb` in Colab and tick the
      in-notebook checkpoint

### Phase 6 — local GPU validation

- [x] NVIDIA driver `580.178.04` installed; Secure Boot compatible
- [x] CUDA available to PyTorch `2.7.1+cu126`
- [x] Qwen2.5-1.5B-Instruct loaded in FP16 on the GTX 1060
- [x] VRAM use about 2.9–3.1 GB of 6 GB
- [x] `/health` returns `200` with `model_loaded: true`
- [x] `/humanize` returns a real rewrite locally
- [x] `/ui` loads and can call the local API
- [x] Repeatable setup and launch scripts added under `scripts/`

The Colab path remains available for machines without a local GPU. Local inference
was validated on this laptop with short paragraphs; long documents are slower.

### Phase 7 — remote backend

- [x] OpenAI-compatible `RemoteHumanizer` using only the standard library
- [x] `INFERENCE_BACKEND=local|remote` selection
- [x] NVIDIA NIM defaults and configurable model, timeout, and extra body
- [x] Nemotron thinking disabled by default for faster edits
- [x] Offline request, auth, rate-limit, timeout, and malformed-response tests
- [x] Opt-in real-provider integration test
- [x] `scripts/run_remote.sh` launch helper
- [ ] Optional: rotate a fresh NIM key into `.env` and run the real remote test

The remote request path was validated end-to-end against a local
OpenAI-compatible test server. Live-provider validation requires your own key;
add it to `.env` before running `RUN_REMOTE_TESTS=1`.

## Limitations

- Instruction-following is probabilistic. Meaning, names, dates, and numbers can
  change despite the prompt. Review important text manually.
- The `<source_text>` tags organize the prompt but are not a security boundary
  against prompt injection.
- The model can produce awkward or repetitive text, especially on CPU fallback.
- The prototype has no persistent storage, user accounts, billing, or rate limits.
- Long documents are chunked in local mode. Cross-paragraph coherence can suffer.
- Remote mode requires internet access, a provider account, and may cost money.
  Submitted text leaves the machine and is subject to the provider's data policy.
- A larger remote model can improve prose, but it does not guarantee a particular
  AI-detector score.
- This is not a plagiarism checker, fact-checker, translator, or AI-detector
  bypass tool.
