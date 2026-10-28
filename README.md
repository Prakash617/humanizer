# AI Humanizer

A beginner-friendly prototype that rewrites English text for better naturalness,
clarity, fluency, and readability while preserving meaning.

The project is developed on an Ubuntu laptop and runs the Hugging Face model in
Google Colab when a GPU is available. The laptop does **not** run model
inference.

> **Current status: Phase 1 — local development setup only.**
> The model, inference function, full API, and automated model tests are added in
> later phases. Do not proceed to Phase 2 until the Phase 1 checkpoint passes.

## How the system will work

```text
Ubuntu laptop (VS Code)
        |
        |  edit, test lightweight API code, git commit/push
        v
   Git repository
        |
        |  git clone / git pull inside Colab
        v
Google Colab runtime (temporary)
        |
        |  FastAPI routes  --->  app/humanizer.py  --->  Qwen model on GPU
        v
  HTTP API (temporary tunnel only)
```

Model inference is kept separate from API routing:

- `app/humanizer.py` owns model loading and text rewriting.
- `app/schemas.py` owns request and response validation.
- `app/main.py` owns HTTP routes and application lifecycle.

A future frontend can call the API over HTTP. It must not import Python model
objects directly.

## Project structure

```text
humanizer/
├── app/
│   ├── __init__.py          # Package marker and version number
│   ├── main.py              # FastAPI app and HTTP routes (Phase 4)
│   ├── schemas.py           # Pydantic request/response models (Phase 4)
│   └── humanizer.py         # Model loading and rewriting logic (Phases 2-3)
├── tests/
│   ├── __init__.py          # Marks tests as a package
│   └── test_api.py          # API tests; model mocked in Phase 5
├── notebooks/
│   └── humanizer_experiments.ipynb  # Colab experiments (Phase 2)
├── requirements.txt         # Lightweight local dependencies only
├── .gitignore               # Files that must never enter Git
├── .env.example             # Safe configuration template; copied to .env
└── README.md                # This guide
```

In Phase 1, `schemas.py` and `humanizer.py` are deliberately placeholders. They
exist now so the architecture is clear and no model code gets mixed into the API
entry point later.

## Requirements

- Ubuntu 24.04 LTS (this setup works on nearby Ubuntu versions too)
- Python 3.11 or 3.12 (3.12 is recommended because it is common in Colab)
- Git
- Visual Studio Code
- Internet access for package installation

No NVIDIA/CUDA setup is needed on the laptop. No model weights are downloaded in
Phase 1.

## Phase 1 setup

Run these commands in a terminal.

### 1. Verify the tools

```bash
python3 --version
git --version
code --version
```

Expected: Python 3.x, a Git version, and a VS Code version. If `code` is missing,
open VS Code and run **Shell Command: Install 'code' command in PATH** from the
Command Palette.

Ubuntu 24.04 normally provides Python 3.12. If `python3 --version` reports a
different version, install the matching venv package:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git
```

### 2. Create and enter the project directory

```bash
mkdir -p ~/Desktop/program/ai/humanizer
cd ~/Desktop/program/ai/humanizer
```

### 3. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now begin with `(.venv)`. To leave the environment later:

```bash
deactivate
```

Never commit `.venv/`; it is already ignored.

### 4. Install lightweight dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Pydantic, configuration helpers, pytest, and
httpx2 (the HTTP client used by FastAPI's `TestClient`). It intentionally does
**not** install PyTorch, Transformers, Accelerate,
or CUDA. Those are Colab inference dependencies and will be handled in Phase 2.

### 5. Create local configuration

```bash
cp .env.example .env
```

Edit `.env` only when you need local values. The real `.env` file is ignored by
Git. Never put tokens, passwords, or private keys in `.env.example` or any source
file.

## Verify Phase 1

### Run the smoke test

```bash
python -m pytest -q
```

Expected:

```text
1 passed
```

The Phase 1 test only checks that the FastAPI scaffold imports and responds. It
does not download or load a model.

### Start the scaffold API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> in a browser, or run this in a second terminal:

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{
  "name": "AI Humanizer API",
  "version": "0.1.0",
  "status": "scaffold"
}
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

Stop the server with `Ctrl+C`. This `status: "scaffold"` response is temporary;
the real `/health` endpoint is added in Phase 4.

## Initialize Git

Run from the project root with the virtual environment active or inactive:

```bash
git init -b main
git add .
git status
git commit -m "chore: scaffold Phase 1 project"
```

Before committing, confirm that `.venv/` and `.env` are not listed by
`git status`. Only source, tests, documentation, and safe configuration files
should be staged.

If Git complains that your identity is unknown:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Add a remote repository later

Create an empty repository on GitHub (do not initialize it with a README), then:

```bash
git remote add origin git@github.com:YOUR_USERNAME/humanizer.git
git push -u origin main
```

Use SSH keys or a credential manager. Do not paste a personal access token into a
URL, source file, notebook, or screenshot.

## What stays out of Git

`.gitignore` excludes:

- virtual environments
- Python and pytest caches
- `.env` and other local secret files
- Hugging Face caches
- model weights (`.safetensors`, `.bin`, `.pt`, `.pth`, `.gguf`, and similar)
- experiment outputs and logs
- notebook checkpoints and Colab drive mounts
- editor and operating-system files

If a secret is ever committed, rotate that secret immediately. Deleting it in a
later commit is not enough because it remains in Git history.

## Important Colab facts

Google Colab runtimes are temporary. When a runtime disconnects or is recycled:

- installed packages are lost
- downloaded model weights and the Hugging Face cache are lost
- the running FastAPI server stops
- any tunnel URL stops working

Only source code in Git, plus results explicitly saved to Google Drive or
downloaded, survive. Do not treat a Colab URL as a permanent public service. Do
not expose an unauthenticated inference endpoint to the internet.

## Phase 1 checkpoint

Confirm all of these before moving on:

- [ ] `python3 --version` works and the project uses a Python 3.11/3.12
      virtual environment
- [ ] `.venv/` exists and activates without errors
- [ ] `pip install -r requirements.txt` completes
- [ ] `python -m pytest -q` reports `1 passed`
- [ ] `uvicorn app.main:app --reload` starts and `/` returns the scaffold JSON
- [ ] `/docs` opens
- [ ] `git status` does not include `.venv/` or `.env`
- [ ] the initial commit exists
- [ ] no model weights or CUDA packages were installed locally

## Troubleshooting

### `ensurepip is not available` or `No module named venv`

Install the matching venv package, then recreate the environment:

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

### `pip: command not found` or installing into the wrong Python

Confirm the environment is active:

```bash
which python
python -m pip --version
```

Use `python -m pip` instead of bare `pip` to avoid installing into the system
Python.

### `ModuleNotFoundError: No module named 'app'`

Run pytest and Uvicorn from the project root (`~/Desktop/program/ai/humanizer`),
not from inside `app/` or `tests/`.

### Port 8000 is already in use

Use another port:

```bash
uvicorn app.main:app --reload --port 8001
```

### `code: command not found`

In VS Code, open the Command Palette and run
**Shell Command: Install 'code' command in PATH**, then reopen the terminal.

### Git commit fails because of missing identity

Set `user.name` and `user.email` as shown above, then commit again.

### A very new Python version causes package problems

Use Python 3.12 for this project. Model libraries often publish wheels later for
brand-new Python releases, so matching Colab's Python version avoids surprises.

## Next phase

Phase 2 will add and explain the Colab notebook: runtime/GPU checks, compatible
inference packages, repository clone, model-card review, model loading,
generation experiments, evaluation, and safe result saving. No model will be
loaded until that phase.
