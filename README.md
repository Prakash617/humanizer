# AI Humanizer

A beginner-friendly prototype that rewrites English text for better naturalness,
clarity, fluency, and readability while preserving meaning.

The project is developed on an Ubuntu laptop and runs the Hugging Face model in
Google Colab when a GPU is available. The laptop does **not** run model
inference.

> **Current status: Phase 2 — Colab model experiments.**
> The Colab notebook loads Qwen2.5-1.5B-Instruct, tests the humanization prompt,
> compares styles, evaluates outputs, and saves results. The reusable production
> inference module and FastAPI integration are added in later phases.

## Model decision (checked before use)

The notebook verifies these facts again at runtime:

| Item | Finding |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| License | **Apache-2.0** |
| Parameters | 1.54B total (1.31B non-embedding) |
| Weight type | BF16 |
| Approximate weights | ~3.1 GB, plus KV cache and runtime overhead |
| Context | 32,768 tokens; up to 8,192 generated tokens |
| Transformers | `>=4.37.0` required for Qwen2; Colab pins 5.21.0 |
| Generation defaults | temperature `0.7`, top-p `0.8`, top-k `20`, repetition penalty `1.1` |
| Fallback | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) |

The model is public and does not require a Hugging Face token. A free Colab T4 GPU
has enough memory. CPU inference is possible but slow, so the notebook reduces the
number of experiments on CPU.

Model card: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>

Fallback: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>


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
│   └── humanizer.py         # Model loading and rewriting logic (Phase 3)
├── tests/
│   ├── __init__.py          # Marks tests as a package
│   └── test_api.py          # API tests; model mocked in Phase 5
├── notebooks/
│   └── humanizer_experiments.ipynb  # Colab experiments (Phase 2)
├── requirements.txt         # Lightweight local dependencies only
├── requirements-colab.txt   # Model inference dependencies for Colab
├── .gitignore               # Files that must never enter Git
├── .env.example             # Safe configuration template; copied to .env
└── README.md                # This guide
```

In Phase 1, `schemas.py` and `humanizer.py` were placeholders. Phase 2 keeps the
production inference module empty while the notebook experiments. Phase 3 moves
the reusable logic into `app/humanizer.py`.

## Requirements

- Ubuntu 24.04 LTS (this setup works on nearby Ubuntu versions too)
- Python 3.11 or 3.12 (3.12 is recommended because it is common in Colab)
- Git
- Visual Studio Code
- Internet access for package installation

No NVIDIA/CUDA setup is needed on the laptop. Model weights are downloaded only in
the Colab runtime. The local virtual environment stays lightweight.

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

## Phase 2: Run the model experiments in Colab

The notebook is `notebooks/humanizer_experiments.ipynb`. It is a client of the Git
repository; it does not share Python memory with your laptop.

### What the notebook does

1. Checks the Colab Python and GPU.
2. Clones or updates the repository.
3. Installs the pinned Colab inference requirements.
4. Verifies the model card, license, revision, and repository size.
5. Loads Qwen2.5-1.5B-Instruct with the official chat template.
6. Runs a basic generation test.
7. Defines and tests `humanize_text(text, style="natural")`.
8. Rewrites several sample paragraphs.
9. Compares all five styles on one factual paragraph.
10. Computes automatic preservation/readability signals.
11. Provides a human review rubric.
12. Saves results without model weights.

### 1. Push the Phase 2 files

From the laptop, after configuring your GitHub remote:

```bash
cd ~/Desktop/program/ai/humanizer
source .venv/bin/activate
python -m pytest -q
git add .
git commit -m "feat: add Phase 2 Colab experiments"
git push -u origin main
```

If this is the first push and no remote exists:

```bash
git remote add origin git@github.com:YOUR_USERNAME/humanizer.git
git push -u origin main
```

### 2. Open the notebook in Colab

Use **File > Open notebook > GitHub**, enter your repository, and select
`notebooks/humanizer_experiments.ipynb`. Uploading the `.ipynb` file manually also
works.

In the notebook, edit:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/humanizer.git"
```

Use a public repository for the simplest setup. Do not paste a GitHub token into a
notebook cell.

### 3. Select a GPU

In Colab: **Runtime > Change runtime type > T4 GPU** (or another available GPU).
If no GPU is available, the notebook can run on CPU, but it is much slower and runs
fewer samples.

### 4. Run the cells from top to bottom

Do not skip the model-card check. If Colab asks you to restart the runtime after
installing packages, restart and run from the top. Installed packages and model
weights do not survive a runtime reset.

### Dependency choices

`requirements-colab.txt` pins:

```text
transformers==5.21.0
accelerate==1.15.0
jinja2==3.1.6
textstat==0.7.13
```

PyTorch is **not** pinned there. Colab normally supplies a CUDA-matched PyTorch
build. Replacing it with an arbitrary wheel is a common cause of
`torch.cuda.is_available() == False`. The notebook checks for PyTorch and installs
an unpinned fallback only when it is missing. Transformers 5.21 is tested with
`torch>=2.6`; start a fresh GPU runtime if yours is older. Model weights are cached
in `/content/hf_cache`, outside Git.

### Expected behavior

- The model-card cell prints `Apache-2.0`, `Gated: False`, and about `3.1 GB`.
- The model loads in BF16 on a T4 GPU.
- The basic prompt produces a short English answer.
- Each sample produces a rewritten paragraph, not a label or explanation.
- The style comparison changes tone while keeping the same names and numbers.
- Results are written to `/content/humanizer_results/<timestamp>/`.

If the 1.5B model cannot load, the notebook retries with
`Qwen/Qwen2.5-0.5B-Instruct`. The fallback is less capable, so review it more
carefully.

### Model evaluation checklist

For every source/output pair:

- [ ] Meaning is preserved, including qualifications such as “may” and “often”.
- [ ] Names, dates, numbers, units, quotations, and technical terms are unchanged.
- [ ] No facts, examples, citations, or statistics were invented.
- [ ] The point of view and level of certainty are preserved.
- [ ] Readability improved without oversimplifying the content.
- [ ] The requested style is recognizable but not exaggerated.
- [ ] The output is coherent, complete, and free of obvious repetition.
- [ ] The output contains only the rewrite, with no commentary or heading.
- [ ] A human would publish it after a light edit.

The automatic table flags missing/extra numbers and missing capitalized terms, but
these are **advisory**. A missing name can be a legitimate rewording; an extra
number is a possible fabrication. Manual review is mandatory.

### Style definitions used in the notebook

- `natural`: polished human-editor English with varied sentence lengths.
- `academic`: formal, precise, restrained, and appropriately qualified.
- `professional`: concise, direct, and workplace-appropriate.
- `simple`: plain wording and short sentences without adding explanations.
- `friendly`: warm and conversational without becoming overly familiar.

### Limits and safety

Instruction-following is probabilistic. The prompt cannot guarantee that every
fact, name, or number is preserved. The `<source_text>` tags organize the prompt
but are not a security boundary against prompt injection.

This project improves writing quality. It does not claim to guarantee a particular
AI-detector result, and it must not be presented as a detector-bypass tool.

For inputs up to 8,000 tokens, the notebook splits text at paragraph/sentence
boundaries into chunks of at most 1,400 tokens. Above 8,000 tokens it raises a clear
error instead of silently truncating. Phase 3 will move this logic into the
production inference module and harden it.

### Saving results

Results are saved locally inside the Colab runtime automatically. Optionally run
the commented Drive or download cells to copy the results to persistent storage.
Model weights, Hugging Face caches, and Colab packages are never saved in Git.

### Phase 2 checkpoint

- [ ] Notebook opens in Colab
- [ ] GPU check runs
- [ ] Repository clone/update succeeds
- [ ] Pinned dependencies install
- [ ] Model card check reports Apache-2.0
- [ ] Model loads (1.5B or documented fallback)
- [ ] Basic generation works
- [ ] At least one sample is humanized
- [ ] Styles are compared
- [ ] Automatic checks and manual review are completed
- [ ] Results are saved outside the repository
- [ ] `git status` on the laptop shows no weights or experiment outputs

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

### Colab reports `CUDA out of memory`

1. Use **Runtime > Restart runtime**.
2. Run the notebook from the top.
3. Let the notebook fall back to `Qwen/Qwen2.5-0.5B-Instruct`.
4. Do not load two models in the same session.

### Colab says CUDA is unavailable

- Confirm the runtime type is GPU.
- Do not reinstall an arbitrary PyTorch CUDA wheel. Restart and let Colab provide
  its matched build.
- Re-run the dependency-check cell and print `torch.version.cuda`.

### Model download fails in Colab

- Check the internet connection and free disk space.
- Confirm the repository URL and branch in the clone cell.
- The Qwen model is public. An invalid `HF_TOKEN` can still cause authentication
  errors; remove the secret and retry.
- Re-run the loading cell after a partial download; Hugging Face may resume it.

### `qwen2`, `apply_chat_template`, or import errors

Restart the runtime and reinstall the pinned Colab requirements. Qwen2 requires
Transformers 4.37+, and mixing versions in one runtime causes confusing errors.

### Long input is rejected

This is deliberate for the Phase 2 prototype: it avoids silent truncation. Split
the document into sections. Production-safe chunking will be implemented in
`app/humanizer.py` in Phase 3.

## Next phase

Phase 3 will move the tested inference logic into `app/humanizer.py`, add the
production `humanize_text(text, style="natural")` interface, harden validation and
error handling, and keep model loading separate from the FastAPI routes. The API
itself starts in Phase 4.
