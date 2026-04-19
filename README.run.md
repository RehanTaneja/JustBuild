# JustBuild Operator Runbook

This runbook documents the supported operator workflow for running JustBuild from an installed CLI, verifying the local environment, and exercising both cloud API and open-source model paths.

## Prerequisites

- Python 3.11 or newer
- Node.js available on `PATH`
- `pip`
- Optional: Playwright for browser validation
- Optional: `gh` for GitHub publishing
- For cloud API runs: a valid LLM API key
- For open-source runs: an OpenAI-compatible local endpoint such as Ollama, LM Studio, or vLLM

## Clean Setup

Recommended online setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
```

Offline or restricted-environment fallback used during readiness testing:

```bash
python3 -m venv .venv-system --system-site-packages
. .venv-system/bin/activate
python -m pip install -e . --no-build-isolation
```

Why the fallback exists:

- On Python 3.12, a fresh venv may not include `setuptools`
- In a network-restricted environment, `pip install -e .` may fail while trying to fetch build requirements
- The fallback reuses the host `setuptools` package and installs the project successfully without network access

Verification commands after install:

```bash
justbuild --help
python -m unittest discover -s tests -v
python -m pytest
```

## Automated Verification

Run these checks before attempting a live build:

```bash
python3 --version
node --version
python -m unittest discover -s tests -v
python -m pytest
justbuild --help
```

Expected outcomes:

- `python -m unittest discover -s tests -v` should pass
- `python -m pytest` should pass
- `justbuild --help` should show the installed CLI usage banner

## Cloud API Run

Set the minimum required environment:

```bash
export JUSTBUILD_LLM_PROVIDER=openai
export JUSTBUILD_LLM_MODEL=gpt-4.1-mini
export JUSTBUILD_LLM_API_KEY=your_api_key
export JUSTBUILD_MEMORY_PATH=./build_output/build_memory.json
```

Run a cloud-backed build:

```bash
justbuild "AI travel planner for remote teams" \
  --output-root ./build_output \
  --provider openai \
  --model gpt-4.1-mini \
  --api-key "$JUSTBUILD_LLM_API_KEY" \
  --pytest-bin pytest \
  --node-bin node \
  --max-workers 4 \
  --memory-path ./build_output/build_memory.json
```

What to verify after the run:

- JSON summary printed to stdout
- `passed` is `true`
- `prototype_dir` points to `build_output/<idea-slug>/prototype`
- `summary_path` points to `build_summary.json`
- `final_report_path` points to `final_report.md`
- `iterations` is at least `1`

## Open-Source Run

Set the minimum required environment for an OpenAI-compatible local endpoint:

```bash
export JUSTBUILD_LLM_PROVIDER=openai_compatible
export JUSTBUILD_LLM_LOCAL_MODEL=llama3
export JUSTBUILD_LLM_BASE_URL=http://localhost:11434/v1
export JUSTBUILD_MEMORY_PATH=./build_output_local/build_memory.json
```

Run against a local model server:

```bash
justbuild "AI travel planner for remote teams" \
  --output-root ./build_output_local \
  --provider openai_compatible \
  --local-model llama3 \
  --base-url http://localhost:11434/v1 \
  --pytest-bin pytest \
  --node-bin node \
  --max-workers 4 \
  --memory-path ./build_output_local/build_memory.json
```

Recommended local provider checks before running:

```bash
ollama --version
curl http://localhost:11434/api/tags
```

If you are using LM Studio or vLLM instead of Ollama, replace `--base-url` and `--local-model` with the values exposed by that server.

## Artifact Inspection

After a successful run, inspect:

- `build_output/<idea-slug>/prototype/`
- `build_output/<idea-slug>/build_summary.json`
- `build_output/<idea-slug>/final_report.md`
- the memory file passed with `--memory-path`

Useful checks:

```bash
ls build_output/<idea-slug>/prototype
cat build_output/<idea-slug>/build_summary.json
cat build_output/<idea-slug>/final_report.md
```

Important fields in `build_summary.json`:

- `llm_backend`
- `testing_backend`
- `workflow_terminal_state`
- `testing.passed`
- `testing.failure_reports`
- `iterations`

## Troubleshooting

### Missing backend config

Symptom:

```text
Configuration error: No LLM backend configured
```

Fix:

- Cloud mode requires `--provider`, `--model`, and `--api-key`
- Local mode requires `--provider openai_compatible`, `--local-model`, and `--base-url`

### Missing `node`

Symptom:

- deterministic JS validation is skipped

Fix:

- install Node.js
- or pass `--node-bin <path>`

### Missing `pytest`

Symptom:

- Python execution validation is skipped

Fix:

- install `pytest`
- or pass `--pytest-bin <path>`

### Missing Playwright

Symptom:

- browser validation is skipped even if `--enable-playwright` is set

Fix:

```bash
python -m pip install playwright
python -m playwright install chromium
```

### Local model server not reachable

Symptom:

- provider transport errors during the first LLM-backed agent step

Fix:

- confirm the local model server is running
- confirm the `--base-url` is correct
- verify the server is reachable from the same shell
- for Ollama, check `curl http://localhost:11434/api/tags`

### Python 3.12 fresh venv install fails offline

Symptom:

- editable install fails because `setuptools` is missing from the venv
- `pip install -e .` tries to fetch build dependencies and cannot reach the network

Fix:

- online: run `python -m pip install --upgrade pip setuptools` before `python -m pip install -e .`
- offline: use the fallback flow shown in `Clean Setup`
