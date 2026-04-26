# JustBuild Operator Runbook

This runbook is the operator-facing guide for running JustBuild as a Codex competition submission. It focuses on repeatable setup, supported execution paths, artifact inspection, and the architectural choices a reviewer or operator should understand before running the system.

## System Overview

JustBuild is a workflow-driven prototype builder with these stages:

1. `load_memory`
2. `specification`
3. `architecture_plan`
4. `architecture_review`
5. `planning_refinement_gate`
6. `implementation`
7. `testing`
8. `debugging` when testing fails
9. `evaluation_quality`, `evaluation_risk`, `evaluation_security`
10. `evaluation_merge`
11. `persist_outputs`
12. optional `publish`

Important architectural choices in the current build:

- the core runtime is Python standard-library based
- all planning stages use structured outputs and local validation
- architecture planning and architecture review are separate steps
- testing is deterministic where possible and not purely LLM judged
- local OpenAI-compatible endpoints are capability-aware rather than treated as generic text-only servers

## What Reviewers Should Expect

A successful run should produce:

- a prototype under `build_output/<idea-slug>/prototype/`
- `build_summary.json`
- `final_report.md`
- `run/build.log`
- `run/build_events.jsonl`
- `run/build_summary.partial.json`

Optional publishing adds:

- a standalone GitHub repository
- iteration snapshots
- a publish-ready README

## Prerequisites

- Python 3.11 or newer
- Node.js available on `PATH`
- `pip`
- optional: Playwright for browser validation
- optional: `gh` for GitHub publishing
- for cloud runs: a valid LLM API key
- for local runs: an OpenAI-compatible server such as Ollama, LM Studio, or vLLM

## Clean Setup

Recommended online setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
```

Offline or restricted-environment fallback:

```bash
python3 -m venv .venv-system --system-site-packages
. .venv-system/bin/activate
python -m pip install -e . --no-build-isolation
```

Why this fallback exists:

- fresh Python 3.12 virtual environments may not include `setuptools`
- network-restricted environments may block build dependency downloads
- the fallback reuses host tooling and keeps the install path simple

## Baseline Verification

Run these before a live build:

```bash
python3 --version
node --version
justbuild --help
PYTHONPATH=src pytest -q
```

Expected outcomes:

- the CLI help banner renders cleanly
- the test suite passes
- Python and Node are visible on `PATH`

## Supported LLM Paths

### Cloud providers

Supported:

- OpenAI
- Anthropic
- Gemini

Characteristics:

- strongest structured-output reliability
- best default choice for end-to-end demonstration runs

### Local or proxy OpenAI-compatible providers

Supported:

- Ollama
- LM Studio
- vLLM
- similar OpenAI-compatible servers

Characteristics:

- JustBuild auto-detects backend family from `base_url` and backend behavior
- it prefers the strongest structured-output strategy the backend supports
- for capable local backends, it can use tool-based structured output
- when unsupported features are detected, it downgrades and caches the safer strategy

Important operational note:

- strong local models can work well
- weak local models can still fail semantically even when structured-output transport is working correctly

## Cloud Run

Environment setup:

```bash
export JUSTBUILD_LLM_PROVIDER=openai
export JUSTBUILD_LLM_MODEL=gpt-4.1-mini
export JUSTBUILD_LLM_API_KEY=your_api_key
export JUSTBUILD_MEMORY_PATH=./build_output/build_memory.json
```

Example command:

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

Post-run checks:

- stdout prints a JSON summary
- `passed` is `true`
- `prototype_dir` exists
- `summary_path` and `final_report_path` exist
- `workflow_terminal_state` is `completed` or a publish-specific completion state

## Local Open-Source Run

Environment setup:

```bash
export JUSTBUILD_LLM_PROVIDER=openai_compatible
export JUSTBUILD_LLM_LOCAL_MODEL=llama3
export JUSTBUILD_LLM_BASE_URL=http://localhost:11434/v1
export JUSTBUILD_MEMORY_PATH=./build_output_local/build_memory.json
```

Example command:

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

Recommended preflight checks for Ollama:

```bash
ollama --version
curl http://localhost:11434/api/tags
```

What to inspect in the resulting summary:

- `llm_backend.backend_family`
- `llm_backend.structured_output_mode`
- `llm_backend.capabilities_probed`
- `llm_backend.capability_source`
- `llm_backend.capability_downgrade`

Those fields show whether the server was detected as Ollama or generic OpenAI-compatible and whether the run stayed on tool mode, schema mode, or downgraded to best-effort.

## Artifact Inspection

After a run, inspect:

- `build_output/<idea-slug>/prototype/`
- `build_output/<idea-slug>/build_summary.json`
- `build_output/<idea-slug>/final_report.md`
- `build_output/<idea-slug>/run/build.log`
- `build_output/<idea-slug>/run/build_events.jsonl`

Useful commands:

```bash
ls build_output/<idea-slug>/prototype
cat build_output/<idea-slug>/build_summary.json
cat build_output/<idea-slug>/final_report.md
tail -n 40 build_output/<idea-slug>/run/build.log
```

Key `build_summary.json` fields:

- `llm_backend`
- `testing_backend`
- `workflow_terminal_state`
- `node_runs`
- `testing.passed`
- `testing.failure_reports`
- `iterations`
- `github_publish`

## GitHub Publishing

Publishing requirements:

- `gh` installed
- `gh auth login` already completed
- `--publish-github` enabled on the CLI when running the build

Publishing behavior:

- creates a repository in the authenticated user account
- copies the prototype, final report, summary, and iteration history
- creates a small commit history instead of one unstructured dump

## Troubleshooting

### Missing backend configuration

Symptom:

```text
Configuration error: No LLM backend configured
```

Fix:

- cloud mode requires `--provider`, `--model`, and `--api-key`
- local mode requires `--provider openai_compatible`, `--local-model`, and `--base-url`

### Missing `node`

Symptom:

- JavaScript execution checks are skipped

Fix:

- install Node.js
- or pass `--node-bin <path>`

### Missing `pytest`

Symptom:

- Python execution checks are skipped

Fix:

- install `pytest`
- or pass `--pytest-bin <path>`

### Missing Playwright

Symptom:

- browser validation is skipped even when `--enable-playwright` is enabled

Fix:

```bash
python -m pip install playwright
python -m playwright install chromium
```

### Local model server not reachable

Symptom:

- transport errors on the first LLM-backed stage

Fix:

- confirm the server is running
- confirm `--base-url`
- verify the endpoint from the same shell
- for Ollama, run `curl http://localhost:11434/api/tags`

### Local model is reachable but still fails planning

Symptom:

- specification or architecture fields are empty or semantically weak
- repeated retries fail on required scalar fields like `product_summary` or `summary`

Fix:

- use a stronger local model
- prefer cloud providers for the most reliable demo path
- inspect `llm_backend.structured_output_mode` first to confirm transport worked
- if transport looks correct, treat the issue as model quality rather than backend wiring

### Python 3.12 editable install fails offline

Symptom:

- editable install fails because `setuptools` is missing
- build isolation tries to fetch packages over a blocked network

Fix:

- online: run `python -m pip install --upgrade pip setuptools`
- offline: use the `--system-site-packages` fallback shown above

## Submission Guidance

For a competition demo, the most reliable path is:

1. verify the test suite with `PYTHONPATH=src pytest -q`
2. run a cloud-backed build for the cleanest end-to-end demonstration
3. run a second build against a capable local backend to show provider flexibility
4. present `build_summary.json`, `final_report.md`, and `build.log` as the primary review artifacts

That combination shows both the system architecture and the operator experience clearly.
