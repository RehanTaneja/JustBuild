# JustBuild Multi-Agent Prototype Builder

`justbuild` is a multi-agent prototype generator built for turning a plain-language product idea into a runnable prototype, validating it, and packaging the build artifacts in a way that is easy to inspect. It is designed as a Codex competition submission, so the emphasis is on clear orchestration, end-to-end autonomy, strong observability, and practical operator ergonomics instead of a single demo script.

## What It Does

Given one idea, JustBuild will:

- generate a structured product specification
- generate an architecture plan and a parallel architecture review
- create an implementation plan and generate the prototype file-by-file
- run deterministic validation and optional browser checks
- produce debugging guidance when testing fails
- generate evaluation reports for quality, maintainability, risk, and security
- persist machine-readable and human-readable build artifacts
- optionally publish successful outputs to a standalone GitHub repository

The default output for each build includes:

- a generated prototype directory
- `build_summary.json`
- `final_report.md`
- run logs and event logs
- iteration history
- optional GitHub publish artifacts

## Core Features

### Multi-agent workflow

- `SpecificationAgent`: turns the idea into a structured product spec
- `ArchitectureAgent`: produces an implementation-oriented architecture plan
- `ArchitectureReview`: checks for blocker-level planning issues before implementation continues
- `ImplementationAgent`: plans files and generates the prototype file-by-file
- `TestingAgent`: combines LLM-authored checks with deterministic runtime validation
- `DebuggingAgent`: proposes a fix plan when testing fails
- `EvaluationAgent`: produces draft evaluations for quality, risk, and security, then merges them
- `OrchestratorAgent`: manages workflow execution, retries, milestones, and artifact persistence

### Structured-output LLM layer

- provider-aware structured-output handling for OpenAI, Anthropic, Gemini, and OpenAI-compatible endpoints
- JSON schema validation for planning outputs
- provider-specific extraction for Anthropic tool results and Gemini structured JSON
- capability-aware structured-output selection for local backends
- automatic backend detection for OpenAI-compatible endpoints, including Ollama-style probing
- dynamic downgrade from tool-based structured output to schema-based or best-effort JSON when a local backend lacks a feature

### Testing and validation

- file existence and content sanity checks
- HTML and schema validation
- JavaScript execution validation through Node
- Python execution validation through `pytest`
- optional Playwright browser verification
- failure reports that feed back into debugging and retry loops

### Observability and artifacts

- live CLI progress logs
- structured event logs in `build_events.jsonl`
- human-readable log output in `build.log`
- partial build snapshots in `build_summary.partial.json`
- final machine-readable summary in `build_summary.json`
- final human-readable report in `final_report.md`

### Publishing

- optional GitHub publishing through the authenticated GitHub CLI
- per-build repository creation
- commit history that reflects build iterations
- bundled publish artifacts including the prototype, reports, and iteration history

## Architecture Choices

These are intentional architectural decisions in the current submission:

- Python standard library first: the core orchestration, logging, transport, workflow runtime, and validation layers are standard-library based
- strict stage separation: specification, architecture, implementation, testing, debugging, and evaluation are modeled as separate agents with explicit handoffs
- workflow graph over ad hoc control flow: retries, branching, and refinement are expressed through a DAG-style runtime instead of scattered conditionals
- structured outputs over free-form prose: the planning stages expect machine-validated JSON so downstream stages are deterministic
- capability-aware local backend handling: local and proxy model servers are not treated as one flat `best_effort` bucket
- deterministic validation after generation: generated artifacts are checked with concrete runtime and structural validation, not only by an LLM
- persistent build memory: previous failures and successful patterns can influence later runs through `build_memory.json`
- optional parallelism: architecture generation and review, evaluation drafts, and some validation work can run concurrently

## Supported LLM Backends

Supported backends:

- OpenAI
- Anthropic
- Gemini
- OpenAI-compatible endpoints such as Ollama, vLLM, and LM Studio

Important note on local models:

- backend capability detection is automatic
- JustBuild will try the strongest structured-output mode the backend supports
- strong local models behave much better than weak ones
- weak local models can still fail semantically even when transport and schema handling are correct

## Repository Structure

- [src/justbuild/orchestrator.py](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/orchestrator.py): workflow assembly, retries, milestones, and state transitions
- [src/justbuild/llm.py](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/llm.py): provider transport, structured-output handling, and local backend capability detection
- [src/justbuild/prompts.py](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/prompts.py): stage prompts and JSON schema definitions
- [src/justbuild/validation.py](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/validation.py): schema normalization and response validation
- [src/justbuild/observability.py](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/observability.py): logs, summaries, and run artifacts
- [src/justbuild/agents](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/src/justbuild/agents): agent implementations for each workflow stage
- [tests](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/tests): transport, CLI, workflow, publishing, agent, and validation regression coverage

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
```

Cloud example:

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

Local OpenAI-compatible example:

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

Generated prototypes land under `build_output/<idea-slug>/prototype` or `build_output_local/<idea-slug>/prototype`.

## Operator Runbook

The full operator guide is in [README.run.md](/Users/rehantaneja/Documents/MyDoc/Carreer/Projects/JustBuild/README.run.md). It covers:

- environment setup
- verification commands
- cloud and local run flows
- artifact inspection
- local-backend capability behavior
- troubleshooting for common failure modes

## Verification

```bash
PYTHONPATH=src pytest -q
```

Current repository status for this submission:

- structured-output transport coverage across supported providers
- local backend capability detection and downgrade behavior covered in tests
- end-to-end fake-LLM workflow tests
- CLI, publishing, workflow, and validation regression coverage

## Submission Notes

This codebase is optimized for demonstrable autonomy and inspectability:

- every major workflow decision is logged
- artifacts are saved even when a run fails
- the system can be exercised with cloud models or local model servers
- the architecture is intentionally modular so stronger providers, stricter enforcement, or richer prototype targets can be added without replacing the whole pipeline
