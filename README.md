# JustBuild Multi-Agent Prototype Builder

`justbuild` is a production-style, layered multi-agent system that takes a product idea and autonomously:

- plans the product,
- designs an architecture,
- generates a working prototype,
- validates the result,
- diagnoses failures with a debugging agent,
- evaluates maintainability and risk,
- iterates when failures are detected.

The current implementation is intentionally dependency-light and uses Python's standard library so the system can run anywhere with `python3`.

## LLM Backends

Agents are now LLM-driven. Every planning, architecture, implementation, testing, and evaluation step builds a prompt, calls an `LLMClient`, requires JSON-only output, and validates the response before continuing.

Supported backends in v1:

- OpenAI
- Anthropic
- Gemini
- OpenAI-compatible endpoints for local/open-source models such as Ollama, vLLM, and LM Studio

Local support is endpoint-based. The project does not load model weights in-process.

## Layers

- `orchestration`: workflow control, milestone tracking, retries, and build summaries
- `planning`: product specification and architecture generation
- `implementation`: prototype file generation and refinement
- `testing`: validation, failure reporting, and test execution
- `debugging`: LLM-driven failure diagnosis and fix-plan generation
- `evaluation`: maintainability, security, and scalability assessment
- `observability`: structured decision logging, timing, and iteration history

## Testing Layer

The testing layer now combines LLM-generated test planning with deterministic runtime validation:

- file and content sanity checks
- `pytest` execution for Python-side validation
- `node` execution for generated browser JavaScript through a DOM harness
- HTML structure validation
- API contract schema validation
- optional Playwright browser verification

## Run

```bash
PYTHONPATH=src python3 -m justbuild "AI travel planner for remote teams" \
  --provider openai \
  --model gpt-4.1-mini \
  --api-key "$JUSTBUILD_LLM_API_KEY" \
  --pytest-bin pytest \
  --node-bin node
```

Generated prototypes are written to `build_output/<idea-slug>/prototype`.

OpenAI-compatible local model example:

```bash
PYTHONPATH=src python3 -m justbuild "AI travel planner for remote teams" \
  --provider openai_compatible \
  --local-model llama3 \
  --base-url http://localhost:11434/v1
```

Environment variable equivalents:

```bash
export JUSTBUILD_LLM_PROVIDER=openai
export JUSTBUILD_LLM_MODEL=gpt-4.1-mini
export JUSTBUILD_LLM_API_KEY=your_api_key
```

```bash
export JUSTBUILD_LLM_PROVIDER=openai_compatible
export JUSTBUILD_LLM_LOCAL_MODEL=llama3
export JUSTBUILD_LLM_BASE_URL=http://localhost:11434/v1
```

Optional browser validation:

```bash
export JUSTBUILD_ENABLE_PLAYWRIGHT=1
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Scaling Direction

The codebase is designed so the current in-process agents can later be replaced with:

- remote LLM-backed services,
- queue-driven workers,
- isolated sandboxed code execution,
- repository-aware implementation agents,
- enterprise workflow orchestration and policy gates.
