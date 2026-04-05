# JustBuild Multi-Agent Prototype Builder

`justbuild` is a production-style, layered multi-agent system that takes a product idea and autonomously:

- plans the product,
- designs an architecture,
- generates a working prototype,
- validates the result,
- evaluates maintainability and risk,
- iterates when failures are detected.

The current implementation is intentionally dependency-light and uses Python's standard library so the system can run anywhere with `python3`.

## Layers

- `orchestration`: workflow control, milestone tracking, retries, and build summaries
- `planning`: product specification and architecture generation
- `implementation`: prototype file generation and refinement
- `testing`: validation, failure reporting, and test execution
- `evaluation`: maintainability, security, and scalability assessment
- `observability`: structured decision logging, timing, and iteration history

## Run

```bash
PYTHONPATH=src python3 -m justbuild "AI travel planner for remote teams"
```

Generated prototypes are written to `build_output/<idea-slug>/prototype`.

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
