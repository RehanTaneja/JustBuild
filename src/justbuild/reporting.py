from __future__ import annotations

from pathlib import Path

from .models import BuildContext

"""
This file is a companion for observability.py. The final function is observability.py is a raw JSON data dump.
This file makes a human readable final report in markdown.
"""

# Decides where the final report should be saved.
def _build_root(context: BuildContext) -> Path:
    implementation = context.implementation
    if implementation and implementation.prototype_dir: # If code was generated
        return implementation.prototype_dir.parent # save report next to it
    return context.request.output_root # otherwise in the user given path

# Writes final markdown report.
def write_final_report(context: BuildContext) -> Path:
    build_root = _build_root(context)
    build_root.mkdir(parents=True, exist_ok=True)
    path = build_root / "final_report.md"
    testing = context.testing
    debugging = context.debugging
    evaluation = context.evaluation
    specification = context.specification
    memory = context.memory
    github_publish = context.github_publish

    report = f"""# Build Report

## Product

- Idea: {context.request.product_idea}
- Title: {specification.title if specification else "Unavailable"}
- Prototype: {context.implementation.prototype_dir if context.implementation else "Unavailable"}
- LLM Backend: {context.request.llm_backend_type or "Unavailable"} / {context.request.llm_provider or "Unavailable"} / {context.request.llm_model or "Unavailable"}
- Max workers: {context.request.max_workers}
- Workflow terminal state: {context.workflow_terminal_state or "Unavailable"}
- Workflow node runs: {len(context.node_runs)}

## Test Results Summary

- Status: {"PASS" if testing and testing.passed else "FAIL"}
- Summary: {testing.summary if testing else "No testing summary available."}
- Execution checks: {len(testing.execution_results) if testing else 0}
- Schema checks: {len(testing.schema_results) if testing else 0}
- Browser checks: {len(testing.browser_results) if testing else 0}
- Skipped checks: {len(testing.skipped_checks) if testing else 0}

## Self-Healing

- Fix plan used: {"YES" if debugging else "NO"}
- Root cause: {debugging.root_cause if debugging else "No debugging diagnosis recorded."}
- Strategy: {debugging.strategy if debugging else "No debugging strategy recorded."}

## Learning Carried Forward

{_render_bullets(_memory_learning_lines(memory))}

## Publishing

{_render_bullets(_publishing_lines(github_publish))}

## Risk Assessment

{_render_bullets(evaluation.risk_assessment if evaluation else ["No risk assessment available."])}

## Technical Debt

{_render_bullets(evaluation.technical_debt if evaluation else ["No technical debt report available."])}

## Next Iteration Roadmap

{_render_bullets([
    "Replace mock API seams with real backend services and persistence.",
    "Add authentication, authorization, and audit controls for production usage.",
    "Move orchestration into a durable queue-backed workflow runtime.",
    "Introduce browser automation, static analysis, and security scanning gates.",
])}
"""
    path.write_text(report, encoding="utf-8")
    context.final_report_path = path
    return path


def _render_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _memory_learning_lines(memory) -> list[str]:
    if memory is None:
        return ["No persistent build memory recorded yet."]

    lines = [f"Past builds remembered: {len(memory.past_builds)}"]
    if memory.failure_patterns:
        lines.append(f"Tracked failure pattern groups: {', '.join(sorted(memory.failure_patterns))}")
    if memory.successful_patterns:
        lines.append(f"Tracked successful pattern groups: {', '.join(sorted(memory.successful_patterns))}")
    if len(lines) == 1:
        lines.append("Persistent memory is enabled, but no stable patterns have been recorded yet.")
    return lines


def _publishing_lines(github_publish) -> list[str]:
    if github_publish is None or not github_publish.enabled:
        return ["GitHub publishing was not enabled for this build."]
    if github_publish.published:
        return [
            f"Published to: {github_publish.repo_url}",
            f"Repository: {github_publish.repo_full_name}",
            f"Branch: {github_publish.branch or 'main'}",
            f"Commits created: {len(github_publish.commits)}",
        ]
    return [
        "GitHub publishing was attempted but did not complete.",
        f"Failure reason: {github_publish.failure_reason or 'Unknown publishing failure.'}",
    ]
