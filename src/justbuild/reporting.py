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

    report = f"""# Build Report

## Product

- Idea: {context.request.product_idea}
- Title: {specification.title if specification else "Unavailable"}
- Prototype: {context.implementation.prototype_dir if context.implementation else "Unavailable"}
- LLM Backend: {context.request.llm_backend_type or "Unavailable"} / {context.request.llm_provider or "Unavailable"} / {context.request.llm_model or "Unavailable"}

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
