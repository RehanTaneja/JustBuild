from __future__ import annotations

from pathlib import Path

from .models import BuildContext


def _build_root(context: BuildContext) -> Path:
    implementation = context.implementation
    if implementation and implementation.prototype_dir:
        return implementation.prototype_dir.parent
    return context.request.output_root


def write_final_report(context: BuildContext) -> Path:
    build_root = _build_root(context)
    build_root.mkdir(parents=True, exist_ok=True)
    path = build_root / "final_report.md"
    testing = context.testing
    evaluation = context.evaluation
    specification = context.specification

    report = f"""# Build Report

## Product

- Idea: {context.request.product_idea}
- Title: {specification.title if specification else "Unavailable"}
- Prototype: {context.implementation.prototype_dir if context.implementation else "Unavailable"}

## Test Results Summary

- Status: {"PASS" if testing and testing.passed else "FAIL"}
- Summary: {testing.summary if testing else "No testing summary available."}

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
