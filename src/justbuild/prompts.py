from __future__ import annotations

import json
from dataclasses import asdict

from .memory import build_memory_prompt_context
from .models import ArchitecturePlan, ArchitectureReview, BuildMemory, FailureReport, FixPlan, ImplementationArtifacts, ProductSpecification, TestResult


SPECIFICATION_SCHEMA = {
    "type": "object",
    "required": [
        "title",
        "product_summary",
        "requirements",
        "features",
        "user_stories",
        "api_contracts",
        "assumptions",
        "constraints",
        "missing_requirements",
    ],
}

ARCHITECTURE_SCHEMA = {
    "type": "object",
    "required": [
        "summary",
        "folder_structure",
        "services",
        "database_schema",
        "design_tradeoffs",
        "justification",
    ],
}

IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "required": ["notes", "files"],
}

TESTING_SCHEMA = {
    "type": "object",
    "required": ["unit_checks", "integration_checks", "failure_focus"],
}

EVALUATION_SCHEMA = {
    "type": "object",
    "required": [
        "code_quality",
        "maintainability",
        "scalability_risks",
        "security_concerns",
        "refactoring_opportunities",
        "technical_debt",
        "risk_assessment",
    ],
}

DEBUGGING_SCHEMA = {
    "type": "object",
    "required": [
        "file_changes",
        "root_cause",
        "strategy",
        "failure_groups",
        "priority_order",
    ],
}

ARCHITECTURE_REVIEW_SCHEMA = {
    "type": "object",
    "required": ["findings", "recommendations", "requires_refinement"],
}


def specification_system_prompt() -> str:
    return (
        "You are the JustBuild specification agent. Return valid JSON only. "
        "Do not include markdown, prose outside JSON, or code fences."
    )


def specification_user_prompt(idea: str, architecture_feedback: list[str] | None, memory: BuildMemory | None = None) -> str:
    feedback = architecture_feedback or []
    memory_block = _memory_prompt_block(memory)
    return (
        "Convert the product idea into a structured specification JSON.\n"
        f"Idea: {idea}\n"
        f"Architecture feedback to incorporate: {json.dumps(feedback)}\n"
        f"{memory_block}"
        "Return fields: title, product_summary, requirements, features, user_stories, "
        "api_contracts, assumptions, constraints, missing_requirements.\n"
        "Each list field must be an array of strings."
    )


def architecture_system_prompt() -> str:
    return (
        "You are the JustBuild architecture agent. Return valid JSON only and keep the design "
        "grounded in the current repository structure."
    )


def architecture_user_prompt(spec: ProductSpecification, memory: BuildMemory | None = None) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Create an architecture plan for this product specification.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"{memory_block}"
        "Return fields: summary, folder_structure, services, database_schema, "
        "design_tradeoffs, justification.\n"
        "Each list field must be an array of strings."
    )


def architecture_review_system_prompt() -> str:
    return (
        "You are the JustBuild architecture review agent. Return valid JSON only. "
        "Critique the proposed architecture against the product specification."
    )


def architecture_review_user_prompt(
    spec: ProductSpecification,
    architecture: ArchitecturePlan | None = None,
    memory: BuildMemory | None = None,
) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Review the architecture for this product specification.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2) if architecture else 'null'}\n"
        f"{memory_block}"
        "Return JSON with findings, recommendations, and requires_refinement.\n"
        "findings and recommendations must be arrays of strings. requires_refinement must be a boolean."
    )


def implementation_system_prompt() -> str:
    return (
        "You are the JustBuild implementation agent. Return valid JSON only. "
        "Generate a static browser prototype file bundle with production-style structure."
    )


def implementation_user_prompt(
    spec: ProductSpecification,
    architecture: ArchitecturePlan,
    failure_reports: list[FailureReport] | None,
    fix_plan: FixPlan | None = None,
    memory: BuildMemory | None = None,
) -> str:
    failures = [
        {"source": report.source, "summary": report.summary, "details": report.details}
        for report in (failure_reports or [])
    ]
    memory_block = _memory_prompt_block(memory)
    return (
        "Generate a static prototype bundle for this product.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Previous failures to fix: {json.dumps(failures, indent=2)}\n"
        f"Fix plan guidance: {json.dumps(asdict(fix_plan), indent=2) if fix_plan else 'null'}\n"
        f"{memory_block}"
        "Return JSON with:\n"
        '- notes: array of strings\n'
        '- files: object with keys index.html, styles.css, app.js, README.md\n'
        "The HTML must include the product title and a 'Feature Breakdown' section.\n"
        "The JS must include the phrase 'Generated Response'.\n"
        "The CSS must include a :root block.\n"
        "If a fix plan is provided, prioritize it over the prior bundle."
    )


def testing_system_prompt() -> str:
    return (
        "You are the JustBuild testing agent. Return valid JSON only. "
        "Produce a test checklist, not a narrative."
    )


def testing_user_prompt(spec: ProductSpecification, architecture: ArchitecturePlan, memory: BuildMemory | None = None) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Create a structured verification checklist for this generated prototype.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"{memory_block}"
        "Return JSON with unit_checks, integration_checks, failure_focus as arrays of strings."
    )


def evaluation_system_prompt() -> str:
    return (
        "You are the JustBuild evaluation agent. Return valid JSON only. "
        "Assess the build after implementation and testing."
    )


def evaluation_draft_system_prompt(draft_name: str) -> str:
    return (
        "You are the JustBuild evaluation draft agent. Return valid JSON only. "
        f"Produce the {draft_name} draft for the final evaluation."
    )


def evaluation_draft_user_prompt(
    draft_fields: list[str],
    spec: ProductSpecification,
    architecture: ArchitecturePlan,
    testing: TestResult,
    memory: BuildMemory | None = None,
) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Produce a partial evaluation draft.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Testing: {json.dumps(asdict(testing), indent=2, default=str)}\n"
        f"{memory_block}"
        f"Return only these fields as arrays of strings: {', '.join(draft_fields)}"
    )


def evaluation_user_prompt(
    spec: ProductSpecification,
    architecture: ArchitecturePlan,
    testing: TestResult,
    memory: BuildMemory | None = None,
) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Review this build and produce a structured evaluation report.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Testing: {json.dumps(asdict(testing), indent=2, default=str)}\n"
        f"{memory_block}"
        "Return JSON with code_quality, maintainability, scalability_risks, "
        "security_concerns, refactoring_opportunities, technical_debt, risk_assessment "
        "as arrays of strings."
    )


def debugging_system_prompt() -> str:
    return (
        "You are the JustBuild debugging agent. Return valid JSON only. "
        "Diagnose test and implementation failures, classify them, and propose a fix plan."
    )


def debugging_user_prompt(
    failure_reports: list[FailureReport],
    implementation: ImplementationArtifacts | None,
    testing: TestResult | None,
    specification: ProductSpecification | None,
    architecture: ArchitecturePlan | None,
    memory: BuildMemory | None = None,
) -> str:
    failures = [asdict(report) for report in failure_reports]
    implementation_data = asdict(implementation) if implementation else None
    testing_data = asdict(testing) if testing else None
    specification_data = asdict(specification) if specification else None
    architecture_data = asdict(architecture) if architecture else None
    memory_block = _memory_prompt_block(memory)
    return (
        "Read the current build failures and produce a fix plan.\n"
        f"Failure reports: {json.dumps(failures, indent=2, default=str)}\n"
        f"Implementation artifacts: {json.dumps(implementation_data, indent=2, default=str)}\n"
        f"Testing result: {json.dumps(testing_data, indent=2, default=str)}\n"
        f"Specification: {json.dumps(specification_data, indent=2, default=str)}\n"
        f"Architecture: {json.dumps(architecture_data, indent=2, default=str)}\n"
        f"{memory_block}"
        "Classify failures using only: missing_file, logic_error, schema_mismatch, content_mismatch, llm_output_invalid.\n"
        "Return JSON with file_changes, root_cause, strategy, failure_groups, priority_order.\n"
        "All list fields must be arrays of strings. root_cause and strategy must be concise strings."
    )


def _memory_prompt_block(memory: BuildMemory | None) -> str:
    failures, successes = build_memory_prompt_context(memory)
    return (
        f"Common past failures: {json.dumps(failures)}\n"
        f"Previously successful patterns: {json.dumps(successes)}\n"
    )
