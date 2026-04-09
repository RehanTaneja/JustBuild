from __future__ import annotations

import json
from dataclasses import asdict

from .models import ArchitecturePlan, FailureReport, ProductSpecification, TestResult


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


def specification_system_prompt() -> str:
    return (
        "You are the JustBuild specification agent. Return valid JSON only. "
        "Do not include markdown, prose outside JSON, or code fences."
    )


def specification_user_prompt(idea: str, architecture_feedback: list[str] | None) -> str:
    feedback = architecture_feedback or []
    return (
        "Convert the product idea into a structured specification JSON.\n"
        f"Idea: {idea}\n"
        f"Architecture feedback to incorporate: {json.dumps(feedback)}\n"
        "Return fields: title, product_summary, requirements, features, user_stories, "
        "api_contracts, assumptions, constraints, missing_requirements.\n"
        "Each list field must be an array of strings."
    )


def architecture_system_prompt() -> str:
    return (
        "You are the JustBuild architecture agent. Return valid JSON only and keep the design "
        "grounded in the current repository structure."
    )


def architecture_user_prompt(spec: ProductSpecification) -> str:
    return (
        "Create an architecture plan for this product specification.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        "Return fields: summary, folder_structure, services, database_schema, "
        "design_tradeoffs, justification.\n"
        "Each list field must be an array of strings."
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
) -> str:
    failures = [
        {"source": report.source, "summary": report.summary, "details": report.details}
        for report in (failure_reports or [])
    ]
    return (
        "Generate a static prototype bundle for this product.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Previous failures to fix: {json.dumps(failures, indent=2)}\n"
        "Return JSON with:\n"
        '- notes: array of strings\n'
        '- files: object with keys index.html, styles.css, app.js, README.md\n'
        "The HTML must include the product title and a 'Feature Breakdown' section.\n"
        "The JS must include the phrase 'Generated Response'.\n"
        "The CSS must include a :root block."
    )


def testing_system_prompt() -> str:
    return (
        "You are the JustBuild testing agent. Return valid JSON only. "
        "Produce a test checklist, not a narrative."
    )


def testing_user_prompt(spec: ProductSpecification, architecture: ArchitecturePlan) -> str:
    return (
        "Create a structured verification checklist for this generated prototype.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        "Return JSON with unit_checks, integration_checks, failure_focus as arrays of strings."
    )


def evaluation_system_prompt() -> str:
    return (
        "You are the JustBuild evaluation agent. Return valid JSON only. "
        "Assess the build after implementation and testing."
    )


def evaluation_user_prompt(
    spec: ProductSpecification,
    architecture: ArchitecturePlan,
    testing: TestResult,
) -> str:
    return (
        "Review this build and produce a structured evaluation report.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Testing: {json.dumps(asdict(testing), indent=2, default=str)}\n"
        "Return JSON with code_quality, maintainability, scalability_risks, "
        "security_concerns, refactoring_opportunities, technical_debt, risk_assessment "
        "as arrays of strings."
    )
