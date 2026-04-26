from __future__ import annotations

import json
from dataclasses import asdict

from .memory import build_memory_prompt_context
from .models import ArchitecturePlan, ArchitectureReview, BuildMemory, FailureReport, FixPlan, ImplementationPlan, ProductSpecification, TestResult


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

IMPLEMENTATION_PLAN_SCHEMA = {
    "type": "object",
    "required": ["prototype_kind", "entrypoint", "files"],
}

IMPLEMENTATION_FILE_SCHEMA = {
    "type": "object",
    "required": ["path", "content"],
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
    "required": ["prototype_blockers", "retry_guidance", "requires_refinement"],
}


def specification_system_prompt() -> str:
    return (
        "You are the JustBuild specification agent. Return valid JSON only. "
        "Do not include markdown, prose outside JSON, or code fences. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
    )


def specification_user_prompt(idea: str, architecture_feedback: list[str] | None, memory: BuildMemory | None = None) -> str:
    feedback = architecture_feedback or []
    memory_block = _memory_prompt_block(memory)
    return (
        "Convert the product idea into a structured specification JSON.\n"
        f"Idea: {idea}\n"
        f"Architecture feedback to incorporate: {json.dumps(feedback)}\n"
        f"{memory_block}"
        "Define the smallest recognizable prototype that still demonstrates the core user value of the idea.\n"
        "For simple browser-based CRUD or utility ideas, prefer a single-user, browser-first, low-infrastructure prototype unless the idea clearly requires more.\n"
        "Do not invent backend services, databases, authentication, deployment setup, or other production-only complexity unless the idea explicitly requires them.\n"
        "Return fields: title, product_summary, requirements, features, user_stories, "
        "api_contracts, assumptions, constraints, missing_requirements.\n"
        "Each list field must be an array of strings.\n"
        "Only include missing_requirements that are unresolved and build-critical for the next implementation pass.\n"
        "Do not include future enhancements, production hardening, or optional roadmap ideas in missing_requirements.\n"
        "Include every required field even if uncertain.\n"
        "If a list field is uncertain, return an empty array instead of omitting it."
    )


def architecture_system_prompt() -> str:
    return (
        "You are the JustBuild architecture agent. Return valid JSON only and keep the design "
        "grounded in the current repository structure. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
    )


def architecture_user_prompt(spec: ProductSpecification, memory: BuildMemory | None = None) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Create an architecture plan for this product specification.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"{memory_block}"
        "Design the smallest recognizable prototype that satisfies the specification without unnecessary layers.\n"
        "Every extra service, persistence layer, manifest, migration, or deployment artifact must be justified by an explicit requirement, constraint, or build-critical missing requirement.\n"
        "When persistence is ambiguous for a simple browser app, prefer local or browser-safe persistence before introducing a backend or database.\n"
        "Do not add backend infrastructure, auth systems, Docker, migrations, or environment scaffolding unless the specification clearly requires them for the prototype.\n"
        "Return fields: summary, folder_structure, services, database_schema, "
        "design_tradeoffs, justification.\n"
        "Each list field must be an array of strings.\n"
        "The justification field is mandatory and must always be included as an array of strings.\n"
        "Include every required field even if uncertain.\n"
        "If a list field is uncertain, return an empty array instead of omitting it."
    )


def architecture_review_system_prompt() -> str:
    return (
        "You are the JustBuild architecture review agent. Return valid JSON only. "
        "Check only for critical or highly problematic issues that would block prototype implementation. "
        "Do not provide general advice, roadmap suggestions, or production hardening recommendations. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
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
        "Return JSON with prototype_blockers, retry_guidance, and requires_refinement.\n"
        "prototype_blockers and retry_guidance must be arrays of strings. requires_refinement must be a boolean.\n"
        "Treat unjustified complexity as a blocker.\n"
        "If the architecture adds backend services, databases, migrations, Docker, env scaffolding, or other infrastructure without a clear requirement, constraint, or build-critical missing requirement, return requires_refinement=true and guidance to simplify toward the smallest recognizable prototype.\n"
        "If there are no critical or highly problematic blocker issues, return requires_refinement=false and empty arrays.\n"
        "Only populate retry_guidance when requires_refinement is true.\n"
        "Keep the output minimal and blocker-focused. Do not include general recommendations."
    )


def implementation_system_prompt() -> str:
    return (
        "You are the JustBuild implementation planning agent. Return valid JSON only. "
        "Plan a concrete prototype file layout that implementation can generate file by file. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
    )

def implementation_file_system_prompt() -> str:
    return (
        "You are the JustBuild implementation file agent. Return valid JSON only. "
        "Generate exactly one prototype file at a time using the requested path and purpose. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
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
        "Create an implementation plan for this prototype.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Previous failures to fix: {json.dumps(failures, indent=2)}\n"
        f"Fix plan guidance: {json.dumps(asdict(fix_plan), indent=2) if fix_plan else 'null'}\n"
        f"{memory_block}"
        "Plan the minimum file set that still makes the prototype recognizable and functional for the stated idea.\n"
        "Prefer one compact runnable prototype over layered scaffolding.\n"
        "Merge thin abstractions when possible, and avoid backend folders, manifests, infrastructure files, or helper sprawl unless the architecture clearly requires them for the prototype.\n"
        "Return JSON with:\n"
        '- prototype_kind: string. Use "static_web" unless the architecture clearly requires something else.\n'
        '- entrypoint: string path to the main prototype entry file.\n'
        "- files: array of file descriptors with path, purpose, required, and optional depends_on.\n"
        '- notes: optional array of strings.\n'
        "For static_web prototypes, prefer a simple browser-oriented file layout.\n"
        "Keep filenames concrete and implementation-ready.\n"
        "If a fix plan is provided, prioritize it over the prior bundle.\n"
        "Include every required field even if uncertain. If an array field is uncertain, return an empty array."
    )


def implementation_file_user_prompt(
    spec: ProductSpecification,
    architecture: ArchitecturePlan,
    implementation_plan: ImplementationPlan,
    file_path: str,
    file_purpose: str,
    dependency_files: dict[str, str],
    failure_reports: list[FailureReport] | None,
    fix_plan: FixPlan | None = None,
    memory: BuildMemory | None = None,
) -> str:
    failures = [
        {"source": report.source, "summary": report.summary, "details": report.details}
        for report in (failure_reports or [])
    ]
    dependency_block = json.dumps(dependency_files, indent=2) if dependency_files else "{}"
    memory_block = _memory_prompt_block(memory)
    return (
        "Generate exactly one file for this prototype.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"Implementation plan: {json.dumps(asdict(implementation_plan), indent=2)}\n"
        f"Target file path: {file_path}\n"
        f"Target file purpose: {file_purpose}\n"
        f"Already generated dependency files: {dependency_block}\n"
        f"Previous failures to fix: {json.dumps(failures, indent=2)}\n"
        f"Fix plan guidance: {json.dumps(asdict(fix_plan), indent=2) if fix_plan else 'null'}\n"
        f"{memory_block}"
        "Write the smallest directly functional version of this file that still supports the requested prototype behavior.\n"
        "Avoid helper sprawl or placeholder abstractions unless this specific file truly needs them.\n"
        "Return JSON with path, content, and optional notes.\n"
        "The path must exactly match the target file path.\n"
        "For a static_web prototype:\n"
        "- the HTML entrypoint must include the product title and a 'Feature Breakdown' section\n"
        "- the main JS behavior file must include the phrase 'Generated Response'\n"
        "- the main CSS theme file must include a :root block\n"
        "Keep the output limited to the requested file only."
    )


def testing_system_prompt() -> str:
    return (
        "You are the JustBuild testing agent. Return valid JSON only. "
        "Produce a test checklist, not a narrative. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
    )


def testing_user_prompt(spec: ProductSpecification, architecture: ArchitecturePlan, memory: BuildMemory | None = None) -> str:
    memory_block = _memory_prompt_block(memory)
    return (
        "Create a structured verification checklist for this generated prototype.\n"
        f"Specification: {json.dumps(asdict(spec), indent=2)}\n"
        f"Architecture: {json.dumps(asdict(architecture), indent=2)}\n"
        f"{memory_block}"
        "Return JSON with unit_checks, integration_checks, failure_focus as arrays of strings.\n"
        "Include every required field even if uncertain. If an array field is uncertain, return an empty array."
    )


def evaluation_system_prompt() -> str:
    return (
        "You are the JustBuild evaluation agent. Return valid JSON only. "
        "Assess the build after implementation and testing. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
    )


def evaluation_draft_system_prompt(draft_name: str) -> str:
    return (
        "You are the JustBuild evaluation draft agent. Return valid JSON only. "
        f"Produce the {draft_name} draft for the final evaluation. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
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
        f"Return only these fields as arrays of strings: {', '.join(draft_fields)}\n"
        "Include every required field even if uncertain. If an array field is uncertain, return an empty array."
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
        "as arrays of strings.\n"
        "Include every required field even if uncertain. If an array field is uncertain, return an empty array."
    )


def debugging_system_prompt() -> str:
    return (
        "You are the JustBuild debugging agent. Return valid JSON only. "
        "Diagnose test and implementation failures, classify them, and propose a fix plan. "
        "All required keys must be present. Never omit a required key. "
        "Do not rename keys. Output exactly one JSON object."
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
