from __future__ import annotations

import json
from typing import Any

from .models import ArchitecturePlan, ArchitectureReview, EvaluationReport, FailureReport, FixPlan, ImplementationPlan, ImplementationPlanFile, ProductSpecification


class JSONValidationError(ValueError):
    """Raised when an LLM response cannot be normalized into the expected schema."""


def parse_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if not stripped:
        raise JSONValidationError("LLM returned an empty response")
    if "```" in stripped:
        raise JSONValidationError("LLM returned markdown fences instead of plain JSON")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise JSONValidationError(f"LLM returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise JSONValidationError("LLM response must be a JSON object")
    return payload


def require_keys(payload: dict[str, Any], required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        present = ", ".join(sorted(payload.keys())) or "(none)"
        raise JSONValidationError(
            f"LLM response is missing required keys: {', '.join(missing)}. Present keys: {present}"
        )


def normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JSONValidationError(f"Field '{field_name}' must be a non-empty string")
    return value.strip()


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise JSONValidationError(f"Field '{field_name}' must be a list of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise JSONValidationError(f"Field '{field_name}' must contain only non-empty strings")
        normalized.append(item.strip())
    return normalized


def build_failure_report(source: str, summary: str, details: list[str]) -> FailureReport:
    return FailureReport(source=source, summary=summary, details=details)


def parse_product_specification(raw_text: str) -> ProductSpecification:
    payload = parse_json_object(raw_text)
    require_keys(
        payload,
        [
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
    )
    return ProductSpecification(
        title=normalize_text(payload["title"], "title"),
        product_summary=normalize_text(payload["product_summary"], "product_summary"),
        requirements=normalize_string_list(payload["requirements"], "requirements"),
        features=normalize_string_list(payload["features"], "features"),
        user_stories=normalize_string_list(payload["user_stories"], "user_stories"),
        api_contracts=normalize_string_list(payload["api_contracts"], "api_contracts"),
        assumptions=normalize_string_list(payload["assumptions"], "assumptions"),
        constraints=normalize_string_list(payload["constraints"], "constraints"),
        missing_requirements=normalize_string_list(payload["missing_requirements"], "missing_requirements"),
    )


def parse_architecture_plan(raw_text: str) -> ArchitecturePlan:
    payload = parse_json_object(raw_text)
    require_keys(
        payload,
        ["summary", "folder_structure", "services", "database_schema", "design_tradeoffs", "justification"],
    )
    return ArchitecturePlan(
        summary=normalize_text(payload["summary"], "summary"),
        folder_structure=normalize_string_list(payload["folder_structure"], "folder_structure"),
        services=normalize_string_list(payload["services"], "services"),
        database_schema=normalize_string_list(payload["database_schema"], "database_schema"),
        design_tradeoffs=normalize_string_list(payload["design_tradeoffs"], "design_tradeoffs"),
        justification=normalize_string_list(payload["justification"], "justification"),
    )


def parse_architecture_review(raw_text: str) -> ArchitectureReview:
    payload = parse_json_object(raw_text)
    require_keys(payload, ["prototype_blockers", "retry_guidance", "requires_refinement"])
    requires_refinement = payload["requires_refinement"]
    if not isinstance(requires_refinement, bool):
        raise JSONValidationError("Field 'requires_refinement' must be a boolean")
    prototype_blockers = normalize_string_list(payload["prototype_blockers"], "prototype_blockers")
    retry_guidance = normalize_string_list(payload["retry_guidance"], "retry_guidance")
    if not requires_refinement and (prototype_blockers or retry_guidance):
        raise JSONValidationError("Non-blocking architecture review must return empty prototype_blockers and retry_guidance")
    return ArchitectureReview(
        prototype_blockers=prototype_blockers,
        retry_guidance=retry_guidance,
        requires_refinement=requires_refinement,
    )


def parse_implementation_plan(raw_text: str) -> ImplementationPlan:
    payload = parse_json_object(raw_text)
    require_keys(payload, ["prototype_kind", "entrypoint", "files", "notes"])
    prototype_kind = normalize_text(payload["prototype_kind"], "prototype_kind")
    entrypoint = normalize_text(payload["entrypoint"], "entrypoint")
    notes = normalize_string_list(payload["notes"], "notes")
    files_value = payload["files"]
    if not isinstance(files_value, list) or not files_value:
        raise JSONValidationError("Field 'files' must be a non-empty list of file descriptors")

    files: list[ImplementationPlanFile] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(files_value):
        if not isinstance(item, dict):
            raise JSONValidationError(f"Field 'files[{index}]' must be an object")
        require_keys(item, ["path", "purpose", "required"])
        path = normalize_text(item["path"], f"files[{index}].path")
        if path in seen_paths:
            raise JSONValidationError(f"Implementation plan contains duplicate file path '{path}'")
        seen_paths.add(path)
        purpose = normalize_text(item["purpose"], f"files[{index}].purpose")
        required = item["required"]
        if not isinstance(required, bool):
            raise JSONValidationError(f"Field 'files[{index}].required' must be a boolean")
        depends_on = normalize_string_list(item.get("depends_on", []), f"files[{index}].depends_on")
        files.append(
            ImplementationPlanFile(
                path=path,
                purpose=purpose,
                required=required,
                depends_on=depends_on,
            )
        )
    return ImplementationPlan(prototype_kind=prototype_kind, entrypoint=entrypoint, files=files, notes=notes)


def parse_implementation_file(raw_text: str) -> tuple[str, str, list[str]]:
    payload = parse_json_object(raw_text)
    require_keys(payload, ["path", "content", "notes"])
    return (
        normalize_text(payload["path"], "path"),
        normalize_text(payload["content"], "content"),
        normalize_string_list(payload["notes"], "notes"),
    )


def parse_testing_plan(raw_text: str) -> tuple[list[str], list[str], list[str]]:
    payload = parse_json_object(raw_text)
    require_keys(payload, ["unit_checks", "integration_checks", "failure_focus"])
    return (
        normalize_string_list(payload["unit_checks"], "unit_checks"),
        normalize_string_list(payload["integration_checks"], "integration_checks"),
        normalize_string_list(payload["failure_focus"], "failure_focus"),
    )


def parse_evaluation_report(raw_text: str) -> EvaluationReport:
    payload = parse_json_object(raw_text)
    require_keys(
        payload,
        [
            "code_quality",
            "maintainability",
            "scalability_risks",
            "security_concerns",
            "refactoring_opportunities",
            "technical_debt",
            "risk_assessment",
        ],
    )
    return EvaluationReport(
        code_quality=normalize_string_list(payload["code_quality"], "code_quality"),
        maintainability=normalize_string_list(payload["maintainability"], "maintainability"),
        scalability_risks=normalize_string_list(payload["scalability_risks"], "scalability_risks"),
        security_concerns=normalize_string_list(payload["security_concerns"], "security_concerns"),
        refactoring_opportunities=normalize_string_list(payload["refactoring_opportunities"], "refactoring_opportunities"),
        technical_debt=normalize_string_list(payload["technical_debt"], "technical_debt"),
        risk_assessment=normalize_string_list(payload["risk_assessment"], "risk_assessment"),
    )


def parse_fix_plan(raw_text: str) -> FixPlan:
    payload = parse_json_object(raw_text)
    require_keys(
        payload,
        ["file_changes", "root_cause", "strategy", "failure_groups", "priority_order"],
    )
    failure_groups = normalize_string_list(payload["failure_groups"], "failure_groups")
    allowed_groups = {
        "missing_file",
        "logic_error",
        "schema_mismatch",
        "content_mismatch",
        "llm_output_invalid",
    }
    invalid_groups = [group for group in failure_groups if group not in allowed_groups]
    if invalid_groups:
        raise JSONValidationError(
            f"Field 'failure_groups' contains invalid values: {', '.join(invalid_groups)}"
        )
    return FixPlan(
        file_changes=normalize_string_list(payload["file_changes"], "file_changes"),
        root_cause=normalize_text(payload["root_cause"], "root_cause"),
        strategy=normalize_text(payload["strategy"], "strategy"),
        failure_groups=failure_groups,
        priority_order=normalize_string_list(payload["priority_order"], "priority_order"),
    )
