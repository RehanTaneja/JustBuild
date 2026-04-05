from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(slots=True)
class BuildRequest:
    product_idea: str
    output_root: Path


@dataclass(slots=True)
class Milestone:
    name: str
    description: str
    owner: str
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionLog:
    agent: str
    message: str
    category: str
    iteration: int
    elapsed_ms: int


@dataclass(slots=True)
class FailureReport:
    source: str
    summary: str
    details: list[str]
    blocking: bool = True


@dataclass(slots=True)
class ProductSpecification:
    title: str
    product_summary: str
    requirements: list[str]
    features: list[str]
    user_stories: list[str]
    api_contracts: list[str]
    assumptions: list[str]
    constraints: list[str]
    missing_requirements: list[str]


@dataclass(slots=True)
class ArchitecturePlan:
    summary: str
    folder_structure: list[str]
    services: list[str]
    database_schema: list[str]
    design_tradeoffs: list[str]
    justification: list[str]


@dataclass(slots=True)
class ImplementationArtifacts:
    prototype_dir: Path | None = None
    generated_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TestResult:
    passed: bool
    summary: str
    unit_results: list[str]
    integration_results: list[str]
    failure_reports: list[FailureReport] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationReport:
    code_quality: list[str]
    maintainability: list[str]
    scalability_risks: list[str]
    security_concerns: list[str]
    refactoring_opportunities: list[str]
    technical_debt: list[str]
    risk_assessment: list[str]


@dataclass(slots=True)
class BuildContext:
    request: BuildRequest
    milestones: list[Milestone] = field(default_factory=list)
    decisions: list[DecisionLog] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    specification: ProductSpecification | None = None
    architecture: ArchitecturePlan | None = None
    implementation: ImplementationArtifacts | None = None
    testing: TestResult | None = None
    evaluation: EvaluationReport | None = None
    build_summary_path: Path | None = None
    final_report_path: Path | None = None
