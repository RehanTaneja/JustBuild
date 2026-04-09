# This import is basically so that we can reference classes before they exist and using tools like BuildContext we can reference classes inside themselves
from __future__ import annotations
# @dataclass autogenerates constructor, repr etc and field() is used for defaults like list/array safely
from dataclasses import dataclass, field
from enum import Enum
# Pathlib is a cleaner way to handle file paths better than strings
from pathlib import Path
# Allows flexible types for generic metadata
from typing import Any

"""
This file is not actually implementing anything, this is just defining the schema for models to interact with each other and the system.
"""

# Defines the current status of the task, uses enums for consistent communication
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

# The user input converted into a class. 
@dataclass(slots=True)
class BuildRequest:
    product_idea: str # Simple user input string like "Build a calorie logging app".
    output_root: Path # Root Path where all the files of the project will be saved.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_backend_type: str | None = None

# This is for the orchestrator to track progress, think of it as tasks.
@dataclass(slots=True)
class Milestone:
    name: str
    description: str
    owner: str
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

# Logging for everything done by every agent for debugging and transparency.
@dataclass(slots=True)
class DecisionLog:
    agent: str # Which agent did this
    message: str # What has been done
    category: str # Type like design, error, retry etc
    iteration: int # No. of iterations passed
    elapsed_ms: int # Time passed
    metadata: dict[str, Any] = field(default_factory=dict)

# Logging for every failure for debugging.
@dataclass(slots=True)
class FailureReport:
    source: str # Which agent failed
    summary: str # What went wrong
    details: list[str]
    blocking: bool = True # Should we stop

# This is for planning output. This is built by the Specification/Planning agent and used by the architecture agent.
# Done to force structured planning before coding and to prevent vibe coding chaos.
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

# Separates what to build and how to build. Takes Product specification and proposed architecture.
@dataclass(slots=True)
class ArchitecturePlan:
    summary: str
    folder_structure: list[str] 
    services: list[str]
    database_schema: list[str]
    design_tradeoffs: list[str]
    justification: list[str]

# Separates Spec-> Architecture-> Implementation. This takes the ArchitecturePlan and tracks the actual output of the coding agent.
@dataclass(slots=True)
class ImplementationArtifacts:
    prototype_dir: Path | None = None # Path to prototype root
    generated_files: list[Path] = field(default_factory=list) # List of files generated.
    notes: list[str] = field(default_factory=list)
    file_bundle: dict[str, str] = field(default_factory=dict)

# Used by orchestrator to enable/continue/stop iteration loop: passed->proceed, else repeat.
@dataclass(slots=True)
class TestResult:
    passed: bool
    summary: str
    unit_results: list[str]
    integration_results: list[str]
    llm_checks: list[str] = field(default_factory=list)
    failure_reports: list[FailureReport] = field(default_factory=list)

# A review that happens after testing that highlights constraints and capabilities after prototype has been built.
@dataclass(slots=True)
class EvaluationReport:
    code_quality: list[str]
    maintainability: list[str]
    scalability_risks: list[str]
    security_concerns: list[str]
    refactoring_opportunities: list[str]
    technical_debt: list[str]
    risk_assessment: list[str]

# This is the most important overall class. It is the current state of the project.
# Contains everything including the input, progress, logs, history etc.
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

"""
Mental Model:
BuildRequest (User Input)
       |
       V
BuildContext (Central Brain)
       |
       V
Agents Update:
    - Specification
    - Architecture
    - Implementation
    - Testing
    - Evaluation
       |
       V
Final Reports
"""
