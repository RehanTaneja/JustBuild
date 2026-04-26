from __future__ import annotations

import json
import time
from contextlib import contextmanager
# Converts dataclass->dictionary. Needed to serialize JSON
from dataclasses import asdict
from pathlib import Path
# Type hinting for context manager
from typing import Iterator

# Use the central brain and log structure
from .models import BuildContext, DecisionLog
# For getting output folder path
from .reporting import _build_root

"""
This file acts as a black box recorder + performance tracker. 
- It looks at what happened
- It records how long it took
- It writes everything into a final JSON summary
"""

class BuildLogger:
    """Structured logger for agent decisions and timing which it records into BuildContext.
    A class that implements the DecisionLogger schema.

    This is intentionally simple: the same interface can later send events to
    OpenTelemetry, Kafka, Datadog, or a workflow database without changing the
    agents themselves.
    """

    # context is the global state (current instance of BuildContext for the project) so that logs are stored inside the system state.
    def __init__(self, context: BuildContext) -> None:
        self.context = context 

    # Creates a DecisionLog instance and appends it to context.decisions
    def log(
        self,
        agent: str,
        message: str,
        category: str,
        iteration: int,
        elapsed_ms: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.context.decisions.append(
            DecisionLog(
                agent=agent,
                message=message,
                category=category,
                iteration=iteration,
                elapsed_ms=elapsed_ms,
                metadata=dict(metadata or {}),
            )
        )

    # Wraps code and automatically logs elapsed time.
    @contextmanager
    def timed(self, agent: str, message: str, category: str, iteration: int) -> Iterator[None]:
        started_at = time.perf_counter() # Starts the timer
        try:
            yield # Runs code
        finally:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000) # Measures elapsed time after code execution is finished.
            self.log(agent=agent, message=message, category=category, iteration=iteration, elapsed_ms=elapsed_ms)

# Creates a final JSON report for the entire build
def write_build_summary(context: BuildContext) -> Path:
    output_dir = _build_root(context) # gets base folder.
    output_dir.mkdir(parents=True, exist_ok=True) # creates it if it doesn't exist.
    path = output_dir / "build_summary.json" # defined file path for the output JSON summary.
    payload = {
        "product_idea": context.request.product_idea,
        "llm_backend": {
            "provider": context.request.llm_provider,
            "model": context.request.llm_model,
            "base_url": context.request.llm_base_url,
            "backend_type": context.request.llm_backend_type,
            "structured_output_mode": context.request.llm_structured_output_mode,
            "timeout_s": context.request.llm_timeout_s,
        },
        "testing_backend": {
            "enable_playwright": context.request.enable_playwright,
            "node_bin": context.request.node_bin,
            "pytest_bin": context.request.pytest_bin,
            "max_workers": context.request.max_workers,
        },
        "memory_path": str(context.request.memory_path) if context.request.memory_path else None,
        "github_delivery": {
            "publish_to_github": context.request.publish_to_github,
            "github_repo_name": context.request.github_repo_name,
            "github_repo_visibility": context.request.github_repo_visibility,
        },
        "milestones": [asdict(m) for m in context.milestones],
        "decisions": [asdict(d) for d in context.decisions],
        "iterations": context.iterations,
        "workflow_terminal_state": context.workflow_terminal_state,
        "node_runs": [asdict(run) for run in context.node_runs],
        "specification": asdict(context.specification) if context.specification else None,
        "architecture": asdict(context.architecture) if context.architecture else None,
        "architecture_review": asdict(context.architecture_review) if context.architecture_review else None,
        "implementation": {
            "prototype_dir": str(context.implementation.prototype_dir) if context.implementation and context.implementation.prototype_dir else None,
            "generated_files": [str(path) for path in context.implementation.generated_files] if context.implementation else [],
            "notes": context.implementation.notes if context.implementation else [],
            "file_bundle": context.implementation.file_bundle if context.implementation else {},
        },
        "testing": {
            "passed": context.testing.passed if context.testing else None,
            "summary": context.testing.summary if context.testing else None,
            "unit_results": context.testing.unit_results if context.testing else [],
            "integration_results": context.testing.integration_results if context.testing else [],
            "llm_checks": context.testing.llm_checks if context.testing else [],
            "execution_results": context.testing.execution_results if context.testing else [],
            "schema_results": context.testing.schema_results if context.testing else [],
            "browser_results": context.testing.browser_results if context.testing else [],
            "skipped_checks": context.testing.skipped_checks if context.testing else [],
            "failure_reports": [asdict(report) for report in context.testing.failure_reports] if context.testing else [],
        },
        "debugging": asdict(context.debugging) if context.debugging else None,
        "evaluation": asdict(context.evaluation) if context.evaluation else None,
        "memory": asdict(context.memory) if context.memory else None,
        "github_publish": {
            "enabled": context.github_publish.enabled if context.github_publish else False,
            "published": context.github_publish.published if context.github_publish else False,
            "repo_name": context.github_publish.repo_name if context.github_publish else None,
            "repo_full_name": context.github_publish.repo_full_name if context.github_publish else None,
            "repo_url": context.github_publish.repo_url if context.github_publish else None,
            "branch": context.github_publish.branch if context.github_publish else None,
            "local_publish_dir": str(context.github_publish.local_publish_dir) if context.github_publish and context.github_publish.local_publish_dir else None,
            "commits": context.github_publish.commits if context.github_publish else [],
            "failure_reason": context.github_publish.failure_reason if context.github_publish else None,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    context.build_summary_path = path
    return path

"""
While agents are doing work, the BuildLogger class automatically logs and records each step into overall system BuildContext.
Note: This is different from real-time logging as it just stores decision timestamps in memory. 
Once agents are done with their work, the write_build_summary function creates a final JSON summary report.
"""
