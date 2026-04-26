from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import BuildContext, DecisionLog
from .reporting import _build_root


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_run_artifacts(context: BuildContext, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    context.run_dir = run_dir
    context.events_log_path = run_dir / "build_events.jsonl"
    context.text_log_path = run_dir / "build.log"
    context.partial_summary_path = run_dir / "build_summary.partial.json"
    for path in (context.events_log_path, context.text_log_path):
        path.write_text("", encoding="utf-8")
    write_partial_summary(context)


class BuildLogger:
    """Structured logger with in-memory, on-disk, and optional stderr sinks."""

    def __init__(self, context: BuildContext, stderr=None) -> None:
        self.context = context
        self.stderr = stderr or sys.stderr

    def log(
        self,
        agent: str,
        message: str,
        category: str,
        iteration: int,
        elapsed_ms: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        event_metadata = dict(metadata or {})
        self.context.decisions.append(
            DecisionLog(
                agent=agent,
                message=message,
                category=category,
                iteration=iteration,
                elapsed_ms=elapsed_ms,
                metadata=event_metadata,
            )
        )
        self.emit_event(
            category=category,
            message=message,
            metadata=event_metadata,
            agent=agent,
            iteration=iteration,
            elapsed_ms=elapsed_ms,
        )

    def emit_event(
        self,
        category: str,
        message: str,
        metadata: dict[str, object] | None = None,
        *,
        agent: str | None = None,
        iteration: int | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        payload = {
            "timestamp": utc_timestamp(),
            "category": category,
            "message": message,
            "agent": agent,
            "iteration": iteration,
            "elapsed_ms": elapsed_ms,
            "metadata": dict(metadata or {}),
        }
        self._write_event(payload)
        self._write_text_line(payload)
        self._write_console_line(payload)

    @contextmanager
    def timed(self, agent: str, message: str, category: str, iteration: int) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            self.log(agent=agent, message=message, category=category, iteration=iteration, elapsed_ms=elapsed_ms)

    def _write_event(self, payload: dict[str, Any]) -> None:
        if self.context.events_log_path is None:
            return
        with self.context.events_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()

    def _write_text_line(self, payload: dict[str, Any]) -> None:
        if self.context.text_log_path is None:
            return
        metadata = payload.get("metadata") or {}
        meta_suffix = f" | {json.dumps(metadata, sort_keys=True)}" if metadata else ""
        agent = payload.get("agent") or "system"
        line = f"{payload['timestamp']} [{payload['category']}] {agent}: {payload['message']}{meta_suffix}\n"
        with self.context.text_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def _write_console_line(self, payload: dict[str, Any]) -> None:
        mode = self.context.request.log_mode
        if mode == "quiet":
            return
        if mode == "progress" and payload["category"] not in {
            "workflow_queue",
            "workflow_start",
            "workflow_complete",
            "workflow_retry",
            "workflow_failure",
            "llm_failure",
            "llm_timeout",
            "schema_repair",
            "schema_completion_repair",
            "build_failure",
            "build_status",
        }:
            return
        line = self._console_message(payload)
        if line:
            print(line, file=self.stderr, flush=True)

    def _console_message(self, payload: dict[str, Any]) -> str:
        category = payload["category"]
        message = payload["message"]
        metadata = payload.get("metadata") or {}
        if category == "workflow_start":
            return f"Starting {metadata.get('node_id', 'step')}"
        if category == "workflow_complete":
            return f"Completed {metadata.get('node_id', 'step')}"
        if category == "workflow_retry":
            return f"Retrying {metadata.get('node_id', 'step')} ({metadata.get('attempt')}/{metadata.get('max_attempts')})"
        if category == "workflow_failure":
            return f"Failed {metadata.get('node_id', 'step')}: {metadata.get('error', message)}"
        if category == "llm_timeout":
            return message
        if category == "schema_repair":
            return message
        if category == "schema_completion_repair":
            return message
        if category == "build_failure":
            return message
        if category == "build_status":
            return message
        if category == "llm_failure":
            return message
        return message if self.context.request.log_mode == "debug" else ""


def partial_summary_payload(context: BuildContext) -> dict[str, Any]:
    return {
        "product_idea": context.request.product_idea,
        "run_dir": str(context.run_dir) if context.run_dir else None,
        "events_log_path": str(context.events_log_path) if context.events_log_path else None,
        "text_log_path": str(context.text_log_path) if context.text_log_path else None,
        "llm_backend": {
            "provider": context.request.llm_provider,
            "model": context.request.llm_model,
            "base_url": context.request.llm_base_url,
            "backend_type": context.request.llm_backend_type,
            "structured_output_mode": context.request.llm_structured_output_mode,
            "timeout_s": context.request.llm_timeout_s,
        },
        "milestones": [asdict(m) for m in context.milestones],
        "decisions": [asdict(d) for d in context.decisions],
        "iterations": context.iterations,
        "workflow_terminal_state": context.workflow_terminal_state,
        "node_runs": [asdict(run) for run in context.node_runs],
        "last_failure": context.last_failure,
        "specification": asdict(context.specification) if context.specification else None,
        "architecture": asdict(context.architecture) if context.architecture else None,
        "architecture_review": asdict(context.architecture_review) if context.architecture_review else None,
        "implementation": {
            "prototype_dir": str(context.implementation.prototype_dir) if context.implementation and context.implementation.prototype_dir else None,
            "generated_files": [str(path) for path in context.implementation.generated_files] if context.implementation else [],
            "notes": context.implementation.notes if context.implementation else [],
        },
        "testing": {
            "passed": context.testing.passed if context.testing else None,
            "summary": context.testing.summary if context.testing else None,
            "failure_reports": [asdict(report) for report in context.testing.failure_reports] if context.testing else [],
        },
        "debugging": asdict(context.debugging) if context.debugging else None,
        "evaluation": asdict(context.evaluation) if context.evaluation else None,
    }


def write_partial_summary(context: BuildContext) -> Path | None:
    if context.partial_summary_path is None:
        return None
    payload = partial_summary_payload(context)
    context.partial_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return context.partial_summary_path


def write_build_summary(context: BuildContext) -> Path:
    output_dir = _build_root(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "build_summary.json"
    payload = {
        "product_idea": context.request.product_idea,
        "run_dir": str(context.run_dir) if context.run_dir else None,
        "events_log_path": str(context.events_log_path) if context.events_log_path else None,
        "text_log_path": str(context.text_log_path) if context.text_log_path else None,
        "partial_summary_path": str(context.partial_summary_path) if context.partial_summary_path else None,
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
        "last_failure": context.last_failure,
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
