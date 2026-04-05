from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

from .models import BuildContext, DecisionLog
from .reporting import _build_root


class BuildLogger:
    """Structured logger for agent decisions and timing.

    This is intentionally simple: the same interface can later send events to
    OpenTelemetry, Kafka, Datadog, or a workflow database without changing the
    agents themselves.
    """

    def __init__(self, context: BuildContext) -> None:
        self.context = context

    def log(self, agent: str, message: str, category: str, iteration: int, elapsed_ms: int) -> None:
        self.context.decisions.append(
            DecisionLog(
                agent=agent,
                message=message,
                category=category,
                iteration=iteration,
                elapsed_ms=elapsed_ms,
            )
        )

    @contextmanager
    def timed(self, agent: str, message: str, category: str, iteration: int) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            self.log(agent=agent, message=message, category=category, iteration=iteration, elapsed_ms=elapsed_ms)


def write_build_summary(context: BuildContext) -> Path:
    output_dir = _build_root(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "build_summary.json"
    payload = {
        "product_idea": context.request.product_idea,
        "milestones": [asdict(m) for m in context.milestones],
        "decisions": [asdict(d) for d in context.decisions],
        "iterations": context.iterations,
        "specification": asdict(context.specification) if context.specification else None,
        "architecture": asdict(context.architecture) if context.architecture else None,
        "implementation": {
            "prototype_dir": str(context.implementation.prototype_dir) if context.implementation and context.implementation.prototype_dir else None,
            "generated_files": [str(path) for path in context.implementation.generated_files] if context.implementation else [],
            "notes": context.implementation.notes if context.implementation else [],
        },
        "testing": {
            "passed": context.testing.passed if context.testing else None,
            "summary": context.testing.summary if context.testing else None,
            "unit_results": context.testing.unit_results if context.testing else [],
            "integration_results": context.testing.integration_results if context.testing else [],
            "failure_reports": [asdict(report) for report in context.testing.failure_reports] if context.testing else [],
        },
        "evaluation": asdict(context.evaluation) if context.evaluation else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    context.build_summary_path = path
    return path
