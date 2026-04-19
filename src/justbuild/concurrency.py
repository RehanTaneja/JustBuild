from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParallelTaskResult:
    name: str
    value: Any
    elapsed_ms: int


def run_parallel(
    executor: ThreadPoolExecutor,
    tasks: dict[str, Callable[[], Any]],
) -> list[ParallelTaskResult]:
    future_map = {executor.submit(_timed_task, name, callback): name for name, callback in tasks.items()}
    results: list[ParallelTaskResult] = []
    for future in as_completed(future_map):
        name, value, elapsed_ms = future.result()
        results.append(ParallelTaskResult(name=name, value=value, elapsed_ms=elapsed_ms))
    results.sort(key=lambda item: item.name)
    return results


def _timed_task(name: str, callback: Callable[[], Any]) -> tuple[str, Any, int]:
    started = time.perf_counter()
    value = callback()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return name, value, elapsed_ms
