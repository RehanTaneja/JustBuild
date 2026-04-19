from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .concurrency import run_parallel
from .models import BuildContext, TaskStatus


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TerminalState(str, Enum):
    COMPLETED = "completed"
    FAILED_BLOCKING = "failed_blocking"
    COMPLETED_WITH_PUBLISH_FAILURE = "completed_with_publish_failure"


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 1


@dataclass(slots=True)
class NodeResult:
    value: Any = None
    activate_nodes: list[str] = field(default_factory=list)
    reset_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_state: TerminalState | None = None


@dataclass(slots=True)
class WorkflowNode:
    node_id: str
    node_type: str
    handler: Callable[["ExecutionState", "WorkflowNode", int], NodeResult]
    dependencies: list[str] = field(default_factory=list)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    condition: Callable[[BuildContext, "ExecutionState"], bool] | None = None
    milestone_name: str | None = None


@dataclass(slots=True)
class WorkflowEdge:
    source: str
    target: str
    label: str | None = None


@dataclass(slots=True)
class WorkflowGraph:
    nodes: dict[str, WorkflowNode]
    edges: list[WorkflowEdge] = field(default_factory=list)
    entry_nodes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NodeExecutionRecord:
    node_id: str
    node_type: str
    attempt: int
    status: str
    elapsed_ms: int
    routed_to: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionState:
    context: BuildContext
    graph: WorkflowGraph
    max_workers: int
    node_statuses: dict[str, NodeStatus] = field(default_factory=dict)
    node_attempts: dict[str, int] = field(default_factory=dict)
    active_queue: deque[str] = field(default_factory=deque)
    active_set: set[str] = field(default_factory=set)
    terminal_state: TerminalState | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def enqueue(self, node_id: str) -> None:
        if node_id in self.active_set:
            return
        self.active_queue.append(node_id)
        self.active_set.add(node_id)

    def dequeue(self, node_id: str) -> None:
        self.active_set.discard(node_id)

    def reset_node(self, node_id: str) -> None:
        self.node_statuses[node_id] = NodeStatus.PENDING
        self.node_attempts[node_id] = 0
        self.active_set.discard(node_id)


class WorkflowRuntime:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor

    def run(self, graph: WorkflowGraph, context: BuildContext, max_workers: int) -> ExecutionState:
        state = ExecutionState(context=context, graph=graph, max_workers=max_workers)
        for node_id in graph.nodes:
            state.node_statuses[node_id] = NodeStatus.PENDING
            state.node_attempts[node_id] = 0
        for node_id in graph.entry_nodes:
            state.enqueue(node_id)

        while state.active_queue and state.terminal_state is None:
            ready_nodes: list[str] = []
            pending_nodes: list[str] = []

            while state.active_queue:
                node_id = state.active_queue.popleft()
                node = graph.nodes[node_id]
                if not self._dependencies_completed(node, state):
                    pending_nodes.append(node_id)
                    continue
                if node.condition is not None and not node.condition(context, state):
                    state.node_statuses[node_id] = NodeStatus.SKIPPED
                    context.node_runs.append(
                        NodeExecutionRecord(
                            node_id=node_id,
                            node_type=node.node_type,
                            attempt=state.node_attempts[node_id] + 1,
                            status=NodeStatus.SKIPPED.value,
                            elapsed_ms=0,
                        )
                    )
                    state.dequeue(node_id)
                    continue
                ready_nodes.append(node_id)

            for node_id in pending_nodes:
                state.enqueue(node_id)

            if not ready_nodes:
                if state.active_queue:
                    raise RuntimeError("Workflow deadlocked: active nodes could not satisfy dependencies.")
                break

            tasks: dict[str, Callable[[], dict[str, Any]]] = {}
            for node_id in ready_nodes:
                node = graph.nodes[node_id]
                attempt = state.node_attempts[node_id] + 1
                state.node_attempts[node_id] = attempt
                state.node_statuses[node_id] = NodeStatus.RUNNING
                tasks[node_id] = self._build_task(node, state, attempt)
            for result in run_parallel(self.executor, tasks):
                node_id = result.name
                payload = result.value
                state.dequeue(node_id)
                record = payload["record"]
                context.node_runs.append(record)
                if payload["exception"] is not None:
                    if state.node_attempts[node_id] < graph.nodes[node_id].retry_policy.max_attempts:
                        state.node_statuses[node_id] = NodeStatus.PENDING
                        self._update_milestone_retry(context, graph.nodes[node_id].milestone_name, state.node_attempts[node_id])
                        state.enqueue(node_id)
                        continue
                    state.node_statuses[node_id] = NodeStatus.FAILED
                    self._update_milestone_failure(context, graph.nodes[node_id].milestone_name, payload["exception"])
                    state.terminal_state = TerminalState.FAILED_BLOCKING
                    continue

                state.node_statuses[node_id] = NodeStatus.COMPLETED
                node_result: NodeResult = payload["node_result"]
                for reset_node_id in node_result.reset_nodes:
                    state.reset_node(reset_node_id)
                for activate_node_id in node_result.activate_nodes:
                    state.enqueue(activate_node_id)
                if node_result.terminal_state is not None:
                    state.terminal_state = node_result.terminal_state

        if state.terminal_state is None:
            state.terminal_state = TerminalState.COMPLETED
        context.workflow_terminal_state = state.terminal_state.value
        return state

    def _build_task(self, node: WorkflowNode, state: ExecutionState, attempt: int) -> Callable[[], dict[str, Any]]:
        def task() -> dict[str, Any]:
            started = time.perf_counter()
            try:
                node_result = node.handler(state, node, attempt)
                error = None
            except Exception as exc:  # pragma: no cover - exercised by runtime tests/orchestrator failures
                node_result = None
                error = str(exc)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            record = NodeExecutionRecord(
                node_id=node.node_id,
                node_type=node.node_type,
                attempt=attempt,
                status=NodeStatus.COMPLETED.value if error is None else NodeStatus.FAILED.value,
                elapsed_ms=elapsed_ms,
                routed_to=list(node_result.activate_nodes if node_result else []),
                error=error,
                metadata=dict(node_result.metadata if node_result else {}),
            )
            return {"node_result": node_result, "record": record, "exception": error}

        return task

    def _dependencies_completed(self, node: WorkflowNode, state: ExecutionState) -> bool:
        return all(state.node_statuses.get(dep) == NodeStatus.COMPLETED for dep in node.dependencies)

    def _update_milestone_retry(self, context: BuildContext, milestone_name: str | None, attempt: int) -> None:
        if milestone_name is None:
            return
        for milestone in context.milestones:
            if milestone.name == milestone_name:
                milestone.status = TaskStatus.RETRYING
                milestone.retries += 1
                milestone.metadata.update({"attempt": attempt})
                return

    def _update_milestone_failure(self, context: BuildContext, milestone_name: str | None, error: str) -> None:
        if milestone_name is None:
            return
        for milestone in context.milestones:
            if milestone.name == milestone_name:
                milestone.status = TaskStatus.FAILED
                milestone.metadata.update({"error": error})
                return
