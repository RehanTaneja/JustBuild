from __future__ import annotations

import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from justbuild.models import BuildContext, BuildRequest
from justbuild.workflow import ExecutionState, NodeResult, RetryPolicy, TerminalState, WorkflowGraph, WorkflowIntegrityError, WorkflowNode, WorkflowRuntime


class WorkflowRuntimeTests(unittest.TestCase):
    def _context(self) -> BuildContext:
        return BuildContext(request=BuildRequest(product_idea="workflow test", output_root=Path(tempfile.gettempdir())))

    def test_runtime_executes_dependency_order_and_parallel_siblings(self) -> None:
        order: list[str] = []

        def handler(name: str, activate: list[str] | None = None):
            def _handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
                order.append(name)
                return NodeResult(activate_nodes=activate or [])

            return _handler

        graph = WorkflowGraph(
            nodes={
                "root": WorkflowNode("root", "system", handler("root", ["left", "right"])),
                "left": WorkflowNode("left", "worker", handler("left", ["merge"]), dependencies=["root"]),
                "right": WorkflowNode("right", "worker", handler("right", ["merge"]), dependencies=["root"]),
                "merge": WorkflowNode("merge", "merge", handler("merge"), dependencies=["left", "right"]),
            },
            entry_nodes=["root"],
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            state = WorkflowRuntime(executor).run(graph, self._context(), max_workers=4)

        self.assertEqual(state.terminal_state, TerminalState.COMPLETED)
        self.assertEqual(order[0], "root")
        self.assertIn("left", order[1:3])
        self.assertIn("right", order[1:3])
        self.assertEqual(order[-1], "merge")

    def test_runtime_skips_conditionally_disabled_node(self) -> None:
        executed: list[str] = []

        def root_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            executed.append("root")
            return NodeResult(activate_nodes=["conditional"])

        def conditional_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            executed.append("conditional")
            return NodeResult()

        context = self._context()
        graph = WorkflowGraph(
            nodes={
                "root": WorkflowNode("root", "system", root_handler),
                "conditional": WorkflowNode(
                    "conditional",
                    "system",
                    conditional_handler,
                    dependencies=["root"],
                    condition=lambda context, state: False,
                ),
            },
            entry_nodes=["root"],
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            state = WorkflowRuntime(executor).run(graph, context, max_workers=2)

        self.assertEqual(state.terminal_state, TerminalState.COMPLETED)
        self.assertEqual(executed, ["root"])
        self.assertEqual(context.node_runs[-1].status, "skipped")

    def test_runtime_retries_then_routes_failure_to_terminal(self) -> None:
        calls = {"count": 0}

        def failing_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            calls["count"] += 1
            raise ValueError("boom")

        graph = WorkflowGraph(
            nodes={
                "failing": WorkflowNode(
                    "failing",
                    "system",
                    failing_handler,
                    retry_policy=RetryPolicy(max_attempts=2),
                )
            },
            entry_nodes=["failing"],
        )

        context = self._context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            state = WorkflowRuntime(executor).run(graph, context, max_workers=1)

        self.assertEqual(calls["count"], 2)
        self.assertEqual(state.terminal_state, TerminalState.FAILED_BLOCKING)
        self.assertEqual(context.node_runs[-1].error, "boom")

    def test_runtime_requeues_deferred_merge_after_upstream_retry(self) -> None:
        order: list[str] = []
        left_calls = {"count": 0}

        def root_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            order.append("root")
            return NodeResult(activate_nodes=["left", "right"])

        def left_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            order.append(f"left-{attempt}")
            time.sleep(0.03)
            left_calls["count"] += 1
            if attempt == 1:
                raise ValueError("left boom")
            return NodeResult(activate_nodes=["merge"])

        def right_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            order.append("right")
            return NodeResult(activate_nodes=["merge"])

        def merge_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            order.append("merge")
            return NodeResult()

        graph = WorkflowGraph(
            nodes={
                "root": WorkflowNode("root", "system", root_handler),
                "left": WorkflowNode("left", "worker", left_handler, dependencies=["root"], retry_policy=RetryPolicy(max_attempts=2)),
                "right": WorkflowNode("right", "worker", right_handler, dependencies=["root"]),
                "merge": WorkflowNode("merge", "merge", merge_handler, dependencies=["left", "right"]),
            },
            entry_nodes=["root"],
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            state = WorkflowRuntime(executor).run(graph, self._context(), max_workers=4)

        self.assertEqual(left_calls["count"], 2)
        self.assertEqual(state.terminal_state, TerminalState.COMPLETED)
        self.assertEqual(order[-1], "merge")

    def test_runtime_reset_node_replays_loop_without_stale_queue_entries(self) -> None:
        steps: list[str] = []
        gate_runs = {"count": 0}

        def root_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            steps.append("root")
            return NodeResult(activate_nodes=["spec"])

        def spec_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            steps.append(f"spec-{attempt}")
            return NodeResult(activate_nodes=["plan", "review"])

        def plan_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            steps.append(f"plan-{attempt}")
            return NodeResult(activate_nodes=["gate"])

        def review_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            steps.append(f"review-{attempt}")
            return NodeResult(activate_nodes=["gate"])

        def gate_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            gate_runs["count"] += 1
            steps.append(f"gate-{gate_runs['count']}")
            if gate_runs["count"] == 1:
                return NodeResult(
                    activate_nodes=["spec"],
                    reset_nodes=["spec", "plan", "review", "gate"],
                )
            return NodeResult(activate_nodes=["done"])

        def done_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            steps.append("done")
            return NodeResult(terminal_state=TerminalState.COMPLETED)

        graph = WorkflowGraph(
            nodes={
                "root": WorkflowNode("root", "system", root_handler),
                "spec": WorkflowNode("spec", "agent", spec_handler, dependencies=["root"]),
                "plan": WorkflowNode("plan", "agent", plan_handler, dependencies=["spec"]),
                "review": WorkflowNode("review", "agent", review_handler, dependencies=["spec"]),
                "gate": WorkflowNode("gate", "gate", gate_handler, dependencies=["plan", "review"]),
                "done": WorkflowNode("done", "system", done_handler, dependencies=["gate"]),
            },
            entry_nodes=["root"],
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            state = WorkflowRuntime(executor).run(graph, self._context(), max_workers=4)

        self.assertEqual(state.terminal_state, TerminalState.COMPLETED)
        self.assertEqual(gate_runs["count"], 2)
        self.assertEqual(steps.count("done"), 1)

    def test_runtime_raises_integrity_error_when_activated_node_is_lost(self) -> None:
        original_enqueue = ExecutionState.enqueue

        def corrupted_enqueue(self: ExecutionState, node_id: str) -> None:
            if node_id == "merge" and node_id in self.activated_nodes:
                self.activated_nodes.add(node_id)
                return
            original_enqueue(self, node_id)

        def root_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            return NodeResult(activate_nodes=["left", "right"])

        def left_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            time.sleep(0.03)
            if attempt == 1:
                raise ValueError("left boom")
            return NodeResult()

        def right_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            return NodeResult(activate_nodes=["merge"])

        def merge_handler(state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
            return NodeResult()

        graph = WorkflowGraph(
            nodes={
                "root": WorkflowNode("root", "system", root_handler),
                "left": WorkflowNode("left", "worker", left_handler, dependencies=["root"], retry_policy=RetryPolicy(max_attempts=2)),
                "right": WorkflowNode("right", "worker", right_handler, dependencies=["root"]),
                "merge": WorkflowNode("merge", "merge", merge_handler, dependencies=["left", "right"]),
            },
            entry_nodes=["root"],
        )

        context = self._context()
        with patch.object(ExecutionState, "enqueue", new=corrupted_enqueue):
            with ThreadPoolExecutor(max_workers=4) as executor:
                with self.assertRaises(WorkflowIntegrityError):
                    WorkflowRuntime(executor).run(graph, context, max_workers=4)

        self.assertIsNotNone(context.last_failure)
        self.assertEqual(context.last_failure["error_type"], "workflow_integrity")


if __name__ == "__main__":
    unittest.main()
