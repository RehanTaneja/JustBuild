from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from justbuild.models import BuildContext, BuildRequest
from justbuild.workflow import ExecutionState, NodeResult, RetryPolicy, TerminalState, WorkflowGraph, WorkflowNode, WorkflowRuntime


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


if __name__ == "__main__":
    unittest.main()
