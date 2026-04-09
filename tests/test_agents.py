from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from justbuild.agents.base import AgentDependencies
from justbuild.agents.debugging import DebuggingAgent
from justbuild.llm import LLMClient
from justbuild.models import BuildContext, BuildRequest, FailureReport
from justbuild.observability import BuildLogger
from justbuild.orchestrator import OrchestratorAgent
from tests.support import FakeLLMClient, debugging_response, default_responses

"""
A Python unittest file that verifies that the pipeline works end-to-end. 
"""

# Groups all tests together, basically a checklist of everything that should not break
class MultiAgentSystemTests(unittest.TestCase):

    # Main pipeline test
    def test_end_to_end_build_generates_prototype_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir: # Creates a fake folder which is automatically deleted after the test
            orchestrator = OrchestratorAgent(
                product_idea="Collaborative roadmap planner for product teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
            )
            context = orchestrator.run() # Runs the full pipeline and returns the build context.

            self.assertIsNotNone(context.specification)
            self.assertIsNotNone(context.architecture)
            self.assertIsNotNone(context.implementation)
            self.assertIsNotNone(context.testing)
            self.assertIsNotNone(context.evaluation)
            self.assertTrue(context.testing.passed)
            self.assertTrue((context.implementation.prototype_dir / "index.html").exists())
            self.assertTrue(context.build_summary_path.exists())
            self.assertTrue(context.final_report_path.exists())

    def test_orchestrator_tracks_milestone_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Internal support copilot for operations teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
            ).run()
            statuses = {milestone.name: milestone.status.value for milestone in context.milestones}
            self.assertEqual(statuses["Discovery & Planning"], "completed")
            self.assertEqual(statuses["Architecture"], "completed")
            self.assertEqual(statuses["Implementation"], "completed")
            self.assertEqual(statuses["Testing"], "completed")
            self.assertEqual(statuses["Debugging"], "pending")
            self.assertEqual(statuses["Evaluation"], "completed")

    def test_invalid_specification_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OrchestratorAgent(
                product_idea="Internal support copilot for operations teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=["not-json"]),
            )
            with self.assertRaises(ValueError):
                orchestrator.run()

    def test_implementation_retries_after_invalid_llm_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = default_responses()
            responses = [
                base[0],
                base[1],
                '{"notes": ["broken"], "files": {"index.html": "missing rest"}}',
                base[2],
                base[3],
                base[4],
            ]
            orchestrator = OrchestratorAgent(
                product_idea="Planning assistant",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=responses),
            )
            context = orchestrator.run()

            self.assertTrue(context.testing.passed)
            implementation_milestone = next(item for item in context.milestones if item.name == "Implementation")
            self.assertGreaterEqual(implementation_milestone.retries, 1)

    def test_debugging_agent_returns_fix_plan(self) -> None:
        context = BuildContext(
            request=BuildRequest(
                product_idea="Planning assistant",
                output_root=Path(tempfile.gettempdir()),
                llm_provider="openai",
                llm_model="fake-model",
                llm_backend_type="cloud",
            )
        )
        logger = BuildLogger(context)
        agent = DebuggingAgent(
            AgentDependencies(
                context=context,
                logger=logger,
                llm=FakeLLMClient(
                    responses=[
                        debugging_response(
                            failure_groups=["missing_file"],
                            file_changes=["Create README.md with the expected prototype instructions."],
                            priority_order=["README.md"],
                        )
                    ]
                ),
            )
        )

        fix_plan = agent.run(
            iteration=1,
            failure_reports=[FailureReport(source="file-check", summary="Missing required prototype file: README.md", details=["Expected README.md to exist."])],
        )

        self.assertEqual(fix_plan.failure_groups, ["missing_file"])
        self.assertIn("README", " ".join(fix_plan.priority_order + fix_plan.file_changes))

    def test_failed_testing_triggers_debugging_and_records_fix_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = default_responses()
            broken_implementation = (
                '{"notes":["Broken implementation"],'
                '"files":{"index.html":"<!DOCTYPE html><html><body><h1>Collaborative roadmap planner</h1></body></html>",'
                '"styles.css":":root { --accent: #0d6f63; }",'
                '"app.js":"console.log(\\"missing flow\\");",'
                '"README.md":"# Collaborative roadmap planner\\n"}}'
            )
            fixed_implementation = base[2]
            responses = [
                base[0],
                base[1],
                broken_implementation,
                base[3],
                debugging_response(failure_groups=["content_mismatch"]),
                fixed_implementation,
                base[3],
                base[4],
            ]
            context = OrchestratorAgent(
                product_idea="Planning assistant",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=responses),
            ).run()

            self.assertTrue(context.testing.passed)
            self.assertIsNotNone(context.debugging)
            self.assertEqual(context.debugging.failure_groups, ["content_mismatch"])
            debugging_milestone = next(item for item in context.milestones if item.name == "Debugging")
            self.assertEqual(debugging_milestone.status.value, "completed")
            self.assertGreaterEqual(debugging_milestone.retries, 0)


if __name__ == "__main__":
    unittest.main()

"""
Mental Model:
Spins up a fake project idea ---> Run the entire pipeline ---> Checks if it actually produced something real.
"""
