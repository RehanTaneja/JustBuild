from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from justbuild.orchestrator import OrchestratorAgent
from tests.support import FakeLLMClient, default_responses

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


if __name__ == "__main__":
    unittest.main()

"""
Mental Model:
Spins up a fake project idea ---> Run the entire pipeline ---> Checks if it actually produced something real.
"""
