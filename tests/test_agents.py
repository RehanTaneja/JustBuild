from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from justbuild.orchestrator import OrchestratorAgent


class MultiAgentSystemTests(unittest.TestCase):
    def test_end_to_end_build_generates_prototype_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OrchestratorAgent(
                product_idea="Collaborative roadmap planner for product teams",
                output_root=Path(tmp_dir),
            )
            context = orchestrator.run()

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
            ).run()
            statuses = {milestone.name: milestone.status.value for milestone in context.milestones}
            self.assertEqual(statuses["Discovery & Planning"], "completed")
            self.assertEqual(statuses["Architecture"], "completed")
            self.assertEqual(statuses["Implementation"], "completed")
            self.assertEqual(statuses["Testing"], "completed")
            self.assertEqual(statuses["Evaluation"], "completed")


if __name__ == "__main__":
    unittest.main()
