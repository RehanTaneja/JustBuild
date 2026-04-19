from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from justbuild.orchestrator import OrchestratorAgent
from justbuild.publishing import GitHubPublisher, GitHubRepoInfo
from tests.support import FakeLLMClient, default_responses


class PublishingTests(unittest.TestCase):
    def test_github_publisher_assembles_publish_directory_and_commit_plan(self) -> None:
        recorded_commands: list[list[str]] = []

        class FakeRepoClient:
            def create_repo(self, repo_name: str, description: str, visibility: str) -> GitHubRepoInfo:
                return GitHubRepoInfo(
                    repo_name=repo_name,
                    repo_full_name=f"demo/{repo_name}",
                    repo_url=f"https://github.com/demo/{repo_name}",
                    clone_url=f"https://github.com/demo/{repo_name}.git",
                )

        def fake_command_runner(cmd: list[str], cwd: Path | None = None) -> str:
            recorded_commands.append(cmd)
            return ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Publisher assembly test",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            ).run()

            publisher = GitHubPublisher(repo_client=FakeRepoClient(), command_runner=fake_command_runner)
            result = publisher.publish(context)

            self.assertTrue(result.published)
            self.assertTrue((result.local_publish_dir / "README.md").exists())
            self.assertTrue((result.local_publish_dir / "prototype" / "index.html").exists())
            self.assertTrue((result.local_publish_dir / "build_summary.json").exists())
            self.assertTrue((result.local_publish_dir / "final_report.md").exists())
            self.assertGreaterEqual(len(result.commits), 2)
            self.assertTrue(any(cmd[:3] == ["git", "push", "-u"] for cmd in recorded_commands))


if __name__ == "__main__":
    unittest.main()
