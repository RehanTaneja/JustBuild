from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from justbuild.cli import main
from justbuild.llm import LLMConfigurationError
from justbuild.models import GitHubPublishResult
from justbuild.workflow import ExecutionState
from tests.support import FakeLLMClient

"""
CLI related testing here
"""

class CLITests(unittest.TestCase):
    def test_cli_outputs_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            stderr = StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch("justbuild.cli.LLMClient", FakeLLMClient):
                exit_code = main(
                    [
                        "AI launch checklist assistant for startup founders",
                        "--output-root",
                        str(Path(tmp_dir)),
                        "--provider",
                        "openai",
                        "--model",
                        "gpt-test",
                        "--api-key",
                        "test-key",
                        "--node-bin",
                        "missing-node",
                        "--pytest-bin",
                        "missing-pytest",
                        "--max-workers",
                        "2",
                        "--memory-path",
                        str(Path(tmp_dir) / "memory.json"),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIn("Starting specification", stderr.getvalue())
        self.assertTrue(payload["passed"])
        self.assertGreaterEqual(payload["iterations"], 1)
        self.assertIn("prototype", payload["prototype_dir"])
        self.assertEqual(payload["llm_backend"]["provider"], "openai")
        self.assertEqual(payload["testing_backend"]["node_bin"], "missing-node")
        self.assertEqual(payload["testing_backend"]["max_workers"], 2)
        self.assertTrue(payload["memory_path"].endswith("memory.json"))
        self.assertEqual(payload["llm_backend"]["structured_output_mode"], "strict_schema")
        self.assertEqual(payload["llm_backend"]["timeout_s"], 60)

    def test_cli_uses_env_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            env = {
                "JUSTBUILD_LLM_PROVIDER": "openai_compatible",
                "JUSTBUILD_LLM_LOCAL_MODEL": "llama-test",
                "JUSTBUILD_LLM_BASE_URL": "http://localhost:11434/v1",
                "JUSTBUILD_LLM_TIMEOUT_S": "75",
                "JUSTBUILD_ENABLE_PLAYWRIGHT": "1",
                "JUSTBUILD_NODE_BIN": "node-from-env",
                "JUSTBUILD_PYTEST_BIN": "pytest-from-env",
                "JUSTBUILD_MAX_WORKERS": "3",
                "JUSTBUILD_MEMORY_PATH": str(Path(tmp_dir) / "memory-from-env.json"),
            }
            with patch.dict(os.environ, env, clear=False), patch("sys.stdout", stdout), patch("justbuild.cli.LLMClient", FakeLLMClient), patch("justbuild.agents.testing.run_playwright_validation", return_value=([], ["SKIP: Playwright mocked as unavailable"], [])):
                exit_code = main(
                    [
                        "Ops assistant",
                        "--output-root",
                        str(Path(tmp_dir)),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["llm_backend"]["provider"], "openai_compatible")
        self.assertEqual(payload["llm_backend"]["model"], "llama-test")
        self.assertEqual(payload["llm_backend"]["backend_type"], "local")
        self.assertEqual(payload["llm_backend"]["structured_output_mode"], "best_effort_schema")
        self.assertEqual(payload["llm_backend"]["timeout_s"], 75)
        self.assertTrue(payload["testing_backend"]["enable_playwright"])
        self.assertEqual(payload["testing_backend"]["node_bin"], "node-from-env")
        self.assertEqual(payload["testing_backend"]["pytest_bin"], "pytest-from-env")
        self.assertEqual(payload["testing_backend"]["max_workers"], 3)
        self.assertTrue(payload["memory_path"].endswith("memory-from-env.json"))

    def test_cli_reports_github_publish_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()

            def fake_publish(self, context):
                publish_dir = context.implementation.prototype_dir.parent / "github_publish"
                publish_dir.mkdir(parents=True, exist_ok=True)
                return GitHubPublishResult(
                    enabled=True,
                    published=True,
                    repo_name="cli-prototype",
                    repo_full_name="demo/cli-prototype",
                    repo_url="https://github.com/demo/cli-prototype",
                    branch="main",
                    local_publish_dir=publish_dir,
                    commits=["feat: initial prototype generation", "docs: add final build summary and report"],
                )

            with patch("sys.stdout", stdout), patch("justbuild.cli.LLMClient", FakeLLMClient), patch("justbuild.orchestrator.GitHubPublisher.publish", fake_publish):
                exit_code = main(
                    [
                        "GitHub publish test",
                        "--output-root",
                        str(Path(tmp_dir)),
                        "--provider",
                        "openai",
                        "--model",
                        "gpt-test",
                        "--api-key",
                        "test-key",
                        "--node-bin",
                        "missing-node",
                        "--pytest-bin",
                        "missing-pytest",
                        "--publish-github",
                        "--github-repo-name",
                        "cli-prototype",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["github_publish"]["enabled"])
        self.assertTrue(payload["github_publish"]["published"])
        self.assertEqual(payload["github_publish"]["repo_full_name"], "demo/cli-prototype")

    def test_cli_reports_actionable_configuration_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr), patch(
            "justbuild.cli.OrchestratorAgent",
            side_effect=LLMConfigurationError("No LLM backend configured"),
        ):
            exit_code = main(["smoke test"])

        message = stderr.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("Configuration error: No LLM backend configured", message)
        self.assertIn("--provider openai --model <model> --api-key <key>", message)
        self.assertIn("--provider openai_compatible --local-model <model> --base-url <url>", message)

    def test_cli_failure_reports_log_paths(self) -> None:
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code = None
            failing_llm = lambda **kwargs: FakeLLMClient(responses=["not-json"], **kwargs)
            with patch("sys.stderr", stderr), patch("justbuild.cli.LLMClient", failing_llm):
                exit_code = main(
                    [
                        "Failure case",
                        "--output-root",
                        str(Path(tmp_dir)),
                        "--provider",
                        "openai",
                        "--model",
                        "gpt-test",
                        "--api-key",
                        "test-key",
                    ]
                )

        message = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("Build failed. See", message)

    def test_cli_accepts_explicit_llm_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            with patch("sys.stdout", stdout), patch("justbuild.cli.LLMClient", FakeLLMClient):
                exit_code = main(
                    [
                        "Timeout test",
                        "--output-root",
                        str(Path(tmp_dir)),
                        "--provider",
                        "openai",
                        "--model",
                        "gpt-test",
                        "--api-key",
                        "test-key",
                        "--llm-timeout-s",
                        "120",
                        "--node-bin",
                        "missing-node",
                        "--pytest-bin",
                        "missing-pytest",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["llm_backend"]["timeout_s"], 120)

    def test_cli_runtime_integrity_failure_reports_logs_and_no_success_payload(self) -> None:
        original_enqueue = ExecutionState.enqueue

        def corrupted_enqueue(self: ExecutionState, node_id: str) -> None:
            if node_id == "implementation":
                self.activated_nodes.add(node_id)
                return
            original_enqueue(self, node_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            stderr = StringIO()
            with patch.object(ExecutionState, "enqueue", new=corrupted_enqueue), patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch("justbuild.cli.LLMClient", FakeLLMClient):
                exit_code = main(
                    [
                        "Integrity failure test",
                        "--output-root",
                        str(Path(tmp_dir)),
                        "--provider",
                        "openai",
                        "--model",
                        "gpt-test",
                        "--api-key",
                        "test-key",
                        "--node-bin",
                        "missing-node",
                        "--pytest-bin",
                        "missing-pytest",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Build failed. See", stderr.getvalue())
        self.assertIn("Workflow integrity failure", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
