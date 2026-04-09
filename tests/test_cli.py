from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from justbuild.cli import main
from tests.support import FakeLLMClient

"""
CLI related testing here
"""

class CLITests(unittest.TestCase):
    def test_cli_outputs_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            with patch("sys.stdout", stdout), patch("justbuild.cli.LLMClient", FakeLLMClient):
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
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertGreaterEqual(payload["iterations"], 1)
        self.assertIn("prototype", payload["prototype_dir"])
        self.assertEqual(payload["llm_backend"]["provider"], "openai")

    def test_cli_uses_env_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            env = {
                "JUSTBUILD_LLM_PROVIDER": "openai_compatible",
                "JUSTBUILD_LLM_LOCAL_MODEL": "llama-test",
                "JUSTBUILD_LLM_BASE_URL": "http://localhost:11434/v1",
            }
            with patch.dict(os.environ, env, clear=False), patch("sys.stdout", stdout), patch("justbuild.cli.LLMClient", FakeLLMClient):
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


if __name__ == "__main__":
    unittest.main()
