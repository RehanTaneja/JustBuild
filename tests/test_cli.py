from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from justbuild.cli import main


class CLITests(unittest.TestCase):
    def test_cli_outputs_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = StringIO()
            with patch("sys.stdout", stdout):
                exit_code = main(
                    [
                        "AI launch checklist assistant for startup founders",
                        "--output-root",
                        str(Path(tmp_dir)),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertGreaterEqual(payload["iterations"], 1)
        self.assertIn("prototype", payload["prototype_dir"])


if __name__ == "__main__":
    unittest.main()
