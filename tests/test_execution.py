from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from justbuild.execution import run_node_validation, validate_api_contracts
from justbuild.models import ProductSpecification


class CompletedProcessStub:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ExecutionTests(unittest.TestCase):
    def test_missing_node_records_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prototype_dir = Path(tmp_dir)
            (prototype_dir / "app.js").write_text("console.log('ok')", encoding="utf-8")
            results, skipped, failures = run_node_validation(prototype_dir, "missing-node")

        self.assertFalse(results)
        self.assertTrue(skipped)
        self.assertFalse(failures)

    def test_node_runtime_error_produces_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prototype_dir = Path(tmp_dir)
            (prototype_dir / "app.js").write_text("throw new Error('boom')", encoding="utf-8")
            with patch("shutil.which", return_value="/usr/bin/node"), patch(
                "subprocess.run",
                return_value=CompletedProcessStub(returncode=1, stderr="Error: boom"),
            ):
                results, skipped, failures = run_node_validation(prototype_dir, "node")

        self.assertFalse(results)
        self.assertFalse(skipped)
        self.assertEqual(failures[0].source, "node-exec")

    def test_api_contract_schema_validation_reports_invalid_contract(self) -> None:
        spec = ProductSpecification(
            title="Spec",
            product_summary="Summary",
            requirements=["req"],
            features=["feat"],
            user_stories=["story"],
            api_contracts=["BROKEN CONTRACT"],
            assumptions=["assumption"],
            constraints=["constraint"],
            missing_requirements=["missing"],
        )

        results, failures = validate_api_contracts(spec)
        self.assertFalse(results)
        self.assertEqual(failures[0].source, "api-schema")


if __name__ == "__main__":
    unittest.main()
