from __future__ import annotations

from pathlib import Path

from .base import BaseAgent
from ..models import FailureReport, TestResult


class TestingAgent(BaseAgent):
    name = "testing-agent"

    def run(self, iteration: int) -> TestResult:
        implementation = self.context.implementation
        spec = self.context.specification
        if implementation is None or implementation.prototype_dir is None or spec is None:
            raise ValueError("Implementation artifacts are required before testing.")

        prototype_dir = implementation.prototype_dir
        unit_results: list[str] = []
        integration_results: list[str] = []
        failure_reports: list[FailureReport] = []

        expected_files = ["index.html", "styles.css", "app.js", "README.md"]
        for file_name in expected_files:
            path = prototype_dir / file_name
            if path.exists():
                unit_results.append(f"PASS: required file present -> {file_name}")
            else:
                failure_reports.append(
                    FailureReport(
                        source="file-check",
                        summary=f"Missing required prototype file: {file_name}",
                        details=[f"Expected {path} to exist."],
                    )
                )

        self._assert_contains(prototype_dir / "index.html", spec.title, "HTML includes product title", unit_results, failure_reports)
        self._assert_contains(prototype_dir / "index.html", "Feature Breakdown", "HTML includes feature section", unit_results, failure_reports)
        self._assert_contains(prototype_dir / "app.js", "Generated Response", "JS includes result rendering flow", integration_results, failure_reports)
        self._assert_contains(prototype_dir / "styles.css", ":root", "CSS theme tokens exist", integration_results, failure_reports)

        passed = not failure_reports
        summary = "All validation checks passed." if passed else f"{len(failure_reports)} validation checks failed."
        result = TestResult(
            passed=passed,
            summary=summary,
            unit_results=unit_results,
            integration_results=integration_results,
            failure_reports=failure_reports,
        )
        self.context.testing = result
        return result

    def _assert_contains(
        self,
        path: Path,
        needle: str,
        success_label: str,
        successes: list[str],
        failures: list[FailureReport],
    ) -> None:
        if not path.exists():
            failures.append(
                FailureReport(
                    source="content-check",
                    summary=f"Unable to inspect missing file: {path.name}",
                    details=[f"{path} did not exist during validation."],
                )
            )
            return
        content = path.read_text(encoding="utf-8")
        if needle in content:
            successes.append(f"PASS: {success_label}")
        else:
            failures.append(
                FailureReport(
                    source="content-check",
                    summary=f"Expected content missing in {path.name}",
                    details=[f'Missing token "{needle}" in {path.name}.'],
                )
            )
