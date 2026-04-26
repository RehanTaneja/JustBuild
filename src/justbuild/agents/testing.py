from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .base import BaseAgent
from ..concurrency import run_parallel
from ..execution import (
    run_node_validation,
    run_playwright_validation,
    run_pytest_validation,
    validate_api_contracts,
    validate_html_rendering,
)
from ..llm import LLMError
from ..models import FailureReport, TestResult
from ..prompts import TESTING_SCHEMA, testing_system_prompt, testing_user_prompt
from ..validation import JSONValidationError, build_failure_report, parse_testing_plan

"""
Static validation of generated files. It checks:
- file existence
- keyword presence in code
It also produces:
- TestResult
- Failure Report

This is synctactic + structural validation only.
"""

class TestingAgent(BaseAgent):
    name = "testing-agent"

    def run(self, iteration: int) -> TestResult:
        result = self.generate_result(iteration)
        self.context.testing = result
        return result

    def generate_result(self, iteration: int) -> TestResult:
        implementation = self.context.implementation
        spec = self.context.specification
        architecture = self.context.architecture
        if implementation is None or implementation.prototype_dir is None or spec is None or architecture is None:
            raise ValueError("Implementation artifacts are required before testing.")

        prototype_dir = implementation.prototype_dir
        implementation_plan = implementation.implementation_plan
        unit_results: list[str] = []
        integration_results: list[str] = []
        llm_checks: list[str] = []
        execution_results: list[str] = []
        schema_results: list[str] = []
        browser_results: list[str] = []
        skipped_checks: list[str] = []
        failure_reports: list[FailureReport] = []

        prompt = testing_user_prompt(spec, architecture, memory=self.context.memory)
        system_prompt = testing_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=TESTING_SCHEMA)
            unit_checks, integration_checks, failure_focus = parse_testing_plan(response)
            llm_checks.extend([f"LLM unit plan: {item}" for item in unit_checks])
            llm_checks.extend([f"LLM integration plan: {item}" for item in integration_checks])
            llm_checks.extend([f"LLM failure focus: {item}" for item in failure_focus])
            self.logger.log(
                self.name,
                "Generated testing checklist via LLM",
                "llm_call",
                iteration,
                0,
                metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "testing"},
            )
        except (LLMError, JSONValidationError) as exc:
            failure_reports.append(
                build_failure_report(
                    "testing-llm",
                    "Testing agent failed to produce a valid structured checklist",
                    [str(exc)],
                )
            )
            self.logger.log(
                self.name,
                "Testing checklist generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "testing", "error": str(exc)},
            )

        planned_files = implementation_plan.files if implementation_plan is not None else []
        expected_files = [file_spec.path for file_spec in planned_files if file_spec.required] or list(implementation.file_bundle.keys())
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

        if implementation_plan is not None and implementation_plan.prototype_kind == "static_web":
            entrypoint = prototype_dir / implementation_plan.entrypoint
            self._assert_contains(entrypoint, spec.title, "HTML includes product title", unit_results, failure_reports)
            self._assert_contains(entrypoint, "Feature Breakdown", "HTML includes feature section", unit_results, failure_reports)
            js_file = self._first_matching_file(planned_files, ".js")
            css_file = self._first_matching_file(planned_files, ".css")
            if js_file is not None:
                self._assert_contains(prototype_dir / js_file, "Generated Response", "JS includes result rendering flow", integration_results, failure_reports)
            if css_file is not None:
                self._assert_contains(prototype_dir / css_file, ":root", "CSS theme tokens exist", integration_results, failure_reports)

        max_workers = max(1, self.context.request.max_workers)
        node_script = self._first_matching_file(planned_files, ".js")
        entrypoint = implementation_plan.entrypoint if implementation_plan is not None else None
        parallel_tasks = {
            "api_schema": lambda: validate_api_contracts(spec),
            "pytest_validation": lambda: run_pytest_validation(self.context.request.pytest_bin),
        }
        if implementation_plan is None or implementation_plan.prototype_kind == "static_web":
            html_entrypoint = entrypoint or "index.html"
            parallel_tasks["html_validation"] = lambda: validate_html_rendering(prototype_dir / html_entrypoint, spec.title)
            if node_script is not None:
                parallel_tasks["node_validation"] = lambda: run_node_validation(prototype_dir, self.context.request.node_bin, node_script)
            parallel_tasks["playwright_validation"] = lambda: run_playwright_validation(
                prototype_dir=prototype_dir,
                expected_title=spec.title,
                enabled=self.context.request.enable_playwright,
                entrypoint=html_entrypoint,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for task in run_parallel(executor, parallel_tasks):
                self.logger.log(
                    self.name,
                    f"Completed parallel testing task: {task.name}",
                    "parallel_task",
                    iteration,
                    task.elapsed_ms,
                    metadata={"task": task.name},
                )
                if task.name == "api_schema":
                    contract_results, contract_failures = task.value
                    schema_results.extend(contract_results)
                    failure_reports.extend(contract_failures)
                elif task.name == "html_validation":
                    html_results, html_failures = task.value
                    browser_results.extend(html_results)
                    failure_reports.extend(html_failures)
                elif task.name == "pytest_validation":
                    pytest_results, pytest_skips, pytest_failures = task.value
                    execution_results.extend(pytest_results)
                    skipped_checks.extend(pytest_skips)
                    failure_reports.extend(pytest_failures)
                elif task.name == "node_validation":
                    node_results, node_skips, node_failures = task.value
                    execution_results.extend(node_results)
                    skipped_checks.extend(node_skips)
                    failure_reports.extend(node_failures)
                elif task.name == "playwright_validation":
                    playwright_results, playwright_skips, playwright_failures = task.value
                    browser_results.extend(playwright_results)
                    skipped_checks.extend(playwright_skips)
                    failure_reports.extend(playwright_failures)

        passed = not failure_reports
        summary = "All validation checks passed." if passed else f"{len(failure_reports)} validation checks failed."
        result = TestResult(
            passed=passed,
            summary=summary,
            unit_results=unit_results,
            integration_results=integration_results,
            llm_checks=llm_checks,
            execution_results=execution_results,
            schema_results=schema_results,
            browser_results=browser_results,
            skipped_checks=skipped_checks,
            failure_reports=failure_reports,
        )
        return result

    def _assert_contains(
        self,
        path,
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

    def _first_matching_file(self, files, suffix: str) -> str | None:
        for file_spec in files:
            if file_spec.path.endswith(suffix):
                return file_spec.path
        return None
