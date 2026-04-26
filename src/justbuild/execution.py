from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path

from .models import FailureReport, ProductSpecification

REPO_ROOT = Path(__file__).resolve().parents[2]
API_CONTRACT_PATTERN = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+/\S+(?:\s*->\s*.+)?$"
)


class PrototypeHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.h1_texts: list[str] = []
        self._current_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self._current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._current_tag == tag:
            self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._current_tag == "h1" and data.strip():
            self.h1_texts.append(data.strip())


def run_pytest_validation(pytest_bin: str) -> tuple[list[str], list[str], list[FailureReport]]:
    results: list[str] = []
    skipped: list[str] = []
    failures: list[FailureReport] = []

    if os.getenv("JUSTBUILD_DISABLE_PYTEST_EXECUTION") == "1":
        skipped.append("SKIP: pytest execution disabled for nested test process")
        return results, skipped, failures

    if shutil.which(pytest_bin) is None:
        skipped.append(f"SKIP: pytest binary not found -> {pytest_bin}")
        return results, skipped, failures

    env = dict(os.environ)
    env["JUSTBUILD_DISABLE_PYTEST_EXECUTION"] = "1"
    completed = subprocess.run(
        [pytest_bin, "tests", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    if completed.returncode == 0:
        results.append("PASS: pytest execution completed successfully")
    else:
        failures.append(
            FailureReport(
                source="python-exec",
                summary="Pytest execution failed",
                details=[completed.stdout.strip(), completed.stderr.strip()],
            )
        )
    return results, skipped, failures


def run_node_validation(
    prototype_dir: Path,
    node_bin: str,
    script_path: str = "app.js",
) -> tuple[list[str], list[str], list[FailureReport]]:
    results: list[str] = []
    skipped: list[str] = []
    failures: list[FailureReport] = []

    if shutil.which(node_bin) is None:
        skipped.append(f"SKIP: node binary not found -> {node_bin}")
        return results, skipped, failures

    harness_path = _write_node_harness(prototype_dir, script_path)
    try:
        completed = subprocess.run(
            [node_bin, str(harness_path)],
            cwd=prototype_dir,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0:
            results.append("PASS: node execution succeeded with DOM harness")
        else:
            failures.append(
                FailureReport(
                    source="node-exec",
                    summary=f"Node execution failed for prototype script {script_path}",
                    details=[completed.stdout.strip(), completed.stderr.strip()],
                )
            )
    finally:
        harness_path.unlink(missing_ok=True)
    return results, skipped, failures


def validate_html_rendering(index_path: Path, expected_title: str) -> tuple[list[str], list[FailureReport]]:
    results: list[str] = []
    failures: list[FailureReport] = []
    if not index_path.exists():
        failures.append(
            FailureReport(
                source="html-validate",
                summary="HTML validation failed because index.html was missing",
                details=[f"Expected {index_path} to exist."],
            )
        )
        return results, failures

    content = index_path.read_text(encoding="utf-8")
    parser = PrototypeHTMLParser()
    parser.feed(content)

    if "<!DOCTYPE html>" in content and "html" in parser.tags and "body" in parser.tags:
        results.append("PASS: HTML document structure is present")
    else:
        failures.append(
            FailureReport(
                source="html-validate",
                summary="HTML document structure is invalid",
                details=["Expected doctype, html, and body tags in index.html."],
            )
        )

    if expected_title in parser.h1_texts:
        results.append("PASS: HTML main heading matches the product title")
    else:
        failures.append(
            FailureReport(
                source="html-validate",
                summary="HTML main heading did not match the product title",
                details=[f'Expected h1 text "{expected_title}" in index.html.'],
            )
        )
    return results, failures


def validate_api_contracts(spec: ProductSpecification) -> tuple[list[str], list[FailureReport]]:
    results: list[str] = []
    failures: list[FailureReport] = []
    for contract in spec.api_contracts:
        if API_CONTRACT_PATTERN.match(contract):
            results.append(f"PASS: API contract validated -> {contract}")
        else:
            failures.append(
                FailureReport(
                    source="api-schema",
                    summary="API contract did not match the expected schema",
                    details=[f'Invalid contract: "{contract}"'],
                )
            )
    return results, failures


def run_playwright_validation(
    prototype_dir: Path,
    expected_title: str,
    enabled: bool,
    entrypoint: str = "index.html",
) -> tuple[list[str], list[str], list[FailureReport]]:
    results: list[str] = []
    skipped: list[str] = []
    failures: list[FailureReport] = []

    if not enabled:
        skipped.append("SKIP: Playwright browser validation disabled")
        return results, skipped, failures

    if importlib.util.find_spec("playwright.sync_api") is None:
        skipped.append("SKIP: Playwright is not installed")
        return results, skipped, failures

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import error path
        skipped.append(f"SKIP: Playwright import failed -> {exc}")
        return results, skipped, failures

    page_errors: list[str] = []
    console_errors: list[str] = []
    index_url = (prototype_dir / entrypoint).resolve().as_uri()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else None,
            )
            page.goto(index_url)
            page.wait_for_load_state("load")
            heading = page.locator("h1").text_content() or ""
            if expected_title in heading:
                results.append("PASS: Playwright verified the main heading")
            else:
                failures.append(
                    FailureReport(
                        source="browser-playwright",
                        summary="Playwright could not verify the page heading",
                        details=[f'Expected h1 text to include "{expected_title}" but found "{heading}".'],
                    )
                )
            if page.locator("text=Feature Breakdown").count() > 0:
                results.append("PASS: Playwright found the Feature Breakdown section")
            else:
                failures.append(
                    FailureReport(
                        source="browser-playwright",
                        summary="Playwright could not find the Feature Breakdown section",
                        details=["Expected visible text 'Feature Breakdown' on the generated page."],
                    )
                )
            browser.close()
    except Exception as exc:
        failures.append(
            FailureReport(
                source="browser-playwright",
                summary="Playwright browser validation failed",
                details=[str(exc)],
            )
        )
        return results, skipped, failures

    if page_errors or console_errors:
        failures.append(
            FailureReport(
                source="browser-playwright",
                summary="Playwright detected browser runtime errors",
                details=page_errors + console_errors,
            )
        )
    return results, skipped, failures


def _write_node_harness(prototype_dir: Path, script_path: str) -> Path:
    harness = """
function createElement() {
  return {
    className: "",
    innerHTML: "",
    textContent: "",
    dataset: {},
    value: "",
    children: [],
    appendChild(child) { this.children.push(child); },
    addEventListener() {},
  };
}

const elements = {
  "idea-form": createElement(),
  "idea-input": Object.assign(createElement(), { value: "" }),
  "result-panel": createElement(),
  "history-list": createElement(),
};

global.document = {
  body: createElement(),
  getElementById(id) {
    if (!elements[id]) {
      elements[id] = createElement();
    }
    return elements[id];
  },
  createElement,
};

global.window = { document: global.document };
global.navigator = { userAgent: "node" };

try {
  require("./__SCRIPT_PATH__");
  process.stdout.write("JUSTBUILD_NODE_HARNESS_OK");
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
""".replace("__SCRIPT_PATH__", script_path)
    fd, temp_path = tempfile.mkstemp(prefix="justbuild-node-harness-", suffix=".js", dir=prototype_dir)
    os.close(fd)
    path = Path(temp_path)
    path.write_text(harness.strip() + "\n", encoding="utf-8")
    return path
