from __future__ import annotations

import copy
import json
import threading
from typing import Any


class FakeLLMClient:
    def __init__(self, responses: dict[str, Any] | list[str] | None = None, **kwargs: str | None) -> None:
        provided = responses or default_responses()
        self._responses = copy.deepcopy(provided)
        self._lock = threading.Lock()
        self.prompt_history: list[tuple[str | None, str]] = []
        self.api_key = kwargs.get("api_key")
        self.local_model = kwargs.get("local_model")
        self.provider = kwargs.get("provider") or ("openai_compatible" if self.local_model else "openai")
        self.model = kwargs.get("model") or self.local_model or "fake-model"
        self.base_url = kwargs.get("base_url")
        self.timeout_s = int(kwargs.get("timeout_s") or 60)

    @property
    def backend_info(self):  # pragma: no cover - simple compatibility shim
        mode = "best_effort_schema" if self.provider == "openai_compatible" else "strict_schema"
        return type(
            "BackendInfo",
            (),
            {
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "backend_type": "local" if self.local_model else "cloud",
                "backend_family": "openai_compatible" if self.local_model else self.provider,
                "structured_output_mode": mode,
                "capabilities_probed": False,
                "capability_source": "static",
                "capability_downgrade": None,
            },
        )()

    def generate(self, prompt: str, system_prompt: str | None = None, response_schema: dict | None = None) -> str:
        with self._lock:
            self.prompt_history.append((system_prompt, prompt))
            if isinstance(self._responses, list):
                if not self._responses:
                    raise AssertionError(f"No fake LLM response left for prompt: {prompt[:80]}")
                return self._responses.pop(0)

            key = self._classify_prompt(prompt, system_prompt)
            if key not in self._responses:
                raise AssertionError(f"No fake LLM response configured for key: {key}")
            value = self._responses[key]
            if isinstance(value, list):
                if not value:
                    raise AssertionError(f"No fake LLM responses left for key: {key}")
                return value.pop(0)
            if isinstance(value, str):
                return value
            raise AssertionError(f"Unsupported fake response value for key {key}: {type(value)!r}")

    def _classify_prompt(self, prompt: str, system_prompt: str | None) -> str:
        system = (system_prompt or "").lower()
        if "specification agent" in system:
            return "specification"
        if "architecture review agent" in system:
            return "architecture_review"
        if "architecture agent" in system:
            return "architecture_plan"
        if "implementation planning agent" in system:
            return "implementation_plan"
        if "implementation file agent" in system:
            marker = "Target file path: "
            if marker in prompt:
                target_path = prompt.split(marker, 1)[1].splitlines()[0].strip()
                return f"implementation_file:{target_path}"
            return "implementation_file"
        if "testing agent" in system:
            return "testing"
        if "debugging agent" in system:
            return "debugging"
        if "evaluation draft agent" in system:
            if "quality" in system:
                return "evaluation_quality"
            if "risk" in system:
                return "evaluation_risk"
            if "security" in system:
                return "evaluation_security"
        if "evaluation agent" in system:
            return "evaluation"
        raise AssertionError(f"Unable to classify fake LLM prompt: {prompt[:80]}")


def default_responses() -> dict[str, str]:
    evaluation_payload = json.dumps(
        {
            "code_quality": [
                "The prototype keeps responsibilities separated.",
            ],
            "maintainability": [
                "Typed dataclasses keep the workflow consistent.",
            ],
            "scalability_risks": [
                "Parallel orchestration improves throughput for larger builds.",
            ],
            "security_concerns": [
                "Authentication is not implemented yet.",
            ],
            "refactoring_opportunities": [
                "Split execution into worker processes.",
            ],
            "technical_debt": [
                "The implementation still targets static prototypes only.",
            ],
            "risk_assessment": [
                "Low risk for prototype generation, higher risk for production deployment.",
            ],
        }
    )
    return {
        "specification": json.dumps(
            {
                "title": "Collaborative roadmap planner",
                "product_summary": "A planning workspace for product teams.",
                "requirements": [
                    "Deliver a working prototype.",
                    "Support a clear planning flow.",
                ],
                "features": [
                    "Feature Breakdown dashboard",
                    "Idea intake flow",
                ],
                "user_stories": [
                    "As a PM, I want to create a plan quickly.",
                    "As a team member, I want to review priorities.",
                ],
                "api_contracts": [
                    "POST /api/intake",
                    "GET /api/items",
                ],
                "assumptions": [
                    "In-memory data is acceptable for v1.",
                ],
                "constraints": [
                    "Remain runnable with standard Python tooling.",
                ],
                "missing_requirements": [
                    "Persona detail should be refined later.",
                ],
            }
        ),
        "architecture_plan": json.dumps(
            {
                "summary": "Layered orchestration that generates a browser prototype.",
                "folder_structure": [
                    "src/justbuild/agents",
                    "src/justbuild/orchestrator.py",
                    "build_output/<slug>/prototype",
                ],
                "services": [
                    "Specification service",
                    "Architecture service",
                    "Implementation service",
                ],
                "database_schema": [
                    "items(id, title, status)",
                ],
                "design_tradeoffs": [
                    "Mock APIs keep the prototype fast to iterate.",
                ],
                "justification": [
                    "The build remains easy to test and inspect.",
                ],
            }
        ),
        "architecture_review": json.dumps(
            {
                "prototype_blockers": [],
                "retry_guidance": [],
                "requires_refinement": False,
            }
        ),
        "implementation_plan": json.dumps(
            {
                "prototype_kind": "static_web",
                "entrypoint": "index.html",
                "notes": [
                    "Generate a simple browser prototype with a small static asset set.",
                ],
                "files": [
                    {"path": "index.html", "purpose": "Browser entrypoint for the prototype UI.", "required": True},
                    {"path": "styles.css", "purpose": "Shared styling for the prototype UI.", "required": True},
                    {"path": "app.js", "purpose": "Client-side interaction logic.", "required": True, "depends_on": ["index.html"]},
                    {"path": "README.md", "purpose": "Instructions for opening the prototype.", "required": True},
                ],
            }
        ),
        "implementation_file:index.html": json.dumps(
            {
                "path": "index.html",
                "content": "<!DOCTYPE html><html><body><h1>Collaborative roadmap planner</h1><section><h2>Feature Breakdown</h2></section><script src=\"./app.js\"></script></body></html>",
                "notes": ["Created the primary browser entrypoint."],
            }
        ),
        "implementation_file:styles.css": json.dumps(
            {
                "path": "styles.css",
                "content": ":root { --accent: #0d6f63; } body { margin: 0; }",
                "notes": ["Created the shared theme stylesheet."],
            }
        ),
        "implementation_file:app.js": json.dumps(
            {
                "path": "app.js",
                "content": "document.body.dataset.ready = 'true'; const message = 'Generated Response';",
                "notes": ["Created the browser interaction logic."],
            }
        ),
        "implementation_file:README.md": json.dumps(
            {
                "path": "README.md",
                "content": "# Collaborative roadmap planner\n\nOpen index.html in a browser.\n",
                "notes": ["Documented how to run the prototype."],
            }
        ),
        "testing": json.dumps(
            {
                "unit_checks": [
                    "Verify required files exist.",
                ],
                "integration_checks": [
                    "Verify the generated response flow is present.",
                ],
                "failure_focus": [
                    "Watch for missing JSON fields.",
                ],
            }
        ),
        "evaluation_quality": evaluation_payload,
        "evaluation_risk": evaluation_payload,
        "evaluation_security": evaluation_payload,
        "evaluation": evaluation_payload,
        "debugging": debugging_response(),
    }


def debugging_response(
    failure_groups: list[str] | None = None,
    root_cause: str = "The generated bundle omitted required prototype content.",
    strategy: str = "Regenerate the failing files and satisfy the testing contract before the next pass.",
    file_changes: list[str] | None = None,
    priority_order: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "file_changes": file_changes or [
                "Update index.html to include the required feature section.",
                "Update app.js to include the generated response flow.",
            ],
            "root_cause": root_cause,
            "strategy": strategy,
            "failure_groups": failure_groups or ["content_mismatch"],
            "priority_order": priority_order or ["index.html", "app.js"],
        }
    )
