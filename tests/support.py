from __future__ import annotations

import json


class FakeLLMClient:
    def __init__(self, responses: list[str] | None = None, **kwargs: str | None) -> None:
        self._responses = list(responses or default_responses())
        self.api_key = kwargs.get("api_key")
        self.local_model = kwargs.get("local_model")
        self.provider = kwargs.get("provider") or ("openai_compatible" if self.local_model else "openai")
        self.model = kwargs.get("model") or self.local_model or "fake-model"
        self.base_url = kwargs.get("base_url")
        self.timeout_s = int(kwargs.get("timeout_s") or 60)

    @property
    def backend_info(self):  # pragma: no cover - simple compatibility shim
        return type(
            "BackendInfo",
            (),
            {
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "backend_type": "local" if self.local_model else "cloud",
            },
        )()

    def generate(self, prompt: str, system_prompt: str | None = None, response_schema: dict | None = None) -> str:
        if not self._responses:
            raise AssertionError(f"No fake LLM response left for prompt: {prompt[:80]}")
        return self._responses.pop(0)


def default_responses() -> list[str]:
    return [
        json.dumps(
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
        json.dumps(
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
        json.dumps(
            {
                "notes": [
                    "Generated a browser prototype from the LLM output.",
                ],
                "files": {
                    "index.html": "<!DOCTYPE html><html><body><h1>Collaborative roadmap planner</h1><section><h2>Feature Breakdown</h2></section><script src=\"./app.js\"></script></body></html>",
                    "styles.css": ":root { --accent: #0d6f63; } body { margin: 0; }",
                    "app.js": "document.body.dataset.ready = 'true'; const message = 'Generated Response';",
                    "README.md": "# Collaborative roadmap planner\n\nOpen index.html in a browser.\n",
                },
            }
        ),
        json.dumps(
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
        json.dumps(
            {
                "code_quality": [
                    "The prototype keeps responsibilities separated.",
                ],
                "maintainability": [
                    "Typed dataclasses keep the workflow consistent.",
                ],
                "scalability_risks": [
                    "Sequential orchestration is a future bottleneck.",
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
        ),
    ]
