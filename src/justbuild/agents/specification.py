from __future__ import annotations

import re

from .base import BaseAgent
from ..models import ProductSpecification

"""
Converts idea --> structured product specification. It generates:
- requirements
- features
- API contracts
- assumptions
- missing requirements
"""

class SpecificationAgent(BaseAgent):
    name = "specification-agent"

    def run(self, iteration: int, architecture_feedback: list[str] | None = None) -> ProductSpecification:
        idea = self.context.request.product_idea.strip()
        title = self._title_from_idea(idea)
        focus_terms = self._extract_focus_terms(idea)
        requirements = [
            f"Deliver a working prototype for {title}.",
            "Provide a clear landing experience explaining the product value.",
            "Allow a user to complete at least one meaningful end-to-end flow.",
            "Use a modular structure so future backend or data layers can be added safely.",
            "Emit machine-readable build metadata for downstream automation.",
        ]
        features = [
            f"Overview dashboard tailored to {focus_terms[0]}",
            "Primary workflow form with validation",
            "Saved items list with lightweight state management",
            "Progress/status panel for prototype transparency",
            "Configurable API integration seam for future production expansion",
        ]
        user_stories = [
            f"As a user, I want to understand how {title} helps me within seconds of landing on the app.",
            "As a user, I want to submit key information and receive a useful generated response.",
            "As a user, I want to review previously created items without losing context.",
            "As an operator, I want enough structure to extend the prototype into a production system.",
        ]
        api_contracts = [
            "POST /api/intake -> accepts product-specific form input and returns a generated result payload.",
            "GET /api/items -> returns recent generated items for the current session.",
            "GET /api/health -> returns service health and version metadata.",
        ]
        assumptions = [
            "The first release can use mocked or in-memory data instead of a persistent database.",
            "Authentication, billing, and external compliance workflows are deferred.",
            "A browser-based prototype is sufficient to demonstrate product value.",
        ]
        constraints = [
            "Must remain runnable with standard Python tooling in this repository.",
            "Generated code should favor readability and extensibility over framework complexity.",
            "The planning layer should flag unknowns instead of guessing hidden business policy.",
        ]
        missing_requirements = [
            "No explicit target user persona was provided.",
            "Non-functional requirements such as latency, SSO, and compliance scope are unspecified.",
            "No preferred frontend/backend framework was mandated.",
        ]
        if architecture_feedback:
            assumptions.append(
                "Architecture review narrowed the build to a browser-first MVP with mocked services and extensible seams."
            )
            constraints.append("Open architecture questions should be resolved through documented assumptions before code generation.")
            missing_requirements = [
                "Enterprise non-functional requirements such as compliance and SSO remain unspecified."
            ]
            requirements.append("Document provisional defaults when requirements are incomplete so downstream agents can keep moving.")
        spec = ProductSpecification(
            title=title,
            product_summary=f"{title} is a concept prototype generated from the idea: {idea}",
            requirements=requirements,
            features=features,
            user_stories=user_stories,
            api_contracts=api_contracts,
            assumptions=assumptions,
            constraints=constraints,
            missing_requirements=missing_requirements,
        )
        self.context.specification = spec
        return spec

    def _title_from_idea(self, idea: str) -> str:
        clean = re.sub(r"\s+", " ", idea).strip().rstrip(".")
        if len(clean) <= 72:
            return clean[:1].upper() + clean[1:]
        return clean[:69].rstrip() + "..."

    def _extract_focus_terms(self, idea: str) -> list[str]:
        words = [word.lower() for word in re.findall(r"[a-zA-Z]{4,}", idea)]
        unique_words: list[str] = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        return unique_words[:3] or ["product"]
