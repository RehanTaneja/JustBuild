from __future__ import annotations

from .base import BaseAgent
from ..models import EvaluationReport

"""
Reads context after everything is done. It produces:
- risks
- technical debt
- scalability issues
- security concerns

This is basically a postmortem system.
"""

class EvaluationAgent(BaseAgent):
    name = "evaluation-agent"

    def run(self, iteration: int) -> EvaluationReport:
        implementation = self.context.implementation
        testing = self.context.testing
        architecture = self.context.architecture
        if implementation is None or testing is None or architecture is None:
            raise ValueError("Implementation, testing, and architecture are required before evaluation.")

        code_quality = [
            "Clear layer boundaries reduce cross-cutting complexity.",
            "Generated prototype code is readable and intentionally lightweight for fast iteration.",
            "Structured build summary enables downstream automation and auditability.",
        ]
        maintainability = [
            "Agents communicate through typed dataclasses rather than ad-hoc dictionaries.",
            "Prototype generation is isolated in a dedicated module, limiting orchestration churn.",
            "Each layer can be swapped for a remote service without rewriting the control flow.",
        ]
        scalability_risks = [
            "The current implementation runs agents sequentially in-process and would bottleneck for large portfolios.",
            "Static prototype generation is excellent for demos but insufficient for full transactional workloads.",
            "Retry strategy is simple and lacks differentiated policies by failure class.",
        ]
        security_concerns = [
            "Generated prototypes use no authentication and should not be exposed as production services without hardening.",
            "Future implementation agents must run in sandboxes before executing generated code or migrations.",
            "Build summaries may eventually require redaction if proprietary prompts or credentials are included.",
        ]
        refactoring_opportunities = [
            "Introduce a task queue and persistent state store for durable long-running workflows.",
            "Replace heuristic planning with domain-aware model prompts plus schema validation.",
            "Split testing into static analysis, unit execution, and browser/runtime verification stages.",
        ]
        technical_debt = [
            "Specification extraction relies on heuristics rather than a domain ontology or prompt templates.",
            "The prototype targets a browser-only experience and does not yet generate deployable backend services.",
            "Evaluation uses deterministic rubric checks instead of deeper semantic analysis.",
        ]
        risk_assessment = [
            "Low delivery risk for prototype generation in small projects.",
            "Moderate architecture risk if teams expect immediate production-grade backend code generation.",
            "High scaling risk unless orchestration becomes event-driven with durable work persistence.",
        ]
        if not testing.passed:
            code_quality.append("Code quality is currently blocked by failing validation checks.")
            risk_assessment.append("Current build is not releasable until validation failures are resolved.")

        report = EvaluationReport(
            code_quality=code_quality,
            maintainability=maintainability,
            scalability_risks=scalability_risks,
            security_concerns=security_concerns,
            refactoring_opportunities=refactoring_opportunities,
            technical_debt=technical_debt,
            risk_assessment=risk_assessment,
        )
        self.context.evaluation = report
        return report
