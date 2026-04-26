from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import ArchitecturePlan, EvaluationReport, ProductSpecification, TestResult
from ..prompts import (
    EVALUATION_SCHEMA,
    evaluation_draft_system_prompt,
    evaluation_draft_user_prompt,
    evaluation_system_prompt,
    evaluation_user_prompt,
)
from ..validation import JSONValidationError, parse_evaluation_report, parse_json_object, normalize_string_list, require_keys

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
        specification = self.context.specification
        if implementation is None or testing is None or architecture is None or specification is None:
            raise ValueError("Implementation, testing, and architecture are required before evaluation.")

        report = self.generate_report(specification, architecture, testing, iteration)
        self.context.evaluation = report
        return report

    def generate_report(
        self,
        specification: ProductSpecification,
        architecture: ArchitecturePlan,
        testing: TestResult,
        iteration: int,
        emit_logs: bool = True,
    ) -> EvaluationReport:
        prompt = evaluation_user_prompt(specification, architecture, testing, memory=self.context.memory)
        system_prompt = evaluation_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=EVALUATION_SCHEMA)
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Generated evaluation via LLM",
                    "llm_call",
                    iteration,
                    0,
                    metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "evaluation"},
                )
            report = parse_evaluation_report(response)
        except (LLMError, JSONValidationError) as exc:
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Evaluation generation failed",
                    "llm_failure",
                    iteration,
                    0,
                    metadata={"prompt_type": "evaluation", "error": str(exc)},
                )
            raise ValueError(f"Evaluation agent failed: {exc}") from exc
        return report

    def generate_draft(
        self,
        draft_name: str,
        draft_fields: list[str],
        specification: ProductSpecification,
        architecture: ArchitecturePlan,
        testing: TestResult,
        iteration: int,
        emit_logs: bool = True,
    ) -> dict[str, list[str]]:
        prompt = evaluation_draft_user_prompt(
            draft_fields,
            specification,
            architecture,
            testing,
            memory=self.context.memory,
        )
        system_prompt = evaluation_draft_system_prompt(draft_name)
        response_schema = {"type": "object", "required": draft_fields}
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=response_schema)
            if emit_logs:
                self.logger.log(
                    self.name,
                    f"Generated {draft_name} evaluation draft via LLM",
                    "llm_call",
                    iteration,
                    0,
                    metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": f"evaluation_{draft_name}"},
                )
            payload = parse_json_object(response)
            require_keys(payload, draft_fields)
            return {field: normalize_string_list(payload[field], field) for field in draft_fields}
        except (LLMError, JSONValidationError) as exc:
            if emit_logs:
                self.logger.log(
                    self.name,
                    f"{draft_name} evaluation draft generation failed",
                    "llm_failure",
                    iteration,
                    0,
                    metadata={"prompt_type": f"evaluation_{draft_name}", "error": str(exc)},
                )
            raise ValueError(f"Evaluation draft {draft_name} failed: {exc}") from exc

    def merge_drafts(self, drafts: list[dict[str, list[str]]]) -> EvaluationReport:
        merged: dict[str, list[str]] = {
            "code_quality": [],
            "maintainability": [],
            "scalability_risks": [],
            "security_concerns": [],
            "refactoring_opportunities": [],
            "technical_debt": [],
            "risk_assessment": [],
        }
        for draft in drafts:
            for key, values in draft.items():
                merged[key].extend(values)
        return EvaluationReport(**merged)
