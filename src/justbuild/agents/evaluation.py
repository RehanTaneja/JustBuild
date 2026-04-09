from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import EvaluationReport
from ..prompts import EVALUATION_SCHEMA, evaluation_system_prompt, evaluation_user_prompt
from ..validation import JSONValidationError, parse_evaluation_report

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

        prompt = evaluation_user_prompt(specification, architecture, testing)
        system_prompt = evaluation_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=EVALUATION_SCHEMA)
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
            self.logger.log(
                self.name,
                "Evaluation generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "evaluation", "error": str(exc)},
            )
            raise ValueError(f"Evaluation agent failed: {exc}") from exc
        self.context.evaluation = report
        return report
