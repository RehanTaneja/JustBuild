from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import FailureReport, FixPlan
from ..prompts import DEBUGGING_SCHEMA, debugging_system_prompt, debugging_user_prompt
from ..validation import JSONValidationError, parse_fix_plan


class DebuggingAgent(BaseAgent):
    name = "debugging-agent"

    def run(self, iteration: int, failure_reports: list[FailureReport]) -> FixPlan:
        if not failure_reports:
            raise ValueError("Debugging agent requires failure reports.")

        prompt = debugging_user_prompt(
            failure_reports=failure_reports,
            implementation=self.context.implementation,
            testing=self.context.testing,
            specification=self.context.specification,
            architecture=self.context.architecture,
        )
        system_prompt = debugging_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=DEBUGGING_SCHEMA)
            self.logger.log(
                self.name,
                "Generated fix plan via LLM",
                "llm_call",
                iteration,
                0,
                metadata={
                    "provider": self.llm.backend_info.provider,
                    "model": self.llm.backend_info.model,
                    "prompt_type": "debugging",
                },
            )
            fix_plan = parse_fix_plan(response)
        except (LLMError, JSONValidationError) as exc:
            self.logger.log(
                self.name,
                "Debugging plan generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "debugging", "error": str(exc)},
            )
            raise ValueError(f"Debugging agent failed: {exc}") from exc

        self.context.debugging = fix_plan
        return fix_plan
