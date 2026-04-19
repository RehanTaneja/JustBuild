from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import ArchitecturePlan, ArchitectureReview, ProductSpecification
from ..prompts import (
    ARCHITECTURE_REVIEW_SCHEMA,
    ARCHITECTURE_SCHEMA,
    architecture_review_system_prompt,
    architecture_review_user_prompt,
    architecture_system_prompt,
    architecture_user_prompt,
)
from ..validation import JSONValidationError, parse_architecture_plan, parse_architecture_review

"""
Converts spec --> system design document. It produces:
- Folder structure
- Service decomposition
- DB schema (in the future, not yet)
- tradeoffs + justification
"""

class ArchitectureAgent(BaseAgent):
    name = "architecture-agent"

    def run(self, iteration: int) -> ArchitecturePlan:
        spec = self.context.specification
        if spec is None:
            raise ValueError("Specification is required before architecture planning.")
        plan = self.generate_plan(spec, iteration)
        self.context.architecture = plan
        return plan

    def generate_plan(
        self,
        spec: ProductSpecification,
        iteration: int,
        emit_logs: bool = True,
    ) -> ArchitecturePlan:
        prompt = architecture_user_prompt(spec, memory=self.context.memory)
        system_prompt = architecture_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=ARCHITECTURE_SCHEMA)
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Generated architecture via LLM",
                    "llm_call",
                    iteration,
                    0,
                    metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "architecture"},
                )
            plan = parse_architecture_plan(response)
        except (LLMError, JSONValidationError) as exc:
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Architecture generation failed",
                    "llm_failure",
                    iteration,
                    0,
                    metadata={"prompt_type": "architecture", "error": str(exc)},
                )
            raise ValueError(f"Architecture agent failed: {exc}") from exc
        return plan

    def review_plan(
        self,
        spec: ProductSpecification,
        architecture: ArchitecturePlan | None,
        iteration: int,
        emit_logs: bool = True,
    ) -> ArchitectureReview:
        prompt = architecture_review_user_prompt(spec, architecture, memory=self.context.memory)
        system_prompt = architecture_review_system_prompt()
        try:
            response = self.llm.generate(
                prompt,
                system_prompt=system_prompt,
                response_schema=ARCHITECTURE_REVIEW_SCHEMA,
            )
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Generated architecture review via LLM",
                    "llm_call",
                    iteration,
                    0,
                    metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "architecture_review"},
                )
            return parse_architecture_review(response)
        except (LLMError, JSONValidationError) as exc:
            if emit_logs:
                self.logger.log(
                    self.name,
                    "Architecture review generation failed",
                    "llm_failure",
                    iteration,
                    0,
                    metadata={"prompt_type": "architecture_review", "error": str(exc)},
                )
            raise ValueError(f"Architecture review failed: {exc}") from exc
