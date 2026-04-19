from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import ProductSpecification
from ..prompts import SPECIFICATION_SCHEMA, specification_system_prompt, specification_user_prompt
from ..validation import JSONValidationError, parse_product_specification

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
        prompt = specification_user_prompt(idea, architecture_feedback, memory=self.context.memory)
        system_prompt = specification_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=SPECIFICATION_SCHEMA)
            self.logger.log(
                self.name,
                "Generated specification via LLM",
                "llm_call",
                iteration,
                0,
                metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "specification"},
            )
            spec = parse_product_specification(response)
        except (LLMError, JSONValidationError) as exc:
            self.logger.log(
                self.name,
                "Specification generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "specification", "error": str(exc)},
            )
            raise ValueError(f"Specification agent failed: {exc}") from exc
        self.context.specification = spec
        return spec
