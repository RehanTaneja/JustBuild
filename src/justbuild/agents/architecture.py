from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import ArchitecturePlan
from ..prompts import ARCHITECTURE_SCHEMA, architecture_system_prompt, architecture_user_prompt
from ..validation import JSONValidationError, parse_architecture_plan

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

        prompt = architecture_user_prompt(spec)
        system_prompt = architecture_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=ARCHITECTURE_SCHEMA)
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
            self.logger.log(
                self.name,
                "Architecture generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "architecture", "error": str(exc)},
            )
            raise ValueError(f"Architecture agent failed: {exc}") from exc
        self.context.architecture = plan
        return plan
