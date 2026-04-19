from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import FailureReport, FixPlan, ImplementationArtifacts
from ..prompts import IMPLEMENTATION_SCHEMA, implementation_system_prompt, implementation_user_prompt
from ..prototype import slugify, write_prototype_bundle
from ..validation import JSONValidationError, parse_implementation_bundle

"""
Takes spec + architecture --> generates files. It writes:
- Prototype (HTML/JS/CSS)
- Uses slugified project directory
"""

class ImplementationAgent(BaseAgent):
    name = "implementation-agent"

    def run(
        self,
        iteration: int,
        failure_reports: list[FailureReport] | None = None,
        fix_plan: FixPlan | None = None,
    ) -> ImplementationArtifacts:
        spec = self.context.specification
        architecture = self.context.architecture
        if spec is None or architecture is None:
            raise ValueError("Specification and architecture are required before implementation.")

        slug = slugify(spec.title)
        output_dir = self.context.request.output_root / slug
        prompt = implementation_user_prompt(
            spec,
            architecture,
            failure_reports,
            fix_plan=fix_plan,
            memory=self.context.memory,
        )
        system_prompt = implementation_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=IMPLEMENTATION_SCHEMA)
            self.logger.log(
                self.name,
                "Generated implementation bundle via LLM",
                "llm_call",
                iteration,
                0,
                metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "implementation"},
            )
            notes, file_bundle = parse_implementation_bundle(response)
        except (LLMError, JSONValidationError) as exc:
            self.logger.log(
                self.name,
                "Implementation generation failed",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "implementation", "error": str(exc)},
            )
            raise ValueError(f"Implementation agent failed: {exc}") from exc

        if failure_reports:
            notes.append(f"Refinement pass applied after {len(failure_reports)} reported failures.")
        if fix_plan is not None:
            notes.append(f"Applied debug-guided strategy: {fix_plan.strategy}")
        generated_files = write_prototype_bundle(output_dir, file_bundle)

        artifacts = ImplementationArtifacts(
            prototype_dir=output_dir / "prototype",
            generated_files=generated_files,
            notes=notes,
            file_bundle=file_bundle,
        )
        self.context.implementation = artifacts
        return artifacts
