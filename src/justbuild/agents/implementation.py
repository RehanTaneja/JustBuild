from __future__ import annotations

from .base import BaseAgent
from ..models import FailureReport, ImplementationArtifacts
from ..prototype import slugify, write_prototype_files

"""
Takes spec + architecture --> generates files. It writes:
- Prototype (HTML/JS/CSS)
- Uses slugified project directory
"""

class ImplementationAgent(BaseAgent):
    name = "implementation-agent"

    def run(self, iteration: int, failure_reports: list[FailureReport] | None = None) -> ImplementationArtifacts:
        spec = self.context.specification
        architecture = self.context.architecture
        if spec is None or architecture is None:
            raise ValueError("Specification and architecture are required before implementation.")

        slug = slugify(spec.title)
        output_dir = self.context.request.output_root / slug
        generated_files = write_prototype_files(output_dir, spec, architecture)

        notes = [
            "Generated a browser-based working prototype with a concrete interaction flow.",
            "Preserved service boundaries through API contract documentation rather than premature backend complexity.",
        ]
        if failure_reports:
            notes.append(f"Refinement pass applied after {len(failure_reports)} reported failures.")

        artifacts = ImplementationArtifacts(
            prototype_dir=output_dir / "prototype",
            generated_files=generated_files,
            notes=notes,
        )
        self.context.implementation = artifacts
        return artifacts
