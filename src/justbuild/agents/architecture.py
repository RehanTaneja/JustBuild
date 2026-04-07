from __future__ import annotations

from .base import BaseAgent
from ..models import ArchitecturePlan

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

        plan = ArchitecturePlan(
            summary=(
                "Use a layered Python orchestration core that generates a static web prototype. "
                "This keeps the control plane testable while giving the output a concrete, user-facing artifact."
            ),
            folder_structure=[
                "src/justbuild/agents -> specialized agents with clean responsibilities",
                "src/justbuild/orchestrator.py -> workflow engine, retries, and iteration loop",
                "src/justbuild/prototype.py -> file generation utilities for the working prototype",
                "src/justbuild/observability.py -> timing, logs, and build summary emission",
                "build_output/<slug>/prototype -> generated product prototype files",
                "tests -> orchestration and agent verification tests",
            ],
            services=[
                "Planning service: specification and requirement synthesis",
                "Architecture service: structure, contracts, and tradeoff documentation",
                "Implementation service: code and asset generation",
                "Verification service: unit-style checks and generated artifact validation",
                "Evaluation service: quality, scalability, and security review",
            ],
            database_schema=[
                "sessions(id, title, created_at) [future persistent store]",
                "items(id, session_id, name, status, summary, created_at)",
                "audit_events(id, session_id, event_type, payload, created_at)",
            ],
            design_tradeoffs=[
                "Static prototype output is faster and safer than generating a dependency-heavy full stack app.",
                "Typed in-process coordination is simpler today, but the interfaces are ready for distributed execution later.",
                "Mocked APIs reduce setup friction while preserving service boundaries for future replacement.",
            ],
            justification=[
                "The generated prototype should be immediately inspectable in a browser.",
                "Clear module boundaries allow enterprise teams to swap LLM providers, runners, or test harnesses independently.",
                "A small control-plane footprint lowers operational risk while still demonstrating autonomous delivery.",
            ],
        )
        self.context.architecture = plan
        return plan
