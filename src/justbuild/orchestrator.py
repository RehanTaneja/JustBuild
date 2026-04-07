from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .agents.architecture import ArchitectureAgent
from .agents.base import AgentDependencies
from .agents.evaluation import EvaluationAgent
from .agents.implementation import ImplementationAgent
from .agents.specification import SpecificationAgent
from .agents.testing import TestingAgent
from .models import BuildContext, BuildRequest, Milestone, TaskStatus
from .observability import BuildLogger, write_build_summary
from .reporting import write_final_report

"""
This is the brain or the pipeline manager. It:
- Creates all the agents
- Decides order of execution
- Handles retries
- Tracks progress
- Logs everything
- Produces final output
Mental Modal: User Idea -> Orchestrator -> Agents -> Iterations -> Final Result
"""

# This class is basically the controller of the entire workflow
class OrchestratorAgent:
    name = "orchestrator-agent"

    def __init__(self, product_idea: str, output_root: Path, max_retries: int = 2) -> None:
        request = BuildRequest(product_idea=product_idea, output_root=output_root) # Create request
        self.context = BuildContext(request=request) # Create global system context
        self.logger = BuildLogger(self.context) # Create logger
        deps = AgentDependencies(context=self.context, logger=self.logger) # Creates shared dependencies
        # Initializes all agents
        self.specification_agent = SpecificationAgent(deps)
        self.architecture_agent = ArchitectureAgent(deps)
        self.implementation_agent = ImplementationAgent(deps)
        self.testing_agent = TestingAgent(deps)
        self.evaluation_agent = EvaluationAgent(deps)
        self.max_retries = max_retries
        self._create_milestones()

    # This is the main pipeline and the full workflow
    def run(self) -> BuildContext:
        with self.logger.timed(self.name, "Initialized orchestration workflow", "orchestration", 0): # Phase 1: Specifications
            self._update_milestone("Discovery & Planning", TaskStatus.IN_PROGRESS)
            specification = self.specification_agent.run(iteration=1)
            self._update_milestone("Discovery & Planning", TaskStatus.COMPLETED, title=specification.title)

        with self.logger.timed(self.name, "Produced architecture plan", "orchestration", 1): # Phase 2 Architecture
            self._update_milestone("Architecture", TaskStatus.IN_PROGRESS)
            self.architecture_agent.run(iteration=1)
            self._update_milestone("Architecture", TaskStatus.COMPLETED)

        planning_feedback = self._detect_architecture_issues() # Phase 3 Architecture feedback loop
        if planning_feedback:
            self.context.iterations.append({"iteration": "planning-refinement", "events": planning_feedback})
            with self.logger.timed(self.name, "Refined specification after architecture review", "orchestration", 1):
                self._update_milestone("Discovery & Planning", TaskStatus.RETRYING, feedback_count=len(planning_feedback))
                specification = self.specification_agent.run(iteration=2, architecture_feedback=planning_feedback)
                self._update_milestone("Discovery & Planning", TaskStatus.COMPLETED, title=specification.title)

            with self.logger.timed(self.name, "Rebuilt architecture after planning refinement", "orchestration", 2):
                self._update_milestone("Architecture", TaskStatus.RETRYING, feedback_count=len(planning_feedback))
                self.architecture_agent.run(iteration=2)
                self._update_milestone("Architecture", TaskStatus.COMPLETED)

        failure_reports = []
        for attempt in range(1, self.max_retries + 2): # Phase 4 Implementation Loop
            iteration_log = {"iteration": attempt, "events": []}
            self.context.iterations.append(iteration_log)

            with self.logger.timed(self.name, f"Implementation pass {attempt}", "orchestration", attempt): # Step 1 Implementation
                self._update_milestone("Implementation", TaskStatus.IN_PROGRESS, attempt=attempt)
                self.implementation_agent.run(iteration=attempt, failure_reports=failure_reports)
                self._update_milestone("Implementation", TaskStatus.COMPLETED, attempt=attempt)
                iteration_log["events"].append("implementation_completed")

            with self.logger.timed(self.name, f"Testing pass {attempt}", "orchestration", attempt): # Step 2 Testing
                self._update_milestone("Testing", TaskStatus.IN_PROGRESS, attempt=attempt)
                test_result = self.testing_agent.run(iteration=attempt)
                iteration_log["events"].append({"testing": asdict(test_result)})
                if test_result.passed: # If tests pass then break loop
                    self._update_milestone("Testing", TaskStatus.COMPLETED, attempt=attempt)
                    failure_reports = []
                    break

                failure_reports = test_result.failure_reports # If tests fail then feed feedback to the next loop
                self._update_milestone("Testing", TaskStatus.RETRYING, attempt=attempt, failures=len(failure_reports))

            if attempt > self.max_retries:
                self._update_milestone("Testing", TaskStatus.FAILED, attempt=attempt)
                break

        with self.logger.timed(self.name, "Evaluation completed", "orchestration", len(self.context.iterations)): # Phase 5 evaluation
            self._update_milestone("Evaluation", TaskStatus.IN_PROGRESS)
            self.evaluation_agent.run(iteration=len(self.context.iterations))
            self._update_milestone("Evaluation", TaskStatus.COMPLETED)

        with self.logger.timed(self.name, "Persisted build summary", "observability", len(self.context.iterations)): # Phase 6 Build final output summaries
            write_build_summary(self.context)
            write_final_report(self.context)

        return self.context

    # Creates all milestones in BuildContext
    def _create_milestones(self) -> None:
        self.context.milestones = [
            Milestone(name="Discovery & Planning", description="Turn the product idea into a working spec.", owner=self.specification_agent.name),
            Milestone(name="Architecture", description="Define structure, boundaries, and tradeoffs.", owner=self.architecture_agent.name),
            Milestone(name="Implementation", description="Generate the working prototype.", owner=self.implementation_agent.name),
            Milestone(name="Testing", description="Validate functionality and surface failures.", owner=self.testing_agent.name),
            Milestone(name="Evaluation", description="Assess quality, risk, and technical debt.", owner=self.evaluation_agent.name),
        ]

    # Tracks all milestones in BuildContext
    def _update_milestone(self, name: str, status: TaskStatus, **metadata: object) -> None:
        for milestone in self.context.milestones:
            if milestone.name == name:
                milestone.status = status
                if status == TaskStatus.RETRYING:
                    milestone.retries += 1
                milestone.metadata.update(metadata)
                return
        raise KeyError(f"Unknown milestone: {name}")

    # Detect if architecture doesn't match specifications
    def _detect_architecture_issues(self) -> list[str]:
        specification = self.context.specification
        architecture = self.context.architecture
        if specification is None or architecture is None:
            return []

        issues: list[str] = []
        if len(specification.missing_requirements) >= 3:
            issues.append("High requirement ambiguity detected; narrow scope with explicit prototype assumptions.")
        if not any("mock" in tradeoff.lower() for tradeoff in architecture.design_tradeoffs):
            issues.append("Architecture should document how prototype seams avoid premature backend lock-in.")
        return issues

"""
Mental Model:
Spec    --->   Architecture
        |  (feedback loop if bad)
        v
Implementation ---> Testing (retry loop if fails)
        |
        V
Evaluation ---> Reports
"""