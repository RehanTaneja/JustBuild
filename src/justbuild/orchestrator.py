from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .agents.architecture import ArchitectureAgent
from .agents.base import AgentDependencies
from .agents.debugging import DebuggingAgent
from .agents.evaluation import EvaluationAgent
from .agents.implementation import ImplementationAgent
from .agents.specification import SpecificationAgent
from .agents.testing import TestingAgent
from .llm import LLMClient
from .memory import default_memory_path, load_build_memory, save_build_memory, update_build_memory
from .models import BuildContext, BuildRequest, FailureReport, FixPlan, GitHubPublishResult, Milestone, TaskStatus
from .observability import BuildLogger, write_build_summary
from .publishing import GitHubPublisher
from .reporting import write_final_report
from .workflow import ExecutionState, NodeResult, RetryPolicy, TerminalState, WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowRuntime

"""
This is the brain or the pipeline manager. It:
- Creates all the agents
- Builds a workflow graph
- Executes retries / branches / loops through a DAG runtime
- Tracks progress
- Logs everything
- Produces final output
"""


class OrchestratorAgent:
    name = "orchestrator-agent"

    def __init__(
        self,
        product_idea: str,
        output_root: Path,
        max_retries: int = 2,
        llm_client: LLMClient | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        local_model: str | None = None,
        timeout_s: int = 60,
        enable_playwright: bool = False,
        node_bin: str = "node",
        pytest_bin: str = "pytest",
        max_workers: int = 4,
        memory_path: Path | None = None,
        publish_to_github: bool = False,
        github_repo_name: str | None = None,
        github_repo_visibility: str = "public",
        publisher: GitHubPublisher | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient(
            api_key=api_key,
            local_model=local_model,
            provider=provider,
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
        )
        backend = self.llm.backend_info
        request = BuildRequest(
            product_idea=product_idea,
            output_root=output_root,
            llm_provider=backend.provider,
            llm_model=backend.model,
            llm_base_url=backend.base_url,
            llm_backend_type=backend.backend_type,
            llm_structured_output_mode=backend.structured_output_mode,
            llm_timeout_s=timeout_s,
            enable_playwright=enable_playwright,
            node_bin=node_bin,
            pytest_bin=pytest_bin,
            max_workers=max_workers,
            memory_path=memory_path or default_memory_path(output_root),
            publish_to_github=publish_to_github,
            github_repo_name=github_repo_name,
            github_repo_visibility=github_repo_visibility,
        )
        self.context = BuildContext(request=request)
        self.logger = BuildLogger(self.context)
        deps = AgentDependencies(context=self.context, logger=self.logger, llm=self.llm)
        self.specification_agent = SpecificationAgent(deps)
        self.architecture_agent = ArchitectureAgent(deps)
        self.implementation_agent = ImplementationAgent(deps)
        self.testing_agent = TestingAgent(deps)
        self.debugging_agent = DebuggingAgent(deps)
        self.evaluation_agent = EvaluationAgent(deps)
        self.publisher = publisher or GitHubPublisher()
        self.max_retries = max_retries
        self.max_workers = max(1, max_workers)
        self._create_milestones()

    def run(self) -> BuildContext:
        self.context.node_runs = []
        self.context.workflow_terminal_state = None
        self.context.github_publish = GitHubPublishResult(enabled=False, published=False)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            runtime = WorkflowRuntime(executor)
            graph = self._build_workflow_graph()
            state = runtime.run(graph, self.context, max_workers=self.max_workers)
        if state.terminal_state == TerminalState.FAILED_BLOCKING:
            last_error = next((run.error for run in reversed(self.context.node_runs) if getattr(run, "error", None)), "Workflow execution failed.")
            raise ValueError(last_error)
        return self.context

    def _build_workflow_graph(self) -> WorkflowGraph:
        nodes = {
            "load_memory": WorkflowNode(
                node_id="load_memory",
                node_type="system",
                handler=self._handle_load_memory,
            ),
            "specification": WorkflowNode(
                node_id="specification",
                node_type="agent",
                handler=self._handle_specification,
                dependencies=["load_memory"],
                retry_policy=RetryPolicy(max_attempts=2),
                milestone_name="Discovery & Planning",
            ),
            "architecture_plan": WorkflowNode(
                node_id="architecture_plan",
                node_type="agent",
                handler=self._handle_architecture_plan,
                dependencies=["specification"],
                retry_policy=RetryPolicy(max_attempts=2),
            ),
            "architecture_review": WorkflowNode(
                node_id="architecture_review",
                node_type="agent",
                handler=self._handle_architecture_review,
                dependencies=["specification"],
                retry_policy=RetryPolicy(max_attempts=2),
            ),
            "planning_refinement_gate": WorkflowNode(
                node_id="planning_refinement_gate",
                node_type="gate",
                handler=self._handle_planning_refinement_gate,
                dependencies=["architecture_plan", "architecture_review"],
                milestone_name="Architecture",
            ),
            "implementation": WorkflowNode(
                node_id="implementation",
                node_type="agent",
                handler=self._handle_implementation,
                dependencies=["planning_refinement_gate"],
                retry_policy=RetryPolicy(max_attempts=self.max_retries + 1),
                milestone_name="Implementation",
            ),
            "testing": WorkflowNode(
                node_id="testing",
                node_type="agent",
                handler=self._handle_testing,
                dependencies=["implementation"],
                retry_policy=RetryPolicy(max_attempts=self.max_retries + 1),
                milestone_name="Testing",
            ),
            "debugging": WorkflowNode(
                node_id="debugging",
                node_type="agent",
                handler=self._handle_debugging,
                dependencies=["testing"],
                retry_policy=RetryPolicy(max_attempts=1),
                condition=lambda context, state: bool(state.data.get("failure_reports")),
                milestone_name="Debugging",
            ),
            "evaluation_quality": WorkflowNode(
                node_id="evaluation_quality",
                node_type="agent",
                handler=self._handle_evaluation_quality,
                dependencies=["testing"],
                retry_policy=RetryPolicy(max_attempts=1),
                condition=self._evaluation_ready,
            ),
            "evaluation_risk": WorkflowNode(
                node_id="evaluation_risk",
                node_type="agent",
                handler=self._handle_evaluation_risk,
                dependencies=["testing"],
                retry_policy=RetryPolicy(max_attempts=1),
                condition=self._evaluation_ready,
            ),
            "evaluation_security": WorkflowNode(
                node_id="evaluation_security",
                node_type="agent",
                handler=self._handle_evaluation_security,
                dependencies=["testing"],
                retry_policy=RetryPolicy(max_attempts=1),
                condition=self._evaluation_ready,
            ),
            "evaluation_merge": WorkflowNode(
                node_id="evaluation_merge",
                node_type="merge",
                handler=self._handle_evaluation_merge,
                dependencies=["evaluation_quality", "evaluation_risk", "evaluation_security"],
                milestone_name="Evaluation",
            ),
            "persist_outputs": WorkflowNode(
                node_id="persist_outputs",
                node_type="system",
                handler=self._handle_persist_outputs,
                dependencies=["evaluation_merge"],
            ),
            "publish": WorkflowNode(
                node_id="publish",
                node_type="delivery",
                handler=self._handle_publish,
                dependencies=["persist_outputs"],
                condition=lambda context, state: context.request.publish_to_github,
                milestone_name="Publishing",
            ),
        }
        edges = [
            WorkflowEdge("load_memory", "specification"),
            WorkflowEdge("specification", "architecture_plan"),
            WorkflowEdge("specification", "architecture_review"),
            WorkflowEdge("architecture_plan", "planning_refinement_gate"),
            WorkflowEdge("architecture_review", "planning_refinement_gate"),
            WorkflowEdge("planning_refinement_gate", "implementation"),
            WorkflowEdge("testing", "debugging", label="failed"),
            WorkflowEdge("testing", "evaluation_quality", label="passed"),
            WorkflowEdge("testing", "evaluation_risk", label="passed"),
            WorkflowEdge("testing", "evaluation_security", label="passed"),
            WorkflowEdge("debugging", "implementation", label="rerun"),
            WorkflowEdge("evaluation_quality", "evaluation_merge"),
            WorkflowEdge("evaluation_risk", "evaluation_merge"),
            WorkflowEdge("evaluation_security", "evaluation_merge"),
            WorkflowEdge("evaluation_merge", "persist_outputs"),
            WorkflowEdge("persist_outputs", "publish"),
        ]
        return WorkflowGraph(nodes=nodes, edges=edges, entry_nodes=["load_memory"])

    def _handle_load_memory(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self.context.memory = load_build_memory(self.context.request.memory_path)
        self.logger.log(
            self.name,
            "Loaded build memory",
            "memory",
            0,
            0,
            metadata={"memory_path": str(self.context.request.memory_path)},
        )
        return NodeResult(activate_nodes=["specification"], metadata={"memory_path": str(self.context.request.memory_path)})

    def _handle_specification(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self._mark_milestone_start("Discovery & Planning", attempt)
        feedback = state.data.get("planning_feedback")
        specification = self.specification_agent.run(iteration=attempt, architecture_feedback=feedback)
        self._update_milestone("Discovery & Planning", TaskStatus.COMPLETED, title=specification.title)
        state.data.pop("planning_feedback", None)
        return NodeResult(activate_nodes=["architecture_plan", "architecture_review"])

    def _handle_architecture_plan(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self._mark_milestone_start("Architecture", attempt)
        specification = self.context.specification
        if specification is None:
            raise ValueError("Specification is required before architecture planning.")
        plan = self.architecture_agent.generate_plan(specification, iteration=attempt, emit_logs=False)
        self.context.architecture = plan
        self.logger.log(
            self.architecture_agent.name,
            "Generated architecture via LLM",
            "llm_call",
            attempt,
            0,
            metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "architecture"},
        )
        return NodeResult(activate_nodes=["planning_refinement_gate"])

    def _handle_architecture_review(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self._mark_milestone_start("Architecture", attempt)
        specification = self.context.specification
        if specification is None:
            raise ValueError("Specification is required before architecture review.")
        review = self.architecture_agent.review_plan(specification, architecture=None, iteration=attempt, emit_logs=False)
        self.context.architecture_review = review
        self.logger.log(
            self.architecture_agent.name,
            "Generated architecture review via LLM",
            "llm_call",
            attempt,
            0,
            metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "architecture_review"},
        )
        return NodeResult(activate_nodes=["planning_refinement_gate"])

    def _handle_planning_refinement_gate(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        issues = self._detect_architecture_issues()
        if issues:
            self.context.iterations.append({"iteration": "planning-refinement", "events": issues})
            state.data["planning_feedback"] = issues
            return NodeResult(
                activate_nodes=["specification"],
                reset_nodes=["specification", "architecture_plan", "architecture_review", "planning_refinement_gate"],
                metadata={"issues": issues},
            )
        self._update_milestone("Architecture", TaskStatus.COMPLETED)
        state.data["implementation_cycle"] = 1
        state.data.setdefault("failure_reports", [])
        state.data.setdefault("fix_plan", None)
        return NodeResult(activate_nodes=["implementation"])

    def _handle_implementation(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        cycle = int(state.data.get("implementation_cycle", 1))
        self._mark_milestone_start("Implementation", cycle)
        failure_reports = state.data.get("failure_reports") or []
        fix_plan = state.data.get("fix_plan")
        implementation = self.implementation_agent.run(
            iteration=cycle,
            failure_reports=failure_reports,
            fix_plan=fix_plan,
        )
        self._update_milestone("Implementation", TaskStatus.COMPLETED, attempt=cycle)
        self._record_iteration_event(cycle, "implementation_completed", [str(path) for path in implementation.generated_files])
        return NodeResult(activate_nodes=["testing"])

    def _handle_testing(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        cycle = int(state.data.get("implementation_cycle", 1))
        self._mark_milestone_start("Testing", cycle)
        test_result = self.testing_agent.run(iteration=cycle)
        self._record_iteration_event(cycle, "testing", asdict(test_result))
        if test_result.passed:
            self._update_milestone("Testing", TaskStatus.COMPLETED, attempt=cycle)
            state.data["failure_reports"] = []
            state.data["fix_plan"] = None
            return NodeResult(activate_nodes=["evaluation_quality", "evaluation_risk", "evaluation_security"])

        state.data["failure_reports"] = test_result.failure_reports
        self._update_milestone("Testing", TaskStatus.RETRYING, attempt=cycle, failures=len(test_result.failure_reports))
        return NodeResult(activate_nodes=["debugging"])

    def _handle_debugging(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        cycle = int(state.data.get("implementation_cycle", 1))
        self._mark_milestone_start("Debugging", cycle)
        failure_reports = state.data.get("failure_reports") or []
        fix_plan = self.debugging_agent.run(iteration=cycle, failure_reports=failure_reports)
        state.data["fix_plan"] = fix_plan
        self._record_iteration_event(cycle, "debugging", asdict(fix_plan))
        self._update_milestone(
            "Debugging",
            TaskStatus.COMPLETED,
            attempt=cycle,
            failure_groups=fix_plan.failure_groups,
            strategy=fix_plan.strategy,
        )
        if cycle >= self.max_retries + 1:
            self._update_milestone("Testing", TaskStatus.FAILED, attempt=cycle)
            return NodeResult(activate_nodes=["evaluation_quality", "evaluation_risk", "evaluation_security"], metadata={"failure_groups": fix_plan.failure_groups})

        state.data["implementation_cycle"] = cycle + 1
        return NodeResult(
            activate_nodes=["implementation"],
            reset_nodes=["implementation", "testing", "debugging"],
            metadata={"failure_groups": fix_plan.failure_groups},
        )

    def _handle_evaluation_quality(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        state.data["evaluation_quality"] = self.evaluation_agent.generate_draft(
            "quality",
            ["code_quality", "maintainability"],
            self.context.specification,
            self.context.architecture,
            self.context.testing,
            iteration=attempt,
            emit_logs=False,
        )
        self.logger.log(
            self.evaluation_agent.name,
            "Generated evaluation_quality evaluation draft via LLM",
            "llm_call",
            attempt,
            0,
            metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "evaluation_quality"},
        )
        return NodeResult(activate_nodes=["evaluation_merge"])

    def _handle_evaluation_risk(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        state.data["evaluation_risk"] = self.evaluation_agent.generate_draft(
            "risk",
            ["scalability_risks", "technical_debt", "risk_assessment"],
            self.context.specification,
            self.context.architecture,
            self.context.testing,
            iteration=attempt,
            emit_logs=False,
        )
        self.logger.log(
            self.evaluation_agent.name,
            "Generated evaluation_risk evaluation draft via LLM",
            "llm_call",
            attempt,
            0,
            metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "evaluation_risk"},
        )
        return NodeResult(activate_nodes=["evaluation_merge"])

    def _handle_evaluation_security(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        state.data["evaluation_security"] = self.evaluation_agent.generate_draft(
            "security",
            ["security_concerns", "refactoring_opportunities"],
            self.context.specification,
            self.context.architecture,
            self.context.testing,
            iteration=attempt,
            emit_logs=False,
        )
        self.logger.log(
            self.evaluation_agent.name,
            "Generated evaluation_security evaluation draft via LLM",
            "llm_call",
            attempt,
            0,
            metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "evaluation_security"},
        )
        return NodeResult(activate_nodes=["evaluation_merge"])

    def _handle_evaluation_merge(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self._mark_milestone_start("Evaluation", attempt)
        self.context.evaluation = self.evaluation_agent.merge_drafts(
            [
                state.data["evaluation_quality"],
                state.data["evaluation_risk"],
                state.data["evaluation_security"],
            ]
        )
        self._update_milestone("Evaluation", TaskStatus.COMPLETED)
        return NodeResult(activate_nodes=["persist_outputs"])

    def _handle_persist_outputs(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        update_build_memory(self.context)
        write_build_summary(self.context)
        write_final_report(self.context)
        save_build_memory(self.context.request.memory_path, self.context.memory)
        self.logger.log(
            self.name,
            "Persisted build outputs",
            "observability",
            attempt,
            0,
            metadata={"memory_path": str(self.context.request.memory_path)},
        )
        return NodeResult(activate_nodes=["publish"])

    def _handle_publish(self, state: ExecutionState, node: WorkflowNode, attempt: int) -> NodeResult:
        self._mark_milestone_start("Publishing", attempt)
        try:
            result = self.publisher.publish(self.context)
        except Exception as exc:
            result = GitHubPublishResult(
                enabled=True,
                published=False,
                repo_name=self.context.request.github_repo_name,
                branch="main",
                failure_reason=str(exc),
            )
            self.context.github_publish = result
            self.logger.log(
                self.name,
                "GitHub publishing failed",
                "publishing_failure",
                attempt,
                0,
                metadata={"error": str(exc)},
            )
            self._update_milestone("Publishing", TaskStatus.FAILED, error=str(exc))
            write_build_summary(self.context)
            write_final_report(self.context)
            return NodeResult(terminal_state=TerminalState.COMPLETED_WITH_PUBLISH_FAILURE, metadata={"error": str(exc)})

        self.context.github_publish = result
        self.logger.log(
            self.name,
            "Published build to GitHub",
            "publishing",
            attempt,
            0,
            metadata={"repo_url": result.repo_url, "repo_full_name": result.repo_full_name},
        )
        self._update_milestone(
            "Publishing",
            TaskStatus.COMPLETED,
            repo_url=result.repo_url,
            repo_full_name=result.repo_full_name,
            commits=len(result.commits),
        )
        write_build_summary(self.context)
        write_final_report(self.context)
        return NodeResult(terminal_state=TerminalState.COMPLETED)

    def _evaluation_ready(self, context: BuildContext, state: ExecutionState) -> bool:
        return context.testing is not None

    def _create_milestones(self) -> None:
        self.context.milestones = [
            Milestone(name="Discovery & Planning", description="Turn the product idea into a working spec.", owner=self.specification_agent.name),
            Milestone(name="Architecture", description="Define structure, boundaries, and tradeoffs.", owner=self.architecture_agent.name),
            Milestone(name="Implementation", description="Generate the working prototype.", owner=self.implementation_agent.name),
            Milestone(name="Testing", description="Validate functionality and surface failures.", owner=self.testing_agent.name),
            Milestone(name="Debugging", description="Diagnose failures and produce a fix plan.", owner=self.debugging_agent.name),
            Milestone(name="Evaluation", description="Assess quality, risk, and technical debt.", owner=self.evaluation_agent.name),
            Milestone(name="Publishing", description="Ship the completed build to GitHub.", owner=self.name),
        ]

    def _mark_milestone_start(self, milestone_name: str, attempt: int) -> None:
        milestone = next(item for item in self.context.milestones if item.name == milestone_name)
        status = TaskStatus.RETRYING if attempt > 1 or milestone.status in {TaskStatus.COMPLETED, TaskStatus.RETRYING} else TaskStatus.IN_PROGRESS
        self._update_milestone(milestone_name, status, attempt=attempt)

    def _update_milestone(self, name: str, status: TaskStatus, **metadata: object) -> None:
        for milestone in self.context.milestones:
            if milestone.name == name:
                milestone.status = status
                if status == TaskStatus.RETRYING:
                    milestone.retries += 1
                milestone.metadata.update(metadata)
                return
        raise KeyError(f"Unknown milestone: {name}")

    def _detect_architecture_issues(self) -> list[str]:
        specification = self.context.specification
        architecture = self.context.architecture
        architecture_review = self.context.architecture_review
        if specification is None or architecture is None:
            return []

        issues: list[str] = []
        if len(specification.missing_requirements) >= 3:
            issues.append("High requirement ambiguity detected; narrow scope with explicit prototype assumptions.")
        if not any("mock" in tradeoff.lower() for tradeoff in architecture.design_tradeoffs):
            issues.append("Architecture should document how prototype seams avoid premature backend lock-in.")
        if architecture_review is not None:
            issues.extend(architecture_review.recommendations)
            if not architecture_review.requires_refinement:
                issues = [issue for issue in issues if issue not in architecture_review.recommendations]
        return issues

    def _record_iteration_event(self, iteration: int, key: str, value: object) -> None:
        iteration_log = self._ensure_iteration_log(iteration)
        iteration_log["events"].append({key: value})

    def _ensure_iteration_log(self, iteration: int) -> dict[str, object]:
        for entry in self.context.iterations:
            if entry.get("iteration") == iteration:
                return entry
        entry = {"iteration": iteration, "events": []}
        self.context.iterations.append(entry)
        return entry
