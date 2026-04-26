from __future__ import annotations

from .base import BaseAgent
from ..llm import LLMError
from ..models import FailureReport, FixPlan, ImplementationArtifacts, ImplementationPlan, ImplementationPlanFile
from ..prompts import (
    IMPLEMENTATION_FILE_SCHEMA,
    IMPLEMENTATION_PLAN_SCHEMA,
    implementation_file_system_prompt,
    implementation_file_user_prompt,
    implementation_system_prompt,
    implementation_user_prompt,
)
from ..prototype import default_static_web_file_content, default_static_web_plan, slugify, write_prototype_bundle
from ..validation import JSONValidationError, parse_implementation_file, parse_implementation_plan

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
        implementation_plan = self._generate_implementation_plan(
            spec,
            architecture,
            iteration,
            failure_reports,
            fix_plan,
        )
        notes = list(implementation_plan.notes)
        file_bundle, file_notes = self._generate_files_from_plan(
            spec,
            architecture,
            implementation_plan,
            iteration,
            failure_reports,
            fix_plan,
        )
        notes.extend(file_notes)

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
            implementation_plan=implementation_plan,
        )
        self.context.implementation = artifacts
        return artifacts

    def _generate_implementation_plan(
        self,
        spec,
        architecture,
        iteration: int,
        failure_reports: list[FailureReport] | None,
        fix_plan: FixPlan | None,
    ) -> ImplementationPlan:
        prompt = implementation_user_prompt(
            spec,
            architecture,
            failure_reports,
            fix_plan=fix_plan,
            memory=self.context.memory,
        )
        system_prompt = implementation_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=IMPLEMENTATION_PLAN_SCHEMA)
            self.logger.log(
                self.name,
                "Generated implementation plan via LLM",
                "llm_call",
                iteration,
                0,
                metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "implementation_plan"},
            )
            return parse_implementation_plan(response)
        except (LLMError, JSONValidationError) as exc:
            self.logger.log(
                self.name,
                "Implementation plan generation failed; falling back when possible",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "implementation_plan", "error": str(exc)},
            )
            fallback_plan = default_static_web_plan(spec)
            self.logger.log(
                self.name,
                "Using default static_web implementation plan fallback",
                "implementation_fallback",
                iteration,
                0,
                metadata={"prototype_kind": fallback_plan.prototype_kind, "entrypoint": fallback_plan.entrypoint},
            )
            return fallback_plan

    def _generate_files_from_plan(
        self,
        spec,
        architecture,
        implementation_plan: ImplementationPlan,
        iteration: int,
        failure_reports: list[FailureReport] | None,
        fix_plan: FixPlan | None,
    ) -> tuple[dict[str, str], list[str]]:
        file_bundle: dict[str, str] = {}
        notes: list[str] = []
        pending_files = [file_spec for file_spec in implementation_plan.files if file_spec.required]
        attempted_repairs: set[str] = set()

        while pending_files:
            next_round: list[ImplementationPlanFile] = []
            for file_spec in pending_files:
                dependency_files = {
                    dependency: file_bundle[dependency]
                    for dependency in file_spec.depends_on
                    if dependency in file_bundle
                }
                if len(dependency_files) < len(file_spec.depends_on):
                    next_round.append(file_spec)
                    continue
                try:
                    generated_path, content, file_notes = self._generate_single_file(
                        spec,
                        architecture,
                        implementation_plan,
                        file_spec,
                        dependency_files,
                        iteration,
                        failure_reports,
                        fix_plan,
                    )
                    file_bundle[generated_path] = content
                    notes.extend(file_notes)
                except ValueError as exc:
                    fallback = default_static_web_file_content(file_spec.path, spec, architecture)
                    if implementation_plan.prototype_kind == "static_web" and fallback is not None:
                        file_bundle[file_spec.path] = fallback
                        notes.append(f"Used local fallback content for {file_spec.path}.")
                        self.logger.log(
                            self.name,
                            f"Fell back to local static_web content for {file_spec.path}",
                            "implementation_fallback",
                            iteration,
                            0,
                            metadata={"file_path": file_spec.path, "error": str(exc)},
                        )
                    else:
                        raise ValueError(f"Implementation agent failed while generating {file_spec.path}: {exc}") from exc
            if len(next_round) == len(pending_files):
                unresolved = [file_spec.path for file_spec in next_round]
                if tuple(unresolved) == tuple(sorted(attempted_repairs)):
                    raise ValueError(f"Implementation agent failed: unresolved file dependencies: {', '.join(unresolved)}")
                attempted_repairs.update(unresolved)
                for file_spec in next_round:
                    if file_spec.path in file_bundle:
                        continue
                    fallback = default_static_web_file_content(file_spec.path, spec, architecture)
                    if implementation_plan.prototype_kind == "static_web" and fallback is not None:
                        file_bundle[file_spec.path] = fallback
                        notes.append(f"Used local fallback content for dependent file {file_spec.path}.")
                break
            pending_files = next_round

        missing_required = [file_spec.path for file_spec in implementation_plan.files if file_spec.required and file_spec.path not in file_bundle]
        if missing_required:
            raise ValueError(f"Implementation agent failed: missing generated files after file-level recovery: {', '.join(missing_required)}")
        return file_bundle, notes

    def _generate_single_file(
        self,
        spec,
        architecture,
        implementation_plan: ImplementationPlan,
        file_spec: ImplementationPlanFile,
        dependency_files: dict[str, str],
        iteration: int,
        failure_reports: list[FailureReport] | None,
        fix_plan: FixPlan | None,
    ) -> tuple[str, str, list[str]]:
        prompt = implementation_file_user_prompt(
            spec,
            architecture,
            implementation_plan,
            file_spec.path,
            file_spec.purpose,
            dependency_files,
            failure_reports,
            fix_plan=fix_plan,
            memory=self.context.memory,
        )
        system_prompt = implementation_file_system_prompt()
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt, response_schema=IMPLEMENTATION_FILE_SCHEMA)
            generated_path, content, notes = parse_implementation_file(response)
            if generated_path != file_spec.path:
                raise JSONValidationError(
                    f"Generated file path '{generated_path}' did not match requested path '{file_spec.path}'"
                )
            self.logger.log(
                self.name,
                f"Generated implementation file {file_spec.path} via LLM",
                "llm_call",
                iteration,
                0,
                metadata={"provider": self.llm.backend_info.provider, "model": self.llm.backend_info.model, "prompt_type": "implementation_file", "file_path": file_spec.path},
            )
            return generated_path, content, notes
        except (LLMError, JSONValidationError, ValueError) as exc:
            self.logger.log(
                self.name,
                f"Implementation file generation failed for {file_spec.path}",
                "llm_failure",
                iteration,
                0,
                metadata={"prompt_type": "implementation_file", "file_path": file_spec.path, "error": str(exc)},
            )
            raise ValueError(str(exc)) from exc
