from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from justbuild.agents.base import AgentDependencies
from justbuild.agents.debugging import DebuggingAgent
from justbuild.llm import LLMClient
from justbuild.models import BuildContext, BuildRequest, FailureReport, GitHubPublishResult
from justbuild.observability import BuildLogger
from justbuild.orchestrator import OrchestratorAgent
from justbuild.publishing import GitHubPublishError
from tests.support import FakeLLMClient, debugging_response, default_responses

"""
A Python unittest file that verifies that the pipeline works end-to-end. 
"""

# Groups all tests together, basically a checklist of everything that should not break
class MultiAgentSystemTests(unittest.TestCase):

    class _MockHTTPResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _classify_provider_prompt(self, request_payload: dict) -> str:
        if "system" in request_payload:
            system = request_payload.get("system", "")
        elif "messages" in request_payload:
            system = next((message["content"] for message in request_payload["messages"] if message["role"] == "system"), "")
        else:
            system = ""
        lowered = system.lower()
        if "specification agent" in lowered:
            return "specification"
        if "architecture review agent" in lowered:
            return "architecture_review"
        if "architecture agent" in lowered:
            return "architecture_plan"
        if "implementation agent" in lowered:
            return "implementation"
        if "testing agent" in lowered:
            return "testing"
        if "debugging agent" in lowered:
            return "debugging"
        if "evaluation draft agent" in lowered:
            if "quality" in lowered:
                return "evaluation_quality"
            if "risk" in lowered:
                return "evaluation_risk"
            if "security" in lowered:
                return "evaluation_security"
        if "evaluation agent" in lowered:
            return "evaluation"
        raise AssertionError(f"Unable to classify request payload: {request_payload}")

    def _openai_compatible_responder(self, responses: dict[str, str]):
        def _responder(request_obj, *args, **kwargs):
            payload = json.loads(request_obj.data.decode("utf-8"))
            key = self._classify_provider_prompt(payload)
            content = f"```json\n{responses[key]}\n```"
            return self._MockHTTPResponse({"choices": [{"message": {"content": content}}]})

        return _responder

    def _anthropic_tool_responder(self, responses: dict[str, str]):
        def _responder(request_obj, *args, **kwargs):
            payload = json.loads(request_obj.data.decode("utf-8"))
            key = self._classify_provider_prompt(payload)
            content = json.loads(responses[key])
            return self._MockHTTPResponse(
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "justbuild_response",
                            "input": content,
                        }
                    ]
                }
            )

        return _responder

    def _openai_compatible_missing_key_then_repair_responder(self, responses: dict[str, str], target_key: str):
        call_count = {"count": 0}

        def _responder(request_obj, *args, **kwargs):
            payload = json.loads(request_obj.data.decode("utf-8"))
            call_count["count"] += 1
            if call_count["count"] == 1:
                key = self._classify_provider_prompt(payload)
                partial = json.loads(responses[key])
                partial.pop(target_key, None)
                return self._MockHTTPResponse({"choices": [{"message": {"content": json.dumps(partial)}}]})
            if call_count["count"] == 2:
                return self._MockHTTPResponse({"choices": [{"message": {"content": responses["architecture_plan"]}}]})
            key = self._classify_provider_prompt(payload)
            return self._MockHTTPResponse({"choices": [{"message": {"content": responses[key]}}]})

        return _responder

    # Main pipeline test
    def test_end_to_end_build_generates_prototype_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir: # Creates a fake folder which is automatically deleted after the test
            orchestrator = OrchestratorAgent(
                product_idea="Collaborative roadmap planner for product teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            )
            context = orchestrator.run() # Runs the full pipeline and returns the build context.

            self.assertIsNotNone(context.specification)
            self.assertIsNotNone(context.architecture)
            self.assertIsNotNone(context.implementation)
            self.assertIsNotNone(context.testing)
            self.assertIsNotNone(context.evaluation)
            self.assertTrue(context.testing.passed)
            self.assertTrue((context.implementation.prototype_dir / "index.html").exists())
            self.assertTrue(context.build_summary_path.exists())
            self.assertTrue(context.final_report_path.exists())

    def test_orchestrator_tracks_milestone_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Internal support copilot for operations teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            ).run()
            statuses = {milestone.name: milestone.status.value for milestone in context.milestones}
            self.assertEqual(statuses["Discovery & Planning"], "completed")
            self.assertEqual(statuses["Architecture"], "completed")
            self.assertEqual(statuses["Implementation"], "completed")
            self.assertEqual(statuses["Testing"], "completed")
            self.assertEqual(statuses["Debugging"], "pending")
            self.assertEqual(statuses["Evaluation"], "completed")
            self.assertEqual(statuses["Publishing"], "pending")

    def test_invalid_specification_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OrchestratorAgent(
                product_idea="Internal support copilot for operations teams",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=["not-json"]),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            )
            with self.assertRaises(ValueError):
                orchestrator.run()
            self.assertIsNotNone(orchestrator.context.text_log_path)
            self.assertTrue(orchestrator.context.text_log_path.exists())
            self.assertTrue(orchestrator.context.events_log_path.exists())
            self.assertTrue(orchestrator.context.partial_summary_path.exists())
            partial_payload = json.loads(orchestrator.context.partial_summary_path.read_text(encoding="utf-8"))
            self.assertIsNotNone(partial_payload["last_failure"])
            self.assertEqual(partial_payload["last_failure"]["failed_node"], "specification")

    def test_implementation_retries_after_invalid_llm_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = default_responses()
            responses = dict(base)
            responses["implementation"] = [
                '{"notes": ["broken"], "files": {"index.html": "missing rest"}}',
                base["implementation"],
            ]
            orchestrator = OrchestratorAgent(
                product_idea="Planning assistant",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=responses),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            )
            context = orchestrator.run()

            self.assertTrue(context.testing.passed)
            implementation_milestone = next(item for item in context.milestones if item.name == "Implementation")
            self.assertGreaterEqual(implementation_milestone.retries, 1)

    def test_debugging_agent_returns_fix_plan(self) -> None:
        context = BuildContext(
            request=BuildRequest(
                product_idea="Planning assistant",
                output_root=Path(tempfile.gettempdir()),
                llm_provider="openai",
                llm_model="fake-model",
                llm_backend_type="cloud",
            )
        )
        logger = BuildLogger(context)
        agent = DebuggingAgent(
            AgentDependencies(
                context=context,
                logger=logger,
                llm=FakeLLMClient(
                    responses=[
                        debugging_response(
                            failure_groups=["missing_file"],
                            file_changes=["Create README.md with the expected prototype instructions."],
                            priority_order=["README.md"],
                        )
                    ]
                ),
            )
        )

        fix_plan = agent.run(
            iteration=1,
            failure_reports=[FailureReport(source="file-check", summary="Missing required prototype file: README.md", details=["Expected README.md to exist."])],
        )

        self.assertEqual(fix_plan.failure_groups, ["missing_file"])
        self.assertIn("README", " ".join(fix_plan.priority_order + fix_plan.file_changes))

    def test_failed_testing_triggers_debugging_and_records_fix_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = default_responses()
            broken_implementation = (
                '{"notes":["Broken implementation"],'
                '"files":{"index.html":"<!DOCTYPE html><html><body><h1>Collaborative roadmap planner</h1></body></html>",'
                '"styles.css":":root { --accent: #0d6f63; }",'
                '"app.js":"console.log(\\"missing flow\\");",'
                '"README.md":"# Collaborative roadmap planner\\n"}}'
            )
            responses = dict(base)
            responses["implementation"] = [
                broken_implementation,
                base["implementation"],
            ]
            responses["debugging"] = debugging_response(failure_groups=["content_mismatch"])
            context = OrchestratorAgent(
                product_idea="Planning assistant",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=responses),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
            ).run()

            self.assertTrue(context.testing.passed)
            self.assertIsNotNone(context.debugging)
            self.assertEqual(context.debugging.failure_groups, ["content_mismatch"])
            debugging_milestone = next(item for item in context.milestones if item.name == "Debugging")
            self.assertEqual(debugging_milestone.status.value, "completed")
            self.assertGreaterEqual(debugging_milestone.retries, 0)

    def test_schema_validation_failure_triggers_debugging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = default_responses()
            responses = dict(base)
            responses["specification"] = (
                '{"title":"Collaborative roadmap planner","product_summary":"A planning workspace for product teams.",'
                '"requirements":["Deliver a working prototype."],"features":["Feature Breakdown dashboard"],'
                '"user_stories":["As a PM, I want to create a plan quickly."],"api_contracts":["BROKEN CONTRACT"],'
                '"assumptions":["In-memory data is acceptable for v1."],"constraints":["Remain runnable with standard Python tooling."],'
                '"missing_requirements":["Persona detail should be refined later."]}'
            )
            responses["debugging"] = [
                debugging_response(failure_groups=["schema_mismatch"], priority_order=["api_contracts"]),
                debugging_response(failure_groups=["schema_mismatch"], priority_order=["api_contracts"]),
            ]
            context = OrchestratorAgent(
                product_idea="Planning assistant",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=responses),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                max_retries=1,
                memory_path=Path(tmp_dir) / "build_memory.json",
            ).run()

            self.assertIsNotNone(context.debugging)
            self.assertIn("schema_mismatch", context.debugging.failure_groups)
            self.assertFalse(context.testing.passed)

    def test_max_workers_one_preserves_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Single worker test",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                max_workers=1,
                memory_path=Path(tmp_dir) / "build_memory.json",
            ).run()

            self.assertTrue(context.testing.passed)
            self.assertEqual(context.request.max_workers, 1)

    def test_build_memory_persists_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_path = Path(tmp_dir) / "build_memory.json"
            first_context = OrchestratorAgent(
                product_idea="Memory planner one",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=memory_path,
            ).run()

            self.assertTrue(memory_path.exists())
            first_payload = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(len(first_payload["past_builds"]), 1)
            self.assertIn("successful_patterns", first_payload)
            self.assertIsNotNone(first_context.memory)

            second_context = OrchestratorAgent(
                product_idea="Memory planner two",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=memory_path,
            ).run()

            second_payload = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(len(second_payload["past_builds"]), 2)
            self.assertTrue(second_context.memory.past_builds)
            self.assertIn("successful_patterns", second_payload["successful_patterns"])

    def test_memory_is_injected_into_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_llm = FakeLLMClient(responses=default_responses())
            memory_path = Path(tmp_dir) / "build_memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "past_builds": [],
                        "failure_patterns": {
                            "repeated_bugs": [
                                {"pattern": "Missing Feature Breakdown section", "count": 2, "examples": ["index.html omitted Feature Breakdown"]}
                            ]
                        },
                        "successful_patterns": {
                            "successful_patterns": [
                                {"pattern": "Passing testing pipeline with execution, schema, and browser-aware checks.", "count": 1, "examples": []}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            OrchestratorAgent(
                product_idea="Memory-aware planner",
                output_root=Path(tmp_dir),
                llm_client=fake_llm,
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=memory_path,
            ).run()

            prompts = "\n".join(prompt for _, prompt in fake_llm.prompt_history)
            self.assertIn("Common past failures", prompts)
            self.assertIn("Previously successful patterns", prompts)

    def test_end_to_end_build_succeeds_with_openai_compatible_fenced_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            responses = default_responses()
            llm_client = LLMClient(
                provider="openai_compatible",
                local_model="llama3",
                base_url="http://localhost:11434/v1",
            )
            with patch("urllib.request.urlopen", side_effect=self._openai_compatible_responder(responses)):
                context = OrchestratorAgent(
                    product_idea="Collaborative roadmap planner for product teams",
                    output_root=Path(tmp_dir),
                    llm_client=llm_client,
                    node_bin="missing-node",
                    pytest_bin="missing-pytest",
                    memory_path=Path(tmp_dir) / "build_memory.json",
                ).run()

            self.assertTrue(context.testing.passed)
            self.assertIsNotNone(context.evaluation)
            self.assertTrue((context.implementation.prototype_dir / "index.html").exists())

    def test_end_to_end_build_succeeds_with_anthropic_tool_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            responses = default_responses()
            llm_client = LLMClient(
                provider="anthropic",
                model="claude-test",
                api_key="secret",
            )
            with patch("urllib.request.urlopen", side_effect=self._anthropic_tool_responder(responses)):
                context = OrchestratorAgent(
                    product_idea="Collaborative roadmap planner for product teams",
                    output_root=Path(tmp_dir),
                    llm_client=llm_client,
                    node_bin="missing-node",
                    pytest_bin="missing-pytest",
                    memory_path=Path(tmp_dir) / "build_memory.json",
                ).run()

            self.assertTrue(context.testing.passed)
            self.assertIsNotNone(context.evaluation)
            self.assertTrue((context.implementation.prototype_dir / "index.html").exists())

    def test_architecture_missing_key_is_repaired_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            responses = default_responses()
            llm_client = LLMClient(
                provider="openai_compatible",
                local_model="llama3",
                base_url="http://localhost:11434/v1",
            )
            with patch(
                "urllib.request.urlopen",
                side_effect=self._openai_compatible_missing_key_then_repair_responder(responses, "justification"),
            ):
                context = OrchestratorAgent(
                    product_idea="Collaborative roadmap planner for product teams",
                    output_root=Path(tmp_dir),
                    llm_client=llm_client,
                    node_bin="missing-node",
                    pytest_bin="missing-pytest",
                    memory_path=Path(tmp_dir) / "build_memory.json",
                ).run()

            self.assertTrue(context.testing.passed)
            self.assertEqual(context.architecture.justification, ["The build remains easy to test and inspect."])

    def test_publish_failure_does_not_fail_build(self) -> None:
        class FailingPublisher:
            def publish(self, context):  # pragma: no cover - exercised through orchestrator
                raise GitHubPublishError("simulated publish outage")

        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Publishing failure test",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
                publish_to_github=True,
                publisher=FailingPublisher(),
            ).run()

            self.assertTrue(context.testing.passed)
            self.assertIsNotNone(context.github_publish)
            self.assertFalse(context.github_publish.published)
            self.assertIn("simulated publish outage", context.github_publish.failure_reason)
            publishing_status = next(item for item in context.milestones if item.name == "Publishing")
            self.assertEqual(publishing_status.status.value, "failed")

    def test_publish_success_is_reported(self) -> None:
        class SuccessfulPublisher:
            def publish(self, context):
                publish_dir = context.implementation.prototype_dir.parent / "github_publish"
                publish_dir.mkdir(parents=True, exist_ok=True)
                return GitHubPublishResult(
                    enabled=True,
                    published=True,
                    repo_name="demo-prototype",
                    repo_full_name="demo/demo-prototype",
                    repo_url="https://github.com/demo/demo-prototype",
                    branch="main",
                    local_publish_dir=publish_dir,
                    commits=["feat: initial prototype generation", "docs: add final build summary and report"],
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            context = OrchestratorAgent(
                product_idea="Publishing success test",
                output_root=Path(tmp_dir),
                llm_client=FakeLLMClient(responses=default_responses()),
                node_bin="missing-node",
                pytest_bin="missing-pytest",
                memory_path=Path(tmp_dir) / "build_memory.json",
                publish_to_github=True,
                publisher=SuccessfulPublisher(),
            ).run()

            self.assertTrue(context.github_publish.published)
            self.assertEqual(context.github_publish.repo_full_name, "demo/demo-prototype")
            publishing_status = next(item for item in context.milestones if item.name == "Publishing")
            self.assertEqual(publishing_status.status.value, "completed")


if __name__ == "__main__":
    unittest.main()

"""
Mental Model:
Spins up a fake project idea ---> Run the entire pipeline ---> Checks if it actually produced something real.
"""
