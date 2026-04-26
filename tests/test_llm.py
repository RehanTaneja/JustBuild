from __future__ import annotations

import io
import json
import unittest
from urllib import error
from unittest.mock import patch

from justbuild.llm import LLMClient, LLMConfigurationError, LLMModelAccessError, LLMResponseError, LLMTimeoutError


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_http_error(code: int, payload: dict) -> error.HTTPError:
    body = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return error.HTTPError("https://example.test", code, "error", hdrs=None, fp=body)


class LLMClientTests(unittest.TestCase):
    def test_missing_backend_config_raises(self) -> None:
        client = LLMClient()
        with self.assertRaises(LLMConfigurationError):
            client.generate("hello")

    def test_openai_compatible_local_mode_uses_base_url_and_local_model(self) -> None:
        client = LLMClient(provider="openai_compatible", local_model="llama3", base_url="http://localhost:11434/v1")
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})) as mocked:
            response = client.generate("hello")

        self.assertEqual(response, '{"ok": true}')
        request_obj = mocked.call_args.args[0]
        self.assertEqual(request_obj.full_url, "http://localhost:11434/v1/chat/completions")

    def test_invalid_provider_response_raises(self) -> None:
        client = LLMClient(provider="openai", model="gpt-test", api_key="secret")
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse({"unexpected": []})):
            with self.assertRaises(LLMResponseError):
                client.generate("hello")

    def test_anthropic_structured_output_uses_forced_tool_schema(self) -> None:
        client = LLMClient(provider="anthropic", model="claude-test", api_key="secret")
        response_payload = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "justbuild_response",
                    "input": {"ok": True},
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(response_payload)) as mocked:
            response = client.generate("hello", response_schema={"type": "object", "required": ["ok"]})

        self.assertEqual(json.loads(response), {"ok": True})
        request_obj = mocked.call_args.args[0]
        request_payload = json.loads(request_obj.data.decode("utf-8"))
        self.assertEqual(request_payload["tool_choice"], {"type": "tool", "name": "justbuild_response"})
        self.assertEqual(request_payload["tools"][0]["input_schema"], {"type": "object", "required": ["ok"]})

    def test_openai_compatible_falls_back_when_schema_mode_is_unsupported(self) -> None:
        client = LLMClient(provider="openai_compatible", local_model="llama3", base_url="http://localhost:11434/v1")
        unsupported = build_http_error(
            400,
            {"error": {"message": "response_format is not supported by this server"}},
        )
        fallback = FakeHTTPResponse({"choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]})
        with patch("urllib.request.urlopen", side_effect=[unsupported, fallback]) as mocked:
            response = client.generate("hello", response_schema={"type": "object", "required": ["ok"]})

        self.assertEqual(json.loads(response), {"ok": True})
        first_payload = json.loads(mocked.call_args_list[0].args[0].data.decode("utf-8"))
        second_payload = json.loads(mocked.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertIn("response_format", first_payload)
        self.assertNotIn("response_format", second_payload)

    def test_prose_wrapped_json_is_normalized(self) -> None:
        client = LLMClient(provider="openai", model="gpt-test", api_key="secret")
        with patch(
            "urllib.request.urlopen",
            return_value=FakeHTTPResponse({"choices": [{"message": {"content": "Here is the result:\n{\"ok\": true}\nDone."}}]}),
        ):
            response = client.generate("hello", response_schema={"type": "object", "required": ["ok"]})

        self.assertEqual(json.loads(response), {"ok": True})

    def test_non_json_output_fails_after_one_repair_attempt(self) -> None:
        client = LLMClient(provider="openai_compatible", local_model="llama3", base_url="http://localhost:11434/v1")
        unsupported = build_http_error(
            400,
            {"error": {"message": "response_format is not supported by this server"}},
        )
        invalid = FakeHTTPResponse({"choices": [{"message": {"content": "definitely not json"}}]})
        with patch("urllib.request.urlopen", side_effect=[unsupported, invalid, invalid]) as mocked:
            with self.assertRaises(LLMResponseError):
                client.generate("hello", response_schema={"type": "object", "required": ["ok"]})

        self.assertEqual(mocked.call_count, 3)

    def test_model_not_found_is_reported_as_model_access_error(self) -> None:
        client = LLMClient(provider="anthropic", model="missing-model", api_key="secret")
        not_found = build_http_error(
            404,
            {"type": "error", "error": {"type": "not_found_error", "message": "model: missing-model"}},
        )
        with patch("urllib.request.urlopen", side_effect=not_found):
            with self.assertRaises(LLMModelAccessError):
                client.generate("hello", response_schema={"type": "object", "required": ["ok"]})

    def test_missing_required_key_triggers_schema_completion_repair(self) -> None:
        client = LLMClient(provider="openai_compatible", local_model="llama3", base_url="http://localhost:11434/v1")
        first = FakeHTTPResponse({"choices": [{"message": {"content": '{"summary":"done"}'}}]})
        repaired = FakeHTTPResponse({"choices": [{"message": {"content": '{"summary":"done","justification":[]}'}}]})
        with patch("urllib.request.urlopen", side_effect=[first, repaired]):
            response = client.generate(
                "hello",
                response_schema={"type": "object", "required": ["summary", "justification"]},
            )

        self.assertEqual(json.loads(response), {"summary": "done", "justification": []})

    def test_anthropic_missing_required_key_uses_structured_schema_completion_repair(self) -> None:
        client = LLMClient(provider="anthropic", model="claude-test", api_key="secret")
        first = FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "justbuild_response",
                        "input": {"notes": ["generated"]},
                    }
                ]
            }
        )
        repaired = FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "justbuild_response",
                        "input": {
                            "notes": ["generated"],
                            "files": {
                                "index.html": "<html></html>",
                                "styles.css": ":root {}",
                                "app.js": "const msg = 'Generated Response';",
                                "README.md": "# Prototype",
                            },
                        },
                    }
                ]
            }
        )
        schema = {"type": "object", "required": ["notes", "files"]}
        with patch("urllib.request.urlopen", side_effect=[first, repaired]) as mocked:
            response = client.generate("hello", response_schema=schema)

        parsed = json.loads(response)
        self.assertIn("files", parsed)
        second_payload = json.loads(mocked.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(second_payload["tool_choice"], {"type": "tool", "name": "justbuild_response"})
        self.assertEqual(second_payload["tools"][0]["input_schema"], schema)

    def test_repeated_incomplete_json_fails_after_schema_completion_repair(self) -> None:
        client = LLMClient(provider="openai_compatible", local_model="llama3", base_url="http://localhost:11434/v1")
        incomplete = FakeHTTPResponse({"choices": [{"message": {"content": '{"summary":"done"}'}}]})
        with patch("urllib.request.urlopen", side_effect=[incomplete, incomplete]):
            with self.assertRaises(LLMResponseError):
                client.generate(
                    "hello",
                    response_schema={"type": "object", "required": ["summary", "justification"]},
                )

    def test_timeout_is_reported_distinctly(self) -> None:
        client = LLMClient(provider="anthropic", model="claude-test", api_key="secret", timeout_s=5)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(LLMTimeoutError):
                client.generate("hello", response_schema={"type": "object", "required": ["ok"]})
