from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from justbuild.llm import LLMClient, LLMConfigurationError, LLMResponseError


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


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
