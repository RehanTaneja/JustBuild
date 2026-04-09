from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


class LLMError(RuntimeError):
    """Base LLM client error."""


class LLMConfigurationError(LLMError):
    """Raised when no usable backend is configured."""


class LLMTransportError(LLMError):
    """Raised when a provider request fails."""


class LLMResponseError(LLMError):
    """Raised when a provider response is malformed."""


@dataclass(slots=True)
class LLMBackendInfo:
    provider: str | None
    model: str | None
    base_url: str | None
    backend_type: str


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        local_model: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 60,
    ) -> None:
        self.api_key = api_key
        self.local_model = local_model
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    @property
    def backend_info(self) -> LLMBackendInfo:
        if self.api_key and self.provider:
            return LLMBackendInfo(
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                backend_type="cloud",
            )
        if self.local_model and self.base_url:
            return LLMBackendInfo(
                provider=self.provider or "openai_compatible",
                model=self.local_model,
                base_url=self.base_url,
                backend_type="local",
            )
        raise LLMConfigurationError("No LLM backend configured")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        backend = self.backend_info
        provider = backend.provider or "openai_compatible"
        model = backend.model
        if model is None:
            raise LLMConfigurationError("No model configured for the active LLM backend")

        if provider in {"openai", "openai_compatible"}:
            payload = self._build_openai_payload(model, prompt, system_prompt, response_schema)
            endpoint = self._openai_endpoint(provider, backend.base_url)
            headers = {"Content-Type": "application/json"}
            if provider == "openai":
                headers["Authorization"] = f"Bearer {self.api_key}"
            return self._extract_openai_text(self._post_json(endpoint, payload, headers))

        if provider == "anthropic":
            payload = self._build_anthropic_payload(model, prompt, system_prompt)
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            }
            endpoint = self.base_url or "https://api.anthropic.com/v1/messages"
            return self._extract_anthropic_text(self._post_json(endpoint, payload, headers))

        if provider == "gemini":
            payload = self._build_gemini_payload(prompt, system_prompt, response_schema)
            endpoint = self._gemini_endpoint(model, self.base_url, self.api_key)
            headers = {"Content-Type": "application/json"}
            return self._extract_gemini_text(self._post_json(endpoint, payload, headers))

        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMTransportError(f"Provider HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMTransportError(f"Provider network error: {exc.reason}") from exc

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Provider response was not valid JSON") from exc

    def _build_openai_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "justbuild_response",
                    "schema": response_schema,
                },
            }
        return payload

    def _build_anthropic_payload(self, model: str, prompt: str, system_prompt: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 4000,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _build_gemini_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        text_parts = []
        if system_prompt:
            text_parts.append(system_prompt)
        text_parts.append(prompt)
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": "\n\n".join(text_parts)}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        if response_schema is not None:
            payload["generationConfig"]["responseSchema"] = response_schema
        return payload

    def _extract_openai_text(self, payload: dict[str, Any]) -> str:
        try:
            choice = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("OpenAI-compatible response did not include choices[0].message") from exc
        content = choice.get("content")
        if isinstance(content, str) and content.strip():
            return content
        raise LLMResponseError("OpenAI-compatible response did not include message content")

    def _extract_anthropic_text(self, payload: dict[str, Any]) -> str:
        try:
            parts = payload["content"]
        except KeyError as exc:
            raise LLMResponseError("Anthropic response did not include content") from exc
        for part in parts:
            if part.get("type") == "text" and part.get("text", "").strip():
                return part["text"]
        raise LLMResponseError("Anthropic response did not include a text block")

    def _extract_gemini_text(self, payload: dict[str, Any]) -> str:
        try:
            candidates = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("Gemini response did not include candidates[0].content.parts") from exc
        for part in candidates:
            if isinstance(part.get("text"), str) and part["text"].strip():
                return part["text"]
        raise LLMResponseError("Gemini response did not include text output")

    def _openai_endpoint(self, provider: str, base_url: str | None) -> str:
        if provider == "openai":
            return (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        if base_url is None:
            raise LLMConfigurationError("OpenAI-compatible local endpoints require base_url")
        return base_url.rstrip("/") + "/chat/completions"

    def _gemini_endpoint(self, model: str, base_url: str | None, api_key: str | None) -> str:
        if not api_key:
            raise LLMConfigurationError("Gemini requires an API key")
        root = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        query = parse.urlencode({"key": api_key})
        return f"{root}/models/{model}:generateContent?{query}"
