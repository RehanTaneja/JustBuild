from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


class LLMError(RuntimeError):
    """Base LLM client error."""


class LLMConfigurationError(LLMError):
    """Raised when no usable backend is configured."""


class LLMModelAccessError(LLMConfigurationError):
    """Raised when a model is invalid or unavailable for the configured provider."""


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
    structured_output_mode: str | None = None


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
                structured_output_mode=self._structured_output_mode(self.provider),
            )
        if self.local_model and self.base_url:
            return LLMBackendInfo(
                provider=self.provider or "openai_compatible",
                model=self.local_model,
                base_url=self.base_url,
                backend_type="local",
                structured_output_mode=self._structured_output_mode(self.provider or "openai_compatible"),
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

        if response_schema is None:
            return self._generate_text_response(
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                base_url=backend.base_url,
            )

        return self._generate_structured_response(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            base_url=backend.base_url,
        )

    def _generate_structured_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        if provider == "openai":
            raw = self._generate_openai_response(
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )
            return self._normalize_or_repair_json(
                raw_text=raw,
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        if provider == "openai_compatible":
            try:
                raw = self._generate_openai_response(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    base_url=base_url,
                )
            except LLMError:
                return self._generate_best_effort_json(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    base_url=base_url,
                )
            try:
                return self._normalize_or_repair_json(
                    raw_text=raw,
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    base_url=base_url,
                )
            except LLMResponseError:
                return self._generate_best_effort_json(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    base_url=base_url,
                )

        if provider == "anthropic":
            payload = self._build_anthropic_tool_payload(model, prompt, system_prompt, response_schema)
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            }
            endpoint = self.base_url or "https://api.anthropic.com/v1/messages"
            return self._extract_anthropic_tool_input(self._post_json(endpoint, payload, headers))

        if provider == "gemini":
            raw = self._generate_gemini_response(prompt, system_prompt, response_schema, model, base_url)
            return self._normalize_or_repair_json(
                raw_text=raw,
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    def _generate_best_effort_json(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        raw = self._generate_text_response(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=self._augment_system_prompt_for_json(system_prompt),
            base_url=base_url,
        )
        return self._normalize_or_repair_json(
            raw_text=raw,
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            base_url=base_url,
        )

    def _generate_text_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        base_url: str | None,
    ) -> str:
        if provider in {"openai", "openai_compatible"}:
            return self._generate_openai_response(
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=None,
                base_url=base_url,
            )

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
            return self._generate_gemini_response(prompt, system_prompt, None, model, base_url)

        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    def _generate_openai_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any] | None,
        base_url: str | None,
    ) -> str:
        payload = self._build_openai_payload(model, prompt, system_prompt, response_schema)
        endpoint = self._openai_endpoint(provider, base_url)
        headers = {"Content-Type": "application/json"}
        if provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self._extract_openai_text(self._post_json(endpoint, payload, headers))

    def _generate_gemini_response(
        self,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any] | None,
        model: str,
        base_url: str | None,
    ) -> str:
        payload = self._build_gemini_payload(prompt, system_prompt, response_schema)
        endpoint = self._gemini_endpoint(model, base_url, self.api_key)
        headers = {"Content-Type": "application/json"}
        return self._extract_gemini_text(self._post_json(endpoint, payload, headers))

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and self._looks_like_model_access_error(detail):
                raise LLMModelAccessError(
                    f"Configured model is unavailable or inaccessible for provider request: {detail}"
                ) from exc
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

    def _build_anthropic_tool_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._build_anthropic_payload(model, prompt, system_prompt)
        payload["tools"] = [
            {
                "name": "justbuild_response",
                "description": "Return the final response as a JSON object that matches the requested schema.",
                "input_schema": response_schema,
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": "justbuild_response"}
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

    def _extract_anthropic_tool_input(self, payload: dict[str, Any]) -> str:
        try:
            parts = payload["content"]
        except KeyError as exc:
            raise LLMResponseError("Anthropic response did not include content") from exc
        for part in parts:
            if part.get("type") == "tool_use" and part.get("name") == "justbuild_response":
                input_payload = part.get("input")
                if isinstance(input_payload, dict):
                    return json.dumps(input_payload)
                raise LLMResponseError("Anthropic tool response did not include a JSON object input")
        raise LLMResponseError("Anthropic response did not include the forced structured-output tool result")

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

    def _structured_output_mode(self, provider: str) -> str:
        if provider == "anthropic":
            return "tool_schema"
        if provider in {"openai", "gemini"}:
            return "strict_schema"
        if provider == "openai_compatible":
            return "best_effort_schema"
        return "unknown"

    def _augment_system_prompt_for_json(self, system_prompt: str | None) -> str:
        extra = (
            " Return exactly one valid JSON object and nothing else. "
            "Do not use markdown fences. Do not include commentary before or after the JSON."
        )
        if system_prompt:
            return system_prompt + extra
        return extra.strip()

    def _normalize_or_repair_json(
        self,
        raw_text: str,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        try:
            return self._normalize_json_text(raw_text)
        except LLMResponseError:
            repair_prompt = (
                "You previously returned output that was not a single raw JSON object.\n"
                "Rewrite it as exactly one valid JSON object that matches the requested schema.\n"
                "Do not include markdown fences or explanatory text.\n"
                f"Original prompt:\n{prompt}\n\n"
                f"Invalid output:\n{raw_text}"
            )
            repaired = self._generate_text_response(
                provider=provider,
                model=model,
                prompt=repair_prompt,
                system_prompt=self._augment_system_prompt_for_json(system_prompt),
                base_url=base_url,
            )
            return self._normalize_json_text(repaired)

    def _normalize_json_text(self, raw_text: str) -> str:
        stripped = raw_text.strip()
        if not stripped:
            raise LLMResponseError("Provider response was empty")

        direct = self._parse_json_object(stripped)
        if direct is not None:
            return json.dumps(direct)

        fenced = self._extract_fenced_json_candidates(stripped)
        if len(fenced) == 1:
            return json.dumps(fenced[0])
        if len(fenced) > 1:
            raise LLMResponseError("Provider response contained multiple fenced JSON objects")

        embedded = self._extract_embedded_json_candidates(stripped)
        if len(embedded) == 1:
            return json.dumps(embedded[0])
        if len(embedded) > 1:
            raise LLMResponseError("Provider response contained multiple JSON objects")

        raise LLMResponseError("Provider response could not be normalized into a single JSON object")

    def _extract_fenced_json_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
        for match in pattern.finditer(text):
            parsed = self._parse_json_object(match.group(1).strip())
            if parsed is not None:
                candidates.append(parsed)
        return candidates

    def _extract_embedded_json_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_ranges: set[tuple[int, int]] = set()
        for start in range(len(text)):
            if text[start] != "{":
                continue
            end = self._find_balanced_json_end(text, start)
            if end is None:
                continue
            span = (start, end)
            if span in seen_ranges:
                continue
            seen_ranges.add(span)
            parsed = self._parse_json_object(text[start : end + 1])
            if parsed is not None:
                candidates.append(parsed)
        return candidates

    def _find_balanced_json_end(self, text: str, start: int) -> int | None:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def _parse_json_object(self, candidate: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def _looks_like_model_access_error(self, detail: str) -> bool:
        lowered = detail.lower()
        return "model" in lowered and ("not_found" in lowered or "not found" in lowered or "does not exist" in lowered)
