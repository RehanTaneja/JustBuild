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


class LLMTimeoutError(LLMTransportError):
    """Raised when a provider request exceeds the configured timeout."""


@dataclass(slots=True)
class LLMBackendInfo:
    provider: str | None
    model: str | None
    base_url: str | None
    backend_type: str
    backend_family: str | None = None
    structured_output_mode: str | None = None
    capabilities_probed: bool = False
    capability_source: str | None = None
    capability_downgrade: str | None = None


@dataclass(slots=True)
class _BackendCapabilities:
    backend_family: str
    supports_chat_completions: bool = True
    supports_json_schema: bool | None = None
    supports_tool_calling: bool | None = None
    structured_strategy: str = "best_effort_schema"
    probed: bool = False
    capability_source: str = "inferred"
    last_downgrade: str | None = None


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        local_model: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 60,
        event_logger=None,
    ) -> None:
        self.api_key = api_key
        self.local_model = local_model
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.event_logger = event_logger
        self._capability_cache: dict[str, _BackendCapabilities] = {}

    @property
    def backend_info(self) -> LLMBackendInfo:
        if self.api_key and self.provider:
            return LLMBackendInfo(
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                backend_type="cloud",
                backend_family=self.provider,
                structured_output_mode=self._structured_output_mode(self.provider),
                capabilities_probed=False,
                capability_source="static",
            )
        if self.local_model and self.base_url:
            capabilities = self._get_openai_compatible_capabilities(self.base_url, self.local_model)
            return LLMBackendInfo(
                provider=self.provider or "openai_compatible",
                model=self.local_model,
                base_url=self.base_url,
                backend_type="local",
                backend_family=capabilities.backend_family,
                structured_output_mode=capabilities.structured_strategy,
                capabilities_probed=capabilities.probed,
                capability_source=capabilities.capability_source,
                capability_downgrade=capabilities.last_downgrade,
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
            self._emit_event("llm_request", "Starting structured provider request", {"provider": provider, "model": model, "structured_output_mode": self._structured_output_mode(provider), "timeout_s": self.timeout_s})
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
            return self._generate_openai_compatible_structured_response(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        if provider == "anthropic":
            self._emit_event("llm_request", "Starting structured provider request", {"provider": provider, "model": model, "structured_output_mode": self._structured_output_mode(provider), "timeout_s": self.timeout_s})
            payload = self._build_anthropic_tool_payload(model, prompt, system_prompt, response_schema)
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            }
            endpoint = self.base_url or "https://api.anthropic.com/v1/messages"
            raw = self._extract_anthropic_tool_input(self._post_json(endpoint, payload, headers))
            return self._normalize_or_repair_json(
                raw_text=raw,
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        if provider == "gemini":
            self._emit_event("llm_request", "Starting structured provider request", {"provider": provider, "model": model, "structured_output_mode": self._structured_output_mode(provider), "timeout_s": self.timeout_s})
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
        self._emit_event("llm_fallback", "Using best-effort structured output mode", {"provider": provider, "model": model})
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

    def _generate_openai_tool_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        payload = self._build_openai_tool_payload(model, prompt, system_prompt, response_schema)
        endpoint = self._openai_endpoint(provider, base_url)
        headers = {"Content-Type": "application/json"}
        if provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self._extract_openai_tool_input(self._post_json(endpoint, payload, headers))

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
            self._emit_event("llm_http_error", "Provider returned an HTTP error", {"status_code": exc.code, "detail": detail[:500]})
            raise LLMTransportError(f"Provider HTTP error {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            self._emit_event("llm_timeout", f"Provider timeout after {self.timeout_s}s", {"timeout_s": self.timeout_s})
            raise LLMTimeoutError(
                f"Provider request timed out after {self.timeout_s}s. Increase the LLM timeout if the model is slow."
            ) from exc
        except error.URLError as exc:
            self._emit_event("llm_transport_error", "Provider network error", {"reason": str(exc.reason)})
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

    def _build_openai_tool_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._build_openai_payload(model, prompt, system_prompt, None)
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "justbuild_response",
                    "description": "Return the final response as a JSON object that matches the requested schema.",
                    "parameters": response_schema,
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "justbuild_response"}}
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

    def _extract_openai_tool_input(self, payload: dict[str, Any]) -> str:
        try:
            choice = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("OpenAI-compatible tool response did not include choices[0].message") from exc
        tool_calls = choice.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str) and arguments.strip():
                return arguments
        content = choice.get("content")
        if isinstance(content, str) and content.strip():
            return content
        raise LLMResponseError("OpenAI-compatible tool response did not include tool call arguments")

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
            if self.base_url and self.local_model:
                return self._get_openai_compatible_capabilities(self.base_url, self.local_model).structured_strategy
            return "best_effort_schema"
        return "unknown"

    def _generate_openai_compatible_structured_response(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        if base_url is None:
            raise LLMConfigurationError("OpenAI-compatible local endpoints require base_url")
        raw = self._generate_openai_compatible_structured_raw_response(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            base_url=base_url,
        )
        return self._normalize_or_repair_json(
            raw_text=raw,
            provider="openai_compatible",
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            base_url=base_url,
        )

    def _generate_openai_compatible_structured_raw_response(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        if base_url is None:
            raise LLMConfigurationError("OpenAI-compatible local endpoints require base_url")
        capabilities = self._get_openai_compatible_capabilities(base_url, model)
        strategies = self._strategy_candidates(capabilities)
        last_error: LLMError | None = None
        for index, strategy in enumerate(strategies):
            capabilities.structured_strategy = strategy
            self._emit_event(
                "llm_request",
                "Starting structured provider request",
                {
                    "provider": "openai_compatible",
                    "model": model,
                    "structured_output_mode": strategy,
                    "timeout_s": self.timeout_s,
                    "backend_family": capabilities.backend_family,
                    "capabilities_probed": capabilities.probed,
                    "capability_source": capabilities.capability_source,
                },
            )
            try:
                if strategy == "tool_schema":
                    return self._generate_openai_tool_response(
                        provider="openai_compatible",
                        model=model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                        base_url=base_url,
                    )
                elif strategy == "strict_schema":
                    return self._generate_openai_response(
                        provider="openai_compatible",
                        model=model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                        base_url=base_url,
                    )
                else:
                    return self._generate_text_response(
                        provider="openai_compatible",
                        model=model,
                        prompt=prompt,
                        system_prompt=self._augment_system_prompt_for_json(system_prompt),
                        base_url=base_url,
                    )
            except LLMError as exc:
                last_error = exc
                if strategy != "best_effort_schema" and self._looks_like_unsupported_feature_error(str(exc)):
                    next_strategy = strategies[index + 1] if index + 1 < len(strategies) else "best_effort_schema"
                    self._downgrade_openai_compatible_capability(capabilities, strategy, next_strategy)
                    continue
                if strategy != "best_effort_schema" and index + 1 < len(strategies):
                    raise
                raise

        if last_error is not None:
            raise last_error
        raise LLMResponseError("No structured output strategy succeeded")

    def _get_openai_compatible_capabilities(self, base_url: str, model: str) -> _BackendCapabilities:
        cache_key = self._capability_cache_key(base_url, model)
        cached = self._capability_cache.get(cache_key)
        if cached is not None:
            return cached
        capabilities = self._infer_openai_compatible_capabilities(base_url)
        self._probe_backend_family(base_url, capabilities)
        self._capability_cache[cache_key] = capabilities
        return capabilities

    def _capability_cache_key(self, base_url: str, model: str) -> str:
        return f"{base_url.rstrip('/')}\n{model}"

    def _infer_openai_compatible_capabilities(self, base_url: str) -> _BackendCapabilities:
        parsed = parse.urlparse(base_url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
        path = parsed.path.rstrip("/").lower()
        is_ollama_like = (
            host in {"localhost", "127.0.0.1", "::1"} and port == 11434
        ) or "ollama" in host or path.endswith("/ollama") or "/ollama/" in path
        backend_family = "ollama" if is_ollama_like else "openai_compatible"
        return _BackendCapabilities(
            backend_family=backend_family,
            supports_chat_completions=True,
            supports_json_schema=None,
            supports_tool_calling=None,
            structured_strategy="tool_schema" if is_ollama_like else "strict_schema",
            probed=False,
            capability_source="inferred",
        )

    def _probe_backend_family(self, base_url: str, capabilities: _BackendCapabilities) -> None:
        if capabilities.probed:
            return
        if capabilities.backend_family == "ollama":
            native_root = self._ollama_native_root(base_url)
            for suffix in ("/api/version", "/api/tags"):
                if self._get_json(f"{native_root}{suffix}") is not None:
                    capabilities.probed = True
                    capabilities.capability_source = "probed"
                    return
            capabilities.backend_family = "openai_compatible"
            capabilities.structured_strategy = "strict_schema"
        if self._get_json(self._openai_models_endpoint(base_url)) is not None:
            capabilities.probed = True
            capabilities.capability_source = "probed"
            return
        capabilities.probed = True
        capabilities.capability_source = "inferred"

    def _strategy_candidates(self, capabilities: _BackendCapabilities) -> list[str]:
        candidates: list[str] = []
        if capabilities.supports_tool_calling is not False:
            candidates.append("tool_schema")
        if capabilities.supports_json_schema is not False:
            candidates.append("strict_schema")
        candidates.append("best_effort_schema")
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        if capabilities.structured_strategy in deduped:
            deduped.remove(capabilities.structured_strategy)
            deduped.insert(0, capabilities.structured_strategy)
        return deduped

    def _downgrade_openai_compatible_capability(
        self,
        capabilities: _BackendCapabilities,
        failed_strategy: str,
        next_strategy: str,
    ) -> None:
        if failed_strategy == "tool_schema":
            capabilities.supports_tool_calling = False
        elif failed_strategy == "strict_schema":
            capabilities.supports_json_schema = False
        capabilities.structured_strategy = next_strategy
        capabilities.last_downgrade = f"{failed_strategy}->{next_strategy}"
        self._emit_event(
            "llm_capability_downgrade",
            "Downgrading structured-output strategy after unsupported provider feature",
            {
                "provider": "openai_compatible",
                "structured_output_mode": next_strategy,
                "backend_family": capabilities.backend_family,
                "capabilities_probed": capabilities.probed,
                "capability_source": capabilities.capability_source,
                "capability_downgrade": capabilities.last_downgrade,
            },
        )

    def _ollama_native_root(self, base_url: str) -> str:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            return root[:-3]
        return root

    def _openai_models_endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/models"

    def _get_json(self, url: str) -> dict[str, Any] | None:
        req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                text = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

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
            normalized = self._normalize_json_text(raw_text)
        except LLMResponseError:
            self._emit_event("schema_repair", "Repairing malformed structured output", {"provider": provider, "model": model})
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
            normalized = self._normalize_json_text(repaired)
        return self._complete_or_raise_missing_keys(
            normalized_json=normalized,
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            base_url=base_url,
        )

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

    def _complete_or_raise_missing_keys(
        self,
        normalized_json: str,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        payload = self._parse_json_object(normalized_json)
        if payload is None:
            raise LLMResponseError("Normalized output was not a JSON object")

        required_keys = response_schema.get("required", [])
        missing_keys = [key for key in required_keys if key not in payload]
        if not missing_keys:
            return normalized_json

        self._emit_event(
            "schema_completion_repair",
            f"Repairing missing required keys: {', '.join(missing_keys)}",
            {"provider": provider, "model": model, "missing_keys": missing_keys, "present_keys": sorted(payload.keys())},
        )
        repair_prompt = (
            "You previously returned a JSON object that is missing required keys.\n"
            "Return exactly one valid JSON object.\n"
            "Preserve all existing keys and values.\n"
            "Add every missing required key.\n"
            "For missing array-like fields, return an empty array if uncertain.\n"
            "For missing scalar fields, return a short placeholder string only if you cannot infer a value.\n"
            "Never remove existing keys. Never rename keys. Do not include markdown or commentary.\n"
            f"Original prompt:\n{prompt}\n\n"
            f"Required keys: {json.dumps(required_keys)}\n"
            f"Missing keys: {json.dumps(missing_keys)}\n"
            f"Current JSON:\n{normalized_json}"
        )
        repaired = self._generate_schema_completion_response(
            provider=provider,
            model=model,
            prompt=repair_prompt,
            system_prompt=self._augment_system_prompt_for_json(system_prompt),
            response_schema=response_schema,
            base_url=base_url,
        )
        repaired_normalized = self._normalize_json_text(repaired)
        repaired_payload = self._parse_json_object(repaired_normalized)
        if repaired_payload is None:
            raise LLMResponseError("Schema completion repair did not return a JSON object")

        remaining_missing = [key for key in required_keys if key not in repaired_payload]
        if remaining_missing:
            present = ", ".join(sorted(repaired_payload.keys())) or "(none)"
            self._emit_event(
                "schema_completion_repair",
                f"Schema completion repair still missing keys: {', '.join(remaining_missing)}",
                {"provider": provider, "model": model, "missing_keys": remaining_missing, "present_keys": sorted(repaired_payload.keys())},
            )
            raise LLMResponseError(
                f"Incomplete JSON after schema completion repair. Missing keys: {', '.join(remaining_missing)}. Present keys: {present}"
            )
        return repaired_normalized

    def _generate_schema_completion_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str | None,
        response_schema: dict[str, Any],
        base_url: str | None,
    ) -> str:
        self._emit_event(
            "schema_completion_request",
            "Requesting provider-assisted schema completion repair",
            {
                "provider": provider,
                "model": model,
                "structured_output_mode": self._structured_output_mode(provider),
                "backend_family": self.backend_info.backend_family,
                "capabilities_probed": self.backend_info.capabilities_probed,
                "capability_source": self.backend_info.capability_source,
                "capability_downgrade": self.backend_info.capability_downgrade,
            },
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

        if provider == "openai":
            return self._generate_openai_response(
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        if provider == "gemini":
            return self._generate_gemini_response(prompt, system_prompt, response_schema, model, base_url)

        if provider == "openai_compatible":
            return self._generate_openai_compatible_structured_raw_response(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                base_url=base_url,
            )

        return self._generate_text_response(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            base_url=base_url,
        )

    def _emit_event(self, category: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        if self.event_logger is None:
            return
        self.event_logger(category=category, message=message, metadata=metadata or {})

    def _looks_like_model_access_error(self, detail: str) -> bool:
        lowered = detail.lower()
        return "model" in lowered and ("not_found" in lowered or "not found" in lowered or "does not exist" in lowered)

    def _looks_like_unsupported_feature_error(self, detail: str) -> bool:
        lowered = detail.lower()
        unsupported_markers = [
            "unsupported",
            "not supported",
            "unknown field",
            "unknown parameter",
            "invalid parameter",
            "tool_choice",
            "\"tools\"",
            "response_format",
            "json_schema",
            "response format",
        ]
        return any(marker in lowered for marker in unsupported_markers)
