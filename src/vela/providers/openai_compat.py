"""OpenAI-compatible provider.

Covers OpenAI itself plus the many services that speak the same wire format
(DeepSeek, Moonshot, Groq, Together, vLLM, Ollama, LM Studio, and most gateway
routers). One implementation therefore unlocks a large share of available
backends.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from vela.core.contracts import ModelResponse, Usage
from vela.providers.base import (
    GenerationConfig,
    Provider,
    StreamEmitter,
    ToolSpec,
    classify_error,
    parse_openai_response,
    sanitize_messages,
)


class OpenAICompatProvider(Provider):
    """Talks to any ``/chat/completions`` endpoint."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str,
        stream: bool,
    ) -> dict[str, Any]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend(sanitize_messages(messages))

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": wire,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [spec.to_openai() for spec in tools]
            payload["tool_choice"] = "auto"
        return payload

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        *,
        system: str = "",
        stream: bool = False,
        on_delta: Any = None,
    ) -> ModelResponse:
        payload = self._build_payload(messages, tools, system, stream)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            if stream:
                return await self._complete_streaming(payload, headers, on_delta)
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            # Transport failures never reach the model, so classify as transient
            # and let the runner retry instead of surfacing a raw stack trace.
            return ModelResponse(
                content="", finish_reason="error", error=f"transport error: {exc}"
            )

        if response.status_code >= 400:
            return ModelResponse(
                content="",
                finish_reason="error",
                error=classify_error(response.status_code, response.text).message
                if hasattr(classify_error(response.status_code, response.text), "message")
                else response.text,
                usage=Usage(estimated=True),
            )

        try:
            return parse_openai_response(response.json())
        except json.JSONDecodeError as exc:
            return ModelResponse(
                content="", finish_reason="error", error=f"invalid JSON response: {exc}"
            )

    async def _complete_streaming(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_delta: Any = None,
    ) -> ModelResponse:
        """Consume SSE and reassemble both text and tool-call fragments.

        Tool-call arguments arrive as a sequence of partial JSON fragments keyed
        by index. They must be concatenated in order before parsing; parsing each
        fragment independently would corrupt every argument object.
        """
        fragments: dict[int, dict[str, Any]] = {}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    error = classify_error(response.status_code, body)
                    return ModelResponse(
                        content="", finish_reason="error", error=str(error)
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if isinstance(delta.get("content"), str):
                            content_parts.append(delta["content"])
                            # Forward each fragment as it parses so the caller
                            # can render progressively rather than waiting for
                            # the whole reply to be buffered.
                            if on_delta is not None and delta["content"]:
                                await on_delta(delta["content"])
                        if isinstance(delta.get("reasoning_content"), str):
                            reasoning_parts.append(delta["reasoning_content"])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        for item in delta.get("tool_calls") or []:
                            index = item.get("index", 0)
                            slot = fragments.setdefault(
                                index, {"id": "", "name": "", "arguments": []}
                            )
                            if item.get("id"):
                                slot["id"] = item["id"]
                            function = item.get("function") or {}
                            if isinstance(function.get("name"), str):
                                slot["name"] = function["name"]
                            if isinstance(function.get("arguments"), str):
                                slot["arguments"].append(function["arguments"])

                    usage_chunk = chunk.get("usage")
                    if isinstance(usage_chunk, dict):
                        usage = extract_stream_usage(usage_chunk)
        except httpx.HTTPError as exc:
            return ModelResponse(
                content="", finish_reason="error", error=f"stream transport error: {exc}"
            )

        return ModelResponse(
            content="".join(content_parts),
            tool_calls=tuple(_reassemble(fragments)),
            finish_reason=finish_reason if finish_reason in {
                "stop", "length", "tool_calls", "content_filter", "error"
            } else "stop",
            usage=usage if "usage" in locals() else Usage(estimated=True),
            reasoning="".join(reasoning_parts),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _reassemble(fragments: dict[int, dict[str, Any]]) -> list:
    """Rebuild complete tool calls from streamed fragments."""
    from vela.providers.base import ToolCall, parse_tool_call_arguments

    calls = []
    for index in sorted(fragments):
        slot = fragments[index]
        name = slot["name"]
        if not isinstance(name, str) or not name.strip():
            # The runner handles the "all calls malformed" case; keep this call
            # out of the result so it cannot be dispatched.
            continue
        calls.append(
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=name,
                arguments=parse_tool_call_arguments("".join(slot["arguments"])),
            )
        )
    return calls


def extract_stream_usage(chunk: dict[str, Any]) -> Usage:
    """Read usage from a streaming chunk when the backend provides it."""
    from vela.providers.base import extract_usage

    return extract_usage({"usage": chunk})
