"""Anthropic Messages API provider.

Implemented against the same :class:`~vela.providers.base.Provider` contract as
the OpenAI-compatible backend, which is the real test of whether that
abstraction holds up. Three structural differences have to be absorbed here so
that nothing above this layer changes:

1. **The system prompt is not a message.** OpenAI carries it as
   ``{"role": "system"}``; Anthropic takes a top-level ``system`` field.
2. **Tool results are not messages.** OpenAI has a dedicated ``role="tool"``
   per result; Anthropic puts ``tool_result`` blocks *inside* a user message,
   and that message must directly follow the assistant turn that requested them.
   Mis-ordering it returns a 400, so the conversion groups results by the call
   they answer.
3. **Tool schemas use ``input_schema``**, and ``stop_reason`` uses
   ``tool_use`` rather than ``tool_calls``.

Everything above this file keeps speaking the OpenAI-shaped vocabulary that
``core.contracts`` defines. That boundary is the point of the exercise.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.providers.anthropic_cache import apply_cache_markers
from vela.providers.caching import CachePolicy, CacheStats
from vela.providers.base import (
    GenerationConfig,
    Provider,
    StreamEmitter,
    ToolSpec,
    classify_error,
    parse_tool_call_arguments,
    sanitize_messages,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

#: Anthropic requires max_tokens, and rejects requests that omit it. Keep the
#: default comfortably below the model ceiling so long answers still fit.
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(Provider):
    """Talks to the Anthropic Messages API."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        cache_policy: CachePolicy | None = None,
        expected_turns: int = 1,
    ) -> None:
        super().__init__(config)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        #: Off by default: caching charges a write premium, so it only pays off
        #: once a prefix is reused. See vela.providers.caching for the model.
        self.cache_policy = cache_policy or CachePolicy()
        self.expected_turns = expected_turns
        #: Realised cache behaviour, so the policy can be judged against facts.
        self.cache_stats = CacheStats()

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def to_anthropic_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert the internal transcript into Anthropic wire format.

        The subtle part is tool results. Internally each result is its own
        ``role="tool"`` message, but Anthropic requires every result belonging
        to one assistant turn to be merged into a *single* user message placed
        immediately after that turn. Iterating with an index lets us look ahead
        and collect the whole run of results rather than emitting them one by one.
        """
        converted: list[dict[str, Any]] = []
        index = 0
        total = len(messages)

        while index < total:
            message = messages[index]

            # Anthropic has no system role inside the message list.
            if message.get("role") == "system":
                index += 1
                continue

            if message.get("role") == "tool":
                # Defensive: a leading orphan result has no assistant turn to
                # attach to. Attach it to a synthetic user message rather than
                # emitting an invalid block sequence.
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": str(message.get("content", "")),
                        **({"is_error": True} if _looks_like_error(message) else {}),
                    }
                ]
                converted.append({"role": "user", "content": blocks})
                index += 1
                continue

            content_blocks: list[dict[str, Any]] = []
            text = message.get("content")
            if isinstance(text, str) and text.strip():
                content_blocks.append({"type": "text", "text": text})
            elif isinstance(text, list):
                # Already in Anthropic block form (e.g. after compaction).
                content_blocks.extend(text)

            for call in message.get("tool_calls") or []:
                block = tool_call_to_block(call)
                if block is not None:
                    content_blocks.append(block)

            # Preserve extended thinking if the backend returned it, so a
            # multi-turn conversation does not lose its reasoning context.
            if isinstance(message.get("reasoning"), str) and message["reasoning"]:
                content_blocks.insert(0, {"type": "thinking", "thinking": message["reasoning"]})

            if content_blocks:
                converted.append(
                    {"role": "assistant" if message.get("role") == "assistant" else "user",
                     "content": content_blocks}
                )

            # Merge the immediately following run of tool results into one
            # user turn, which is the shape Anthropic requires.
            following = index + 1
            result_blocks: list[dict[str, Any]] = []
            while following < total and messages[following].get("role") == "tool":
                result = messages[following]
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.get("tool_call_id", ""),
                        "content": str(result.get("content", "")),
                        **({"is_error": True} if _looks_like_error(result) else {}),
                    }
                )
                following += 1

            if result_blocks:
                converted.append({"role": "user", "content": result_blocks})
                index = following
                continue

            index += 1

        return _enforce_alternation(converted)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": self.to_anthropic_messages(sanitize_messages(messages)),
            "stream": stream,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        # System is a top-level parameter here, not a message.
        if system.strip():
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": _strip_empty_schema(spec.parameters),
                }
                for spec in tools
            ]
        return apply_cache_markers(
            payload=payload,
            messages=payload["messages"],
            policy=self.cache_policy,
            expected_turns=self.expected_turns,
        )

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

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
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

        if stream:
            return await self._stream(payload, headers, on_delta)

        try:
            response = await self._client.post(
                f"{self._base_url}/v1/messages",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return ModelResponse(
                content="", finish_reason="error", error=f"transport error: {exc}"
            )

        if response.status_code >= 400:
            return ModelResponse(
                content="",
                finish_reason="error",
                error=_error_text(response.status_code, response.text),
                usage=Usage(estimated=True),
            )


        try:
            parsed = parse_anthropic_response(response.json())
        except json.JSONDecodeError as exc:
            return ModelResponse(
                content="", finish_reason="error", error=f"invalid JSON response: {exc}"
            )
        self._record_cache(parsed.usage)
        return parsed

    def _record_cache(self, usage: Usage) -> None:
        """Accumulate realised cache usage for this provider.

        The policy is a prediction; this is the measurement. A cache read below
        the break-even threshold shows up as a cost ratio above 1.0, which is the
        signal to stop caching rather than leave the write premium unrepaid.
        """
        self.cache_stats.record(
            read=usage.cache_read_tokens,
            write=usage.cache_write_tokens,
            input_tokens=usage.input_tokens,
        )

    async def _stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_delta: Any = None,
    ) -> ModelResponse:
        """Consume Anthropic's SSE stream.

        Anthropic streams typed events rather than raw text chunks: content
        blocks open and close, and deltas arrive tagged by block type. Tool
        arguments stream as ``input_json_delta`` fragments that must be joined
        per block index before parsing.
        """
        payload = {**payload, "stream": True}
        parts: list[str] = []
        reasoning: list[str] = []
        partial_json: dict[int, list[str]] = {}
        pending_blocks: dict[int, dict[str, Any]] = {}
        stop_reason = "stop"
        usage = Usage(estimated=True)

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/messages",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    return ModelResponse(
                        content="", finish_reason="error", error=_error_text(response.status_code, body)
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    kind = event.get("type")

                    if kind == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            pending_blocks[event.get("index", 0)] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                            }
                            partial_json.setdefault(event.get("index", 0), [])

                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        delta_kind = delta.get("type")
                        if delta_kind == "text_delta":
                            text = delta.get("text", "")
                            parts.append(text)
                            # Forward fragments as they arrive; reasoning deltas
                            # are deliberately not forwarded, since they are not
                            # part of the user-visible answer.
                            if on_delta is not None and text:
                                await on_delta(text)
                        elif delta_kind == "thinking_delta":
                            reasoning.append(delta.get("thinking", ""))
                        elif delta_kind == "input_json_delta":
                            index = event.get("index", 0)
                            partial_json.setdefault(index, []).append(delta.get("partial_json", ""))

                    elif kind == "message_delta":
                        delta = event.get("delta") or {}
                        if delta.get("stop_reason"):
                            stop_reason = delta["stop_reason"]
                        # Anthropic splits usage across two events: input tokens
                        # arrive in message_start, output tokens here. Merging
                        # rather than replacing keeps both, since the second
                        # chunk reports only the metric it owns.
                        chunk = event.get("usage")
                        if isinstance(chunk, dict):
                            usage = _merge_stream_usage(usage, chunk, len("".join(parts)))

                    elif kind == "message_start":
                        chunk = event.get("message", {}).get("usage")
                        if isinstance(chunk, dict):
                            usage = _merge_stream_usage(usage, chunk, 0)

                    elif kind == "error":
                        detail = event.get("error", {}).get("message", "stream error")
                        return ModelResponse(
                            content="".join(parts), finish_reason="error", error=detail
                        )
        except httpx.HTTPError as exc:
            return ModelResponse(
                content="", finish_reason="error", error=f"stream transport error: {exc}"
            )

        calls = []
        for index, block in sorted(pending_blocks.items()):
            name = block["name"]
            if not isinstance(name, str) or not name.strip():
                continue
            calls.append(
                ToolCall(
                    id=block["id"] or f"call_{index}",
                    name=name,
                    arguments=parse_tool_call_arguments("".join(partial_json.get(index, []))),
                )
            )

        response = ModelResponse(
            content="".join(parts),
            tool_calls=tuple(calls),
            finish_reason=_map_stop_reason(stop_reason),
            usage=usage,
            reasoning="".join(reasoning),
        )
        self._record_cache(response.usage)
        return response

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_anthropic_response(payload: dict[str, Any]) -> ModelResponse:
    """Normalise a Messages API response into :class:`ModelResponse`."""
    if payload.get("type") == "error" or "content" not in payload:
        return ModelResponse(
            content="",
            finish_reason="error",
            error=(payload.get("error") or {}).get("message", "unexpected payload"),
            usage=extract_anthropic_usage(payload.get("usage")),
        )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: list[ToolCall] = []

    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif kind == "tool_use":
            name = block.get("name")
            # A nameless tool_use block cannot be dispatched. Dropping it here
            # keeps it out of the transcript, where it would otherwise be
            # replayed and rejected on every subsequent turn.
            if isinstance(name, str) and name.strip():
                calls.append(
                    ToolCall(
                        id=block.get("id") or f"call_{len(calls)}",
                        name=name,
                        arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                    )
                )

    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tuple(calls),
        finish_reason=_map_stop_reason(payload.get("stop_reason")),
        usage=extract_anthropic_usage(payload.get("usage")),
        reasoning="".join(reasoning_parts),
    )


def _map_stop_reason(stop_reason: str | None) -> str:
    """Map Anthropic stop reasons onto the internal vocabulary."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "refusal",
        "pause_turn": "stop",
    }.get(stop_reason or "", "stop")


def extract_anthropic_usage(raw: Any) -> Usage:
    """Read token counts, flagging absent data as estimated.

    Anthropic omits ``cache_creation_input_tokens`` on most responses; that
    stays ``None`` rather than becoming ``0`` so callers can tell "not reported"
    from "reported as zero".
    """
    if not isinstance(raw, dict):
        return Usage(estimated=True)

    def _int(key: str) -> int:
        value = raw.get(key)
        return value if isinstance(value, int) else 0

    def _opt(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) else None

    return Usage(
        input_tokens=_int("input_tokens"),
        output_tokens=_int("output_tokens"),
        cache_read_tokens=_opt("cache_read_input_tokens"),
        cache_write_tokens=_opt("cache_creation_input_tokens"),
        estimated=False,
    )


def _merge_stream_usage(current: Usage, raw: dict[str, Any], output_tokens: int) -> Usage:
    """Fold a streamed usage chunk into what has been collected so far.

    The API reports different metrics in different events -- input tokens in
    ``message_start``, output tokens in ``message_delta`` -- so each chunk only
    carries the fields it owns. Overwriting wholesale would drop the fields the
    new chunk omits, under-reporting input tokens; substituting zero would be
    worse, since a zero reads as a measured value.

    Key presence, not value, decides what to take: an omitted key leaves the
    previous figure intact, while an explicit value (including ``0``) wins.
    """
    incoming = extract_anthropic_usage(raw)

    def pick(attribute: str, key: str) -> int | None:
        value = raw.get(key)
        if isinstance(value, int):
            return value
        return getattr(current, attribute)

    return Usage(
        input_tokens=pick("input_tokens", "input_tokens") or 0,
        output_tokens=pick("output_tokens", "output_tokens") or output_tokens,
        cache_read_tokens=pick("cache_read_tokens", "cache_read_input_tokens"),
        cache_write_tokens=pick("cache_write_tokens", "cache_creation_input_tokens"),
        estimated=incoming.estimated and current.estimated,
    )


def tool_call_to_block(call: dict[str, Any]) -> dict[str, Any] | None:
    """Render an internal OpenAI-shaped tool_call as an Anthropic block."""
    function = call.get("function") or {}
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = parse_tool_call_arguments(arguments)
    return {
        "type": "tool_use",
        "id": call.get("id", ""),
        "name": name,
        "input": arguments if isinstance(arguments, dict) else {},
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _strip_empty_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Ensure a tool schema is acceptable to Anthropic.

    Anthropic rejects an ``input_schema`` whose ``properties`` is empty when a
    schema declares one, and some gateways are stricter still about empty
    object schemas. Providing an explicit empty ``properties`` map is the form
    both accept.
    """
    schema = dict(parameters or {})
    schema.setdefault("type", "object")
    if "properties" not in schema:
        schema["properties"] = {}
    return schema


def _looks_like_error(message: dict[str, Any]) -> bool:
    """Heuristic: does this tool message report a failure?

    The internal transcript keeps error state in ``status`` while the wire
    format has no room for it, so it is inferred from the message body. Getting
    this wrong only affects how the model reads a failure, not control flow.
    """
    content = str(message.get("content", "")).lower()
    return content.startswith(("error", "invalid parameters", "unknown tool", "tool execution failed"))


def _enforce_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge same-role neighbours, which Anthropic rejects.

    The API requires strictly alternating user/assistant turns. Compaction or a
    caller-supplied transcript can easily produce two assistants or two users in
    a row, which returns a 400.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            previous = merged[-1]
            blocks = list(previous["content"])
            blocks.extend(message["content"])
            merged[-1] = {"role": previous["role"], "content": blocks}
            continue
        merged.append(message)
    return merged


def _error_text(status: int, body: str) -> str:
    try:
        parsed = json.loads(body)
        message = (parsed.get("error") or {}).get("message")
        if message:
            return str(message)
    except json.JSONDecodeError:
        pass
    error = classify_error(status, body)
    return str(error)