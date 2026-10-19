"""SSE streaming tests for both wire protocols.

Streaming is where a provider integration most often breaks quietly: text
arrives in pieces, tool arguments arrive as JSON fragments that must be joined
in order, and usage may only show up in the final chunk. A non-streaming test
can pass while any of that is wrong.

These tests drive both backends through a mock transport that emits a realistic
SSE byte stream, and assert the reassembled result equals what a single-shot
response would have produced.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from vela.providers.anthropic import AnthropicProvider
from vela.providers.base import GenerationConfig, ToolSpec
from vela.providers.openai_compat import OpenAICompatProvider


def sse_transport(chunks: list[str]) -> httpx.MockTransport:
    """Serve ``chunks`` as an SSE body, one ``data:`` line per chunk."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(f"data: {chunk}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


def client(chunks: list[str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=sse_transport(chunks))


def openai_provider(chunks: list[str]) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=client(chunks),
    )


def anthropic_provider(chunks: list[str]) -> AnthropicProvider:
    return AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=client(chunks),
    )


def openai_chunk(delta: dict[str, Any], finish: str | None = None) -> str:
    payload: dict[str, Any] = {"choices": [{"delta": delta, "finish_reason": finish}]}
    return json.dumps(payload)


def anthropic_event(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields})


SPEC = ToolSpec(
    name="get_weather",
    description="weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# OpenAI-compatible streaming
# --------------------------------------------------------------------------


async def test_openai_text_deltas_are_concatenated_in_order() -> None:
    chunks = [
        openai_chunk({"role": "assistant"}),
        openai_chunk({"content": "Hello"}),
        openai_chunk({"content": ", "}),
        openai_chunk({"content": "world"}),
        openai_chunk({}, finish="stop"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.content == "Hello, world"
    assert response.finish_reason == "stop"
    assert response.wants_tools is False


async def test_openai_tool_arguments_are_reassembled_from_fragments() -> None:
    """Argument JSON arrives split across deltas and must be joined, not parsed.

    Parsing each fragment independently would fail on every one of them, since
    no single piece is valid JSON on its own.
    """
    chunks = [
        openai_chunk({
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ]
        }),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]}),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Bei'}}]}),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'jing"}'}}]}),
        openai_chunk({}, finish="tool_calls"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [SPEC], stream=True)

    assert response.finish_reason == "tool_calls"
    assert response.wants_tools
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Beijing"}


async def test_openai_parallel_tool_calls_keep_their_own_fragments() -> None:
    """Two interleaved tool calls must not have their arguments cross-contaminated."""
    chunks = [
        openai_chunk({
            "tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "first", "arguments": '{"x":'}},
                {"index": 1, "id": "b", "function": {"name": "second", "arguments": '{"y":'}},
            ]
        }),
        openai_chunk({"tool_calls": [{"index": 1, "function": {"arguments": "2}"}}]}),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
        openai_chunk({}, finish="tool_calls"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)

    calls = {call.id: call.arguments for call in response.tool_calls}
    assert calls == {"a": {"x": 1}, "b": {"y": 2}}


async def test_openai_malformed_json_in_arguments_is_repaired() -> None:
    chunks = [
        openai_chunk({
            "tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "t", "arguments": ""}}
            ]
        }),
        openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1,}'}}]}),
        openai_chunk({}, finish="tool_calls"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.tool_calls[0].arguments == {"a": 1}


async def test_openai_streamed_tool_call_without_name_is_dropped() -> None:
    chunks = [
        openai_chunk({"tool_calls": [{"index": 0, "function": {"name": "", "arguments": "{}"}}]}),
        openai_chunk({}, finish="tool_calls"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.tool_calls == ()


async def test_openai_usage_is_read_from_the_stream() -> None:
    chunks = [
        openai_chunk({"content": "hi"}),
        json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7},
            }
        ),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 7
    assert response.usage.estimated is False


async def test_openai_stream_without_usage_is_marked_estimated() -> None:
    chunks = [openai_chunk({"content": "hi"}), openai_chunk({}, finish="stop"), "[DONE]"]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.usage.estimated is True


async def test_openai_reasoning_content_is_captured() -> None:
    chunks = [
        openai_chunk({"reasoning_content": "thinking..."}),
        openai_chunk({"content": "answer"}),
        openai_chunk({}, finish="stop"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.reasoning == "thinking..."
    assert response.content == "answer"


async def test_openai_ignores_unparseable_chunks() -> None:
    """A malformed frame must not abort the stream."""
    chunks = [
        openai_chunk({"content": "before"}),
        "{not json",
        openai_chunk({"content": "after"}),
        openai_chunk({}, finish="stop"),
        "[DONE]",
    ]
    response = await openai_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.content == "beforeafter"


async def test_openai_empty_stream_does_not_crash() -> None:
    response = await openai_provider(["[DONE]"]).complete(MESSAGES, [], stream=True)
    assert response.content == ""
    assert response.finish_reason == "stop"


async def test_openai_http_error_in_stream_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    provider = OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await provider.complete(MESSAGES, [], stream=True)
    assert response.is_error


async def test_streamed_and_non_streamed_results_agree() -> None:
    """The same reply must be identical either way, or callers get inconsistent text."""
    stream_chunks = [
        openai_chunk({"content": "The answer"}),
        openai_chunk({"content": " is 42"}),
        openai_chunk({}, finish="stop"),
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "The answer is 42"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    streamed = await openai_provider(stream_chunks).complete(MESSAGES, [], stream=True)
    single = await OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ).complete(MESSAGES, [], stream=False)

    assert streamed.content == single.content
    assert streamed.finish_reason == single.finish_reason


async def test_openai_streaming_is_requested_with_stream_flag() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    provider = OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await provider.complete(MESSAGES, [SPEC], system="sys", stream=True)

    assert seen["stream"] is True
    assert seen["tools"][0]["function"]["parameters"] == SPEC.parameters


# --------------------------------------------------------------------------
# Anthropic streaming
# --------------------------------------------------------------------------


async def test_anthropic_text_deltas_are_concatenated() -> None:
    chunks = [
        anthropic_event("message_start", message={"usage": {"input_tokens": 20, "output_tokens": 0}}),
        anthropic_event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "Bon"}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "jour"}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "!"}),
        anthropic_event("content_block_stop", index=0),
        anthropic_event("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 6}),
        anthropic_event("message_stop"),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)

    assert response.content == "Bonjour!"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 20


async def test_anthropic_tool_input_fragments_are_reassembled() -> None:
    chunks = [
        anthropic_event("message_start", message={"usage": {"input_tokens": 30, "output_tokens": 0}}),
        anthropic_event(
            "content_block_start", index=0, content_block={"type": "tool_use", "id": "tu1", "name": "get_weather"}
        ),
        anthropic_event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"ci'}),
        anthropic_event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": 'ty": "Bei'}),
        anthropic_event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": 'jing"}'}),
        anthropic_event("content_block_stop", index=0),
        anthropic_event("message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 12}),
        anthropic_event("message_stop"),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [SPEC], stream=True)

    assert response.finish_reason == "tool_calls"
    assert response.wants_tools
    assert response.tool_calls[0].id == "tu1"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Beijing"}


async def test_anthropic_parallel_tool_blocks_stay_separate() -> None:
    chunks = [
        anthropic_event(
            "content_block_start", index=0, content_block={"type": "tool_use", "id": "a", "name": "first"}
        ),
        anthropic_event(
            "content_block_start", index=1, content_block={"type": "tool_use", "id": "b", "name": "second"}
        ),
        anthropic_event("content_block_delta", index=1, delta={"type": "input_json_delta", "partial_json": '{"y":2}'}),
        anthropic_event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"x":1}'}),
        anthropic_event("message_delta", delta={"stop_reason": "tool_use"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)

    calls = {call.id: call.arguments for call in response.tool_calls}
    assert calls == {"a": {"x": 1}, "b": {"y": 2}}


async def test_anthropic_thinking_deltas_are_separated_from_text() -> None:
    """Reasoning must not leak into the user-visible answer."""
    chunks = [
        anthropic_event(
            "content_block_start", index=0, content_block={"type": "thinking", "thinking": ""}
        ),
        anthropic_event("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "step 1 "}),
        anthropic_event("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "step 2"}),
        anthropic_event("content_block_start", index=1, content_block={"type": "text", "text": ""}),
        anthropic_event("content_block_delta", index=1, delta={"type": "text_delta", "text": "Final."}),
        anthropic_event("message_delta", delta={"stop_reason": "end_turn"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.reasoning == "step 1 step 2"
    assert response.content == "Final."


async def test_anthropic_stream_error_event_is_reported() -> None:
    chunks = [
        anthropic_event("message_start", message={"usage": {"input_tokens": 5, "output_tokens": 0}}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "partial"}),
        anthropic_event("error", error={"message": "overloaded_error"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.is_error
    assert "overloaded" in response.error
    # Partial output received before the failure is preserved.
    assert response.content == "partial"


async def test_anthropic_max_tokens_maps_to_length() -> None:
    chunks = [
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "trunc"}),
        anthropic_event("message_delta", delta={"stop_reason": "max_tokens"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.finish_reason == "length"


async def test_anthropic_nameless_tool_block_is_dropped() -> None:
    chunks = [
        anthropic_event(
            "content_block_start", index=0, content_block={"type": "tool_use", "id": "x", "name": ""}
        ),
        anthropic_event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": "{}"}),
        anthropic_event("message_delta", delta={"stop_reason": "tool_use"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.tool_calls == ()


async def test_anthropic_stream_ignores_unknown_event_types() -> None:
    """New event types must not break an older client."""
    chunks = [
        anthropic_event("message_start", message={"usage": {"input_tokens": 1, "output_tokens": 0}}),
        anthropic_event("some_future_event", payload={"whatever": True}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "ok"}),
        anthropic_event("message_delta", delta={"stop_reason": "end_turn"}),
    ]
    response = await anthropic_provider(chunks).complete(MESSAGES, [], stream=True)
    assert response.content == "ok"


async def test_anthropic_streaming_sends_stream_flag_and_input_schema() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: {}\n\n")

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await provider.complete(MESSAGES, [SPEC], system="sys", stream=True)

    assert seen["stream"] is True
    assert "input_schema" in seen["tools"][0]
    assert seen["system"] == "sys"


async def test_both_backends_produce_identical_results_for_the_same_reply() -> None:
    """Cross-protocol parity: the same answer must reach the caller unchanged."""

    openai_chunks = [
        openai_chunk({"content": "Shared"}),
        openai_chunk({"content": " answer"}),
        openai_chunk({}, finish="stop"),
        "[DONE]",
    ]
    anthropic_chunks = [
        anthropic_event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": "Shared"}),
        anthropic_event("content_block_delta", index=0, delta={"type": "text_delta", "text": " answer"}),
        anthropic_event("message_delta", delta={"stop_reason": "end_turn"}),
    ]

    from_openai = await openai_provider(openai_chunks).complete(MESSAGES, [], stream=True)
    from_anthropic = await anthropic_provider(anthropic_chunks).complete(MESSAGES, [], stream=True)

    assert from_openai.content == from_anthropic.content == "Shared answer"
    assert from_openai.finish_reason == from_anthropic.finish_reason == "stop"

# --------------------------------------------------------------------------
# End-to-end fragmentation over a realistic byte stream
# --------------------------------------------------------------------------


class DelayedByteStream(httpx.AsyncByteStream):
    """Yields the body in small, fixed chunks.

    Real HTTP delivers an SSE body split across packets whose boundaries rarely
    line up with frame or line boundaries. A mock that returns the whole body at
    once hides any buffering bug in the reader, so tests that care about
    incremental delivery must fragment here.
    """

    def __init__(self, raw: bytes, chunk_size: int = 7) -> None:
        self.raw = raw
        self.chunk_size = chunk_size

    async def __aiter__(self):
        for offset in range(0, len(self.raw), self.chunk_size):
            await asyncio.sleep(0)  # yield control, as a real socket would
            yield self.raw[offset:offset + self.chunk_size]


def raw_stream_provider(provider_cls, raw: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=DelayedByteStream(raw),
        )

    return provider_cls(
        GenerationConfig(model="m"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def openai_body(texts: list[str]) -> bytes:
    parts = [
        f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in texts
    ]
    parts.append('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def anthropic_body(texts: list[str], input_tokens: int = 12, output_tokens: int = 7) -> bytes:
    parts = [
        'data: {"type":"message_start","message":{"usage":'
        f'{{"input_tokens":{input_tokens},"output_tokens":0}}}}}}\n\n',
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n',
    ]
    for text in texts:
        parts.append(
            'data: {"type":"content_block_delta","index":0,'
            f'"delta":{{"type":"text_delta","text":"{text}"}}}}\n\n'
        )
    parts.append(
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        f'"usage":{{"output_tokens":{output_tokens}}}}}\n\n'
    )
    return "".join(parts).encode()


async def test_openai_fragments_arrive_one_by_one() -> None:
    """Each network fragment must be forwarded as its own callback.

    Delivering the whole answer in a single callback would pass every
    content assertion yet defeat the purpose of streaming, so the fragment count
    is part of the contract.
    """
    seen: list[str] = []

    async def on_delta(delta: str) -> None:
        seen.append(delta)

    provider = raw_stream_provider(OpenAICompatProvider, openai_body(["你", "好", "，", "世", "界"]))
    response = await provider.complete(MESSAGES, [], stream=True, on_delta=on_delta)

    assert seen == ["你", "好", "，", "世", "界"]
    assert "".join(seen) == response.content == "你好，世界"


async def test_anthropic_fragments_arrive_one_by_one() -> None:
    seen: list[str] = []

    async def on_delta(delta: str) -> None:
        seen.append(delta)

    provider = raw_stream_provider(AnthropicProvider, anthropic_body(["你", "好", "，", "世", "界"]))
    response = await provider.complete(MESSAGES, [], stream=True, on_delta=on_delta)

    assert seen == ["你", "好", "，", "世", "界"]
    assert "".join(seen) == response.content == "你好，世界"


async def test_fragments_survive_frames_split_across_chunks() -> None:
    """A frame boundary need not align with a network packet boundary.

    The reader must buffer a partial line rather than dropping it, which would
    silently truncate the answer mid-word.
    """
    body = (
        b'data: {"choices":[{"delta":{"content":"par"}}]}\n'
        b"\n"
        b'data: {"choices":[{"delta":{"content":"tial"}}]}\n'
        b"\n"
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    seen: list[str] = []

    async def on_delta(delta: str) -> None:
        seen.append(delta)

    provider = raw_stream_provider(OpenAICompatProvider, body)
    response = await provider.complete(MESSAGES, [], stream=True, on_delta=on_delta)
    assert "".join(seen) == response.content == "partial"


async def test_anthropic_usage_survives_fragmented_transport() -> None:
    """Usage arrives in two separate events; both must survive fragmentation."""
    provider = raw_stream_provider(
        AnthropicProvider, anthropic_body(["ok"], input_tokens=31, output_tokens=9)
    )
    response = await provider.complete(MESSAGES, [], stream=True)
    assert response.usage.input_tokens == 31
    assert response.usage.output_tokens == 9


async def test_streaming_is_off_unless_requested() -> None:
    """A caller that did not ask for streaming must not be handed a callback."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["stream"] = json.loads(request.content)["stream"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
        )

    provider = OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await provider.complete(MESSAGES, [], stream=False)
    assert captured["stream"] is False
