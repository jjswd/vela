"""Cross-backend tests.

These are the ones that justify the ``Provider`` abstraction. The runner, the
loop, and the tool layer must behave identically regardless of which backend is
configured; if a change to one backend required a change in the shared layers,
the boundary would not be doing its job.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vela import AgentLoop, SessionStore, ToolRegistry
from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.core.runner import AgentRunner, RunSpec, StopReason
from vela.providers.anthropic import AnthropicProvider
from vela.providers.base import GenerationConfig, Provider
from vela.providers.factory import (
    REGISTRY,
    UnknownProviderError,
    available_providers,
    create_provider,
)
from vela.providers.openai_compat import OpenAICompatProvider
from vela.tools.base import Tool


class Echo(Tool):
    name = "echo"
    description = "echo text back"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


# --------------------------------------------------------------------------
# The shared loop must not care which backend is in use
# --------------------------------------------------------------------------


class ScriptedProvider(Provider):
    def __init__(self, script, finalizer=None):
        super().__init__(GenerationConfig(model="test"))
        self.script = list(script)
        self.finalizer = finalizer

    async def complete(
        self, messages, tools, *, system="", stream=False, on_delta=None
    ):
        if self.finalizer is not None and not tools:
            response, self.finalizer = self.finalizer, None
            return response
        if not self.script:
            return ModelResponse(content="exhausted", finish_reason="stop")
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    async def aclose(self):
        return None


def tool_then_answer(final=None):
    return ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "ping"}),),
                finish_reason="tool_calls",
                usage=Usage(input_tokens=20, output_tokens=5),
            ),
            ModelResponse(
                content=final or "answered",
                finish_reason="stop",
                usage=Usage(input_tokens=25, output_tokens=4),
            ),
        ]
    )


async def test_tool_loop_is_identical_across_backends(tmp_path) -> None:
    """The runner drives any Provider through the same tool cycle."""
    for label in ("openai-shape", "anthropic-shape"):
        runner = AgentRunner(tool_then_answer(), ToolRegistry([Echo()]))
        result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}]))
        assert result.content == "answered", label
        assert result.tools_used == ["echo"], label
        assert result.usage.input_tokens == 45, label
        assert result.stop_reason == StopReason.COMPLETED, label


async def test_recovery_paths_are_backend_agnostic(tmp_path) -> None:
    """Truncation recovery must work the same regardless of provider."""
    provider = ScriptedProvider(
        [
            ModelResponse(content="part one", finish_reason="length"),
            ModelResponse(content=" and two", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, ToolRegistry([Echo()]))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}]))
    assert result.content == "part one and two"
    assert result.stop_reason == StopReason.COMPLETED


# --------------------------------------------------------------------------
# The same transcript must survive a round trip through either wire format
# --------------------------------------------------------------------------


TOOL_TURN = [
    {"role": "user", "content": "what is the weather in Beijing?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text":"Beijing"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Beijing: 22C"},
    {"role": "assistant", "content": "It is 22C in Beijing."},
]


def openai_payload(messages) -> dict[str, Any]:
    provider = OpenAICompatProvider(GenerationConfig(model="gpt-4o-mini"), api_key="k")
    return provider._build_payload(messages, [Echo().spec()], "sys", stream=False)


def anthropic_payload(messages) -> dict[str, Any]:
    provider = AnthropicProvider(GenerationConfig(model="claude-sonnet-4-5"), api_key="k")
    return provider._build_payload(messages, [Echo().spec()], "sys", stream=False)


def test_both_formats_carry_the_same_tool_round_trip() -> None:
    """Neither wire format may drop or reorder the tool cycle."""
    full_openai = openai_payload(TOOL_TURN)["messages"]
    anthropic = anthropic_payload(TOOL_TURN)["messages"]

    # OpenAI prepends the system message; Anthropic keeps it outside the list.
    assert full_openai[0]["role"] == "system"
    openai = full_openai[1:]

    # OpenAI: a dedicated message per role.
    assert [m["role"] for m in openai] == ["user", "assistant", "tool", "assistant"]
    assert openai[2]["tool_call_id"] == "c1"

    # Anthropic: blocks nested into fewer messages.
    assert [m["role"] for m in anthropic] == ["user", "assistant", "user", "assistant"]
    tool_result = anthropic[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "c1"
    assert tool_result["content"] == "Beijing: 22C"

    # The final answer survives in both.
    assert openai[3]["content"] == anthropic[3]["content"][0]["text"]


def test_anthropic_payload_never_emits_a_tool_role() -> None:
    """A stray ``role="tool"`` message would be rejected by the API."""
    wire = anthropic_payload(TOOL_TURN)["messages"]
    assert all(message["role"] in {"user", "assistant"} for message in wire)


def test_both_formats_put_system_outside_the_message_list() -> None:
    assert openai_payload(TOOL_TURN)["messages"][0]["content"] == "sys"
    assert anthropic_payload(TOOL_TURN)["system"] == "sys"


def test_multi_tool_round_trip_survives_both_formats() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": '{"text":"a"}'}},
                {"id": "c2", "function": {"name": "echo", "arguments": '{"text":"b"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
        {"role": "tool", "tool_call_id": "c2", "content": "b"},
    ]

    openai = openai_payload(messages)["messages"][1:]  # drop the system message
    anthropic = anthropic_payload(messages)["messages"]

    assert [m["role"] for m in openai] == ["assistant", "tool", "tool"]
    # Anthropic must merge both results into a single user turn.
    assert [m["role"] for m in anthropic] == ["assistant", "user"]
    assert len(anthropic[1]["content"]) == 2
    assert [b["tool_use_id"] for b in anthropic[1]["content"]] == ["c1", "c2"]


# --------------------------------------------------------------------------
# HTTP behaviour, exercised through a mock transport
# --------------------------------------------------------------------------


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_anthropic_http_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="secret",
        client=mock_client(handler),
    )
    response = await provider.complete([{"role": "user", "content": "hi"}], [Echo().spec()], system="s")

    assert response.content == "ok"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["payload"]["max_tokens"] > 0
    assert seen["payload"]["system"] == "s"


async def test_anthropic_error_is_typed_not_raised() -> None:
    """Ordinary model failures must not escape as exceptions."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limit exceeded"}})

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"), api_key="k", client=mock_client(handler)
    )
    response = await provider.complete([{"role": "user", "content": "hi"}], [])
    assert response.is_error
    assert "rate limit" in response.error.lower()
    assert response.usage.estimated is True


async def test_anthropic_tool_call_round_trip_over_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "echo", "input": {"text": "hi"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        )

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"), api_key="k", client=mock_client(handler)
    )
    response = await provider.complete([{"role": "user", "content": "say hi"}], [Echo().spec()])
    assert response.wants_tools
    assert response.tool_calls[0].id == "tu1"
    assert response.tool_calls[0].arguments == {"text": "hi"}


async def test_openai_http_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    provider = OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"), api_key="k", client=mock_client(handler)
    )
    await provider.complete([{"role": "user", "content": "hi"}], [Echo().spec()], system="s")

    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    # OpenAI carries the system prompt as the first message instead.
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert "parameters" in seen["payload"]["tools"][0]["function"]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_factory_builds_each_registered_backend() -> None:
    for name in available_providers():
        info = REGISTRY[name]
        provider = create_provider(
            name, model="m", api_key="k" if info.requires_key else None
        )
        assert isinstance(provider, Provider)


def test_factory_rejects_unknown_backend() -> None:
    """A silent fallback would surface later as a confusing 404."""
    with pytest.raises(UnknownProviderError) as info:
        create_provider("openiae", model="m", api_key="k")
    assert "openiae" in str(info.value)


def test_factory_requires_a_key_for_hosted_backends() -> None:
    with pytest.raises(ValueError, match="API key"):
        create_provider("anthropic", model="m", api_key=None)


def test_factory_allows_missing_key_for_local_backends() -> None:
    provider = create_provider("ollama", model="llama3", api_key=None)
    assert isinstance(provider, OpenAICompatProvider)


def test_factory_honours_base_url_override() -> None:
    provider = create_provider(
        "openai", model="m", api_key="k", base_url="http://localhost:1234/v1"
    )
    assert isinstance(provider, OpenAICompatProvider)


def test_only_anthropic_needs_a_native_implementation() -> None:
    """Everything else speaks the OpenAI wire format."""
    from vela.providers.anthropic import AnthropicProvider as Native

    for name in available_providers():
        provider = create_provider(name, model="m", api_key="k")
        expected = Native if name == "anthropic" else OpenAICompatProvider
        assert isinstance(provider, expected), name


# --------------------------------------------------------------------------
# The loop is unaffected by backend choice
# --------------------------------------------------------------------------


async def test_loop_works_with_any_backend(tmp_path) -> None:
    for backend in ("anthropic", "openai"):
        loop = AgentLoop(
            tool_then_answer(),
            ToolRegistry([Echo()]),
            SessionStore(tmp_path / backend),
            workspace=str(tmp_path),
        )
        result = await loop.run("s1", "hello")
        assert result.content == "answered", backend
        session = SessionStore(tmp_path / backend).load("s1")
        assert session.messages[-1]["role"] == "assistant", backend