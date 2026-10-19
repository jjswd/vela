"""Tests for the Anthropic provider's format conversion and parsing.

The conversion from the internal transcript to Anthropic's block format is
where this backend actually differs from the OpenAI-compatible one, so most of
these tests target that: tool-result placement, alternation, and the stop-reason
vocabulary. The runner and loop are not exercised here because they must not
know which backend is in use.
"""

from __future__ import annotations

import json

import pytest

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.providers.anthropic import (
    AnthropicProvider,
    parse_anthropic_response,
)
from vela.providers.base import GenerationConfig, ToolSpec, sanitize_messages


def convert(messages):
    return AnthropicProvider.to_anthropic_messages(messages)


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


def test_system_prompt_is_not_a_message() -> None:
    """Anthropic takes the system prompt as a top-level field."""
    wire = convert([{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}])
    assert all(message["role"] != "system" for message in wire)
    assert wire[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}


def test_plain_turns_become_text_blocks() -> None:
    wire = convert(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    )
    assert wire[0]["role"] == "user"
    assert wire[0]["content"][0]["type"] == "text"
    assert wire[1]["role"] == "assistant"


def test_tool_calls_become_tool_use_blocks() -> None:
    wire = convert(
        [
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            }
        ]
    )
    blocks = wire[0]["content"]
    tool_use = [b for b in blocks if b["type"] == "tool_use"][0]
    assert tool_use["name"] == "read_file"
    assert tool_use["input"] == {"path": "a.txt"}


def test_tool_results_are_merged_into_one_user_turn() -> None:
    """Anthropic requires every result for one assistant turn in a single message.

    Emitting one user message per result produces consecutive user turns and a
    400 from the API.
    """
    wire = convert(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result A"},
            {"role": "tool", "tool_call_id": "c2", "content": "result B"},
        ]
    )

    # One assistant turn, then exactly one user turn holding both results.
    assert [m["role"] for m in wire] == ["assistant", "user"]
    blocks = wire[1]["content"]
    assert len(blocks) == 2
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]
    assert all(b["type"] == "tool_result" for b in blocks)


def test_tool_results_immediately_follow_their_assistant_turn() -> None:
    wire = convert(
        [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "a", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "assistant", "content": "final"},
        ]
    )
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "assistant"]
    # The result turn sits directly after the assistant turn that asked for it.
    assert wire[2]["content"][0]["type"] == "tool_result"


def test_consecutive_same_role_turns_are_merged() -> None:
    """The API rejects non-alternating roles; compaction can produce them."""
    wire = convert(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply"},
            {"role": "assistant", "content": "more"},
        ]
    )
    assert [m["role"] for m in wire] == ["user", "assistant"]
    assert len(wire[0]["content"]) == 2
    assert len(wire[1]["content"]) == 2


def test_orphan_tool_result_does_not_break_the_sequence() -> None:
    wire = convert([{"role": "tool", "tool_call_id": "c9", "content": "orphan"}])
    assert len(wire) == 1
    assert wire[0]["role"] == "user"
    assert wire[0]["content"][0]["type"] == "tool_result"


def test_failed_tool_result_is_flagged_as_error() -> None:
    """Marking errors lets the model distinguish failure from a valid empty result."""
    wire = convert(
        [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "a", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "Unknown tool 'a'"},
        ]
    )
    assert wire[1]["content"][0]["is_error"] is True


def test_successful_result_is_not_marked_as_error() -> None:
    wire = convert(
        [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "a", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "60"},
        ]
    )
    assert "is_error" not in wire[1]["content"][0]


def test_nameless_tool_call_is_dropped() -> None:
    """An unexecutable block must not reach the wire format."""
    wire = convert(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": None, "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "ok", "arguments": "{}"}},
                ],
            }
        ]
    )
    names = [b.get("name") for b in wire[0]["content"] if b["type"] == "tool_use"]
    assert names == ["ok"]


def test_reasoning_is_preserved_for_multi_turn() -> None:
    wire = convert([{"role": "assistant", "content": "answer", "reasoning": "because"}])
    kinds = [b["type"] for b in wire[0]["content"]]
    assert kinds[0] == "thinking"


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def test_parse_text_response() -> None:
    payload = {
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 30, "output_tokens": 5},
    }
    response = parse_anthropic_response(payload)
    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 30
    assert response.usage.estimated is False


def test_parse_tool_use_response() -> None:
    payload = {
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "北京"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 40, "output_tokens": 8},
    }
    response = parse_anthropic_response(payload)
    assert response.finish_reason == "tool_calls"
    assert response.wants_tools
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "北京"}


def test_parse_thinking_response() -> None:
    payload = {
        "content": [
            {"type": "thinking", "thinking": "step one"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    response = parse_anthropic_response(payload)
    assert response.reasoning == "step one"
    assert response.content == "answer"


def test_parse_drops_nameless_tool_use() -> None:
    payload = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": None, "input": {}},
            {"type": "tool_use", "id": "t2", "name": "ok", "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert [c.name for c in parse_anthropic_response(payload).tool_calls] == ["ok"]


def test_parse_error_envelope() -> None:
    payload = {"type": "error", "error": {"message": "overloaded"}}
    response = parse_anthropic_response(payload)
    assert response.is_error
    assert "overloaded" in response.error


@pytest.mark.parametrize(
    "stop,expected",
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "refusal"),
        (None, "stop"),
        ("something_new", "stop"),
    ],
)
def test_stop_reason_mapping(stop, expected) -> None:
    payload = {"content": [{"type": "text", "text": "x"}], "stop_reason": stop}
    assert parse_anthropic_response(payload).finish_reason == expected


def test_max_tokens_maps_to_length_and_blocks_tools() -> None:
    """Truncated arguments must not be dispatched as a tool call."""
    payload = {
        "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        "stop_reason": "max_tokens",
    }
    response = parse_anthropic_response(payload)
    assert response.finish_reason == "length"
    assert response.wants_tools is False


def test_usage_keeps_missing_cache_metrics_as_none() -> None:
    payload = {
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    usage = parse_anthropic_response(payload).usage
    assert usage.cache_write_tokens is None
    assert usage.cache_read_tokens is None
    # Summing with another record must not invent a zero for a metric never reported.
    combined = usage + Usage(input_tokens=1, cache_read_tokens=7)
    assert combined.cache_read_tokens == 7


def test_cache_metrics_are_read_when_present() -> None:
    payload = {
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "cache_read_input_tokens": 1200,
            "cache_creation_input_tokens": 300,
        },
    }
    usage = parse_anthropic_response(payload).usage
    assert usage.cache_read_tokens == 1200
    assert usage.cache_write_tokens == 300


# --------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------


def build_payload(provider: AnthropicProvider, messages, tools=(), system=""):
    return provider._build_payload(list(messages), list(tools), system, stream=False)


def make_provider() -> AnthropicProvider:
    return AnthropicProvider(GenerationConfig(model="claude-sonnet-4-5"), api_key="test")


def test_payload_places_system_at_the_top_level() -> None:
    payload = build_payload(make_provider(), [{"role": "user", "content": "hi"}], system="rules")
    assert payload["system"] == "rules"
    assert all(message["role"] != "system" for message in payload["messages"])


def test_payload_always_sets_max_tokens() -> None:
    """Anthropic rejects a request without max_tokens."""
    payload = build_payload(make_provider(), [{"role": "user", "content": "hi"}])
    assert payload["max_tokens"] > 0


def test_payload_uses_input_schema_not_parameters() -> None:
    spec = ToolSpec(
        name="get_weather",
        description="get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    payload = build_payload(make_provider(), [{"role": "user", "content": "hi"}], tools=[spec])
    tool = payload["tools"][0]
    assert "input_schema" in tool
    assert "parameters" not in tool
    assert tool["input_schema"]["properties"]["city"]["type"] == "string"


def test_empty_schema_is_given_explicit_properties() -> None:
    spec = ToolSpec(name="noop", description="x", parameters={"type": "object"})
    payload = build_payload(make_provider(), [{"role": "user", "content": "hi"}], tools=[spec])
    assert payload["tools"][0]["input_schema"]["properties"] == {}


def test_payload_is_compatible_with_the_shared_sanitiser() -> None:
    """Sanitising before conversion keeps both backends on the same footing."""
    messages = sanitize_messages([{"role": "assistant", "content": ""}])
    payload = build_payload(make_provider(), messages)
    assert payload["messages"][0]["content"][0]["text"] == "(no content)"