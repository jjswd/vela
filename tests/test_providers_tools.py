"""Tests for provider parsing, error classification, and tool contracts."""

from __future__ import annotations

import json

import pytest

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.providers.base import (
    AuthError,
    BillingError,
    ContextWindowError,
    ProviderError,
    RateLimitError,
    TransientError,
    classify_error,
    extract_usage,
    is_retryable,
    parse_openai_response,
    parse_tool_call_arguments,
    sanitize_messages,
)
from vela.tools.base import Tool, schema_from_signature, validate_arguments
from vela.tools.registry import ToolRegistry


class Sample(Tool):
    name = "sample"
    description = "sample tool"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "count": {"type": "integer"},
            "mode": {"type": "string", "enum": ["read", "write"]},
        },
        "required": ["path"],
    }

    async def run(self, arguments: dict) -> str:
        return "ok"


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, "", AuthError),
        (402, "", BillingError),
        (429, "", RateLimitError),
        (500, "", TransientError),
        (503, "", TransientError),
        (400, "insufficient_quota", BillingError),
        (400, "invalid api key", AuthError),
        (400, "maximum context length exceeded", ContextWindowError),
        (418, "teapot", ProviderError),
    ],
)
def test_classify_error(status, body, expected) -> None:
    assert isinstance(classify_error(status, body), expected)


def test_billing_errors_are_not_retryable() -> None:
    """Retrying an out-of-credit account just wastes time and money."""
    assert not is_retryable(BillingError("no credit"))
    assert not is_retryable(AuthError("bad key"))
    assert not is_retryable(ContextWindowError("too long"))
    assert is_retryable(TransientError("503"))
    assert is_retryable(RateLimitError("slow down"))


# --------------------------------------------------------------------------
# Usage accounting
# --------------------------------------------------------------------------


def test_usage_marks_missing_data_as_estimated() -> None:
    usage = extract_usage({})
    assert usage.estimated is True


def test_usage_addition_preserves_missing_metrics() -> None:
    """None means 'not reported' and must not collapse into a real number."""
    left = Usage(input_tokens=10, cache_read_tokens=None)
    right = Usage(input_tokens=5, cache_read_tokens=100)
    total = left + right
    assert total.cache_read_tokens == 100  # one side has a real value

    both_missing = Usage() + Usage()
    assert both_missing.cache_read_tokens is None
    assert both_missing.estimated is True


def test_usage_sum_is_estimated_if_either_part_was() -> None:
    total = Usage(input_tokens=1, estimated=False) + Usage(input_tokens=1, estimated=True)
    assert total.estimated is True


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def test_parse_tool_calls_and_usage() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    response = parse_openai_response(payload)
    assert response.finish_reason == "tool_calls"
    assert response.wants_tools
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "a.txt"}
    assert response.usage.input_tokens == 100
    assert response.usage.estimated is False


def test_tool_calls_with_bad_names_are_dropped_at_parse_time() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": None, "arguments": "{}"}},
                        {"id": "c2", "function": {"name": "ok", "arguments": "{}"}},
                    ]
                },
            }
        ]
    }
    response = parse_openai_response(payload)
    assert [c.name for c in response.tool_calls] == ["ok"]


def test_abnormal_finish_reason_blocks_tool_execution() -> None:
    """Truncated arguments must never reach a tool."""
    call = (ToolCall(id="c1", name="x", arguments={"a": 1}),)
    assert ModelResponse(tool_calls=call, finish_reason="length").wants_tools is False
    assert ModelResponse(tool_calls=call, finish_reason="tool_calls").wants_tools is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        ('{"a": 1,}', {"a": 1}),          # trailing comma
        ('```json\n{"a": 1}\n```', {"a": 1}),  # fenced
        ('{"a": {"b": 2}', {"a": {"b": 2}}),  # unclosed
        ('{"a": "brace } inside string"}', {"a": "brace } inside string"}),
    ],
)
def test_lenient_json_repair(raw, expected) -> None:
    assert parse_tool_call_arguments(raw) == expected


def test_sanitize_fills_empty_messages() -> None:
    cleaned = sanitize_messages(
        [
            {"role": "assistant", "content": ""},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            {"role": "user", "content": "keep"},
        ]
    )
    assert cleaned[0]["content"] == "(no content)"
    assert cleaned[1]["content"] == "(empty result)"
    assert cleaned[2]["content"] == "keep"


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------


def test_validate_reports_all_problems_at_once() -> None:
    """One round trip should surface every argument error."""
    errors = validate_arguments(
        Sample.parameters, {"count": "many", "mode": "delete", "extra": 1}
    )
    joined = " ".join(errors)
    assert "missing required parameter 'path'" in joined
    assert "'count' should be integer" in joined
    assert "'mode' must be one of" in joined
    assert "unknown parameter 'extra'" in joined


def test_boolean_is_not_accepted_as_integer() -> None:
    """``True`` is an int in Python but not a valid integer argument."""
    errors = validate_arguments(Sample.parameters, {"path": "a", "count": True})
    assert any("integer" in error for error in errors)


def test_schema_from_signature_infers_types() -> None:
    def handler(name: str, count: int = 3, tags: list[str] = []) -> str:
        return name

    schema = schema_from_signature(handler)
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["count"] == {"type": "integer"}
    assert schema["properties"]["tags"] == {"type": "array"}
    assert schema["required"] == ["name"]


async def test_tool_crash_becomes_an_error_result() -> None:
    class Boom(Tool):
        name = "boom"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def run(self, arguments: dict) -> str:
            raise ValueError("kaboom")

    result = await Boom().execute({}, call_id="c1")
    assert result.is_error
    assert "kaboom" in result.content
    assert result.call_id == "c1"


async def test_registry_suggests_typo_corrections() -> None:
    registry = ToolRegistry([Sample()])
    record = await registry.dispatch(
        [ToolCall(id="c1", name="sampl", arguments={"path": "a"})]
    )
    assert "Did you mean 'sample'?" in record[0].result.content


async def test_registry_returns_one_record_per_call() -> None:
    """A timing-out tool still owes the model a matching tool message."""
    import asyncio

    class Slow(Tool):
        name = "slow"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def run(self, arguments: dict) -> str:
            await asyncio.sleep(5)
            return "never"

    registry = ToolRegistry([Slow()])
    records = await registry.dispatch(
        [ToolCall(id="c1", name="slow", arguments={})], timeout=0.05
    )
    assert len(records) == 1
    assert records[0].result.call_id == "c1"
    assert "exceeded" in records[0].result.content
