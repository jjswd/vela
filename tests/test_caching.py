"""Tests for prompt caching: cost model, breakpoint placement, measurement.

Caching is the one feature where doing it "correctly" can still lose money, so
the tests concentrate on the decision logic first and the wire format second.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vela.providers.anthropic import AnthropicProvider
from vela.providers.anthropic_cache import (
    CACHE_CONTROL_EPHEMERAL,
    MAX_BREAKPOINTS,
    apply_cache_markers,
    mark_history,
    mark_system,
    mark_tools,
)
from vela.providers.base import GenerationConfig, ToolSpec
from vela.providers.caching import (
    PRICE_CACHE_READ,
    CachePolicy,
    CacheStats,
    CacheTTL,
    estimate_savings,
    plan,
)


def message(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


def test_break_even_matches_published_prices() -> None:
    """5m writes (1.25x) repay at two uses; 1h writes (2.0x) need three."""
    assert CacheTTL.FIVE_MINUTES.break_even_uses == 2
    assert CacheTTL.ONE_HOUR.break_even_uses == 3


@pytest.mark.parametrize("ttl", [CacheTTL.FIVE_MINUTES, CacheTTL.ONE_HOUR])
def test_caching_below_break_even_costs_more(ttl: CacheTTL) -> None:
    """One request always pays the write premium with nothing to show for it."""
    write = ttl.write_multiplier
    single = write + 0 * PRICE_CACHE_READ
    assert single > 1.0


@pytest.mark.parametrize(
    "ttl,uses", [(CacheTTL.FIVE_MINUTES, 2), (CacheTTL.ONE_HOUR, 3), (CacheTTL.FIVE_MINUTES, 10)]
)
def test_caching_above_break_even_saves(ttl: CacheTTL, uses: int) -> None:
    cached = ttl.write_multiplier + (uses - 1) * PRICE_CACHE_READ
    assert cached < uses


def test_policy_declines_when_conversation_is_too_short() -> None:
    """A two-request conversation would lose money with caching on."""
    policy = CachePolicy(enabled=True, ttl=CacheTTL.ONE_HOUR)
    assert policy.should_cache(10) is True
    assert policy.should_cache(1) is False


def test_disabled_policy_never_caches() -> None:
    assert CachePolicy(enabled=False).should_cache(1000) is False


def test_plan_is_empty_when_not_worth_it() -> None:
    resolved = plan(CachePolicy(enabled=True), message_count=40, expected_turns=1)
    assert resolved.has_breakpoints is False


def test_plan_marks_stable_prefix_when_worth_it() -> None:
    resolved = plan(CachePolicy(enabled=True), message_count=40, expected_turns=20)
    assert resolved.has_breakpoints is True
    assert resolved.mark_system is True
    assert resolved.mark_tools is True


def test_estimate_savings_scales_with_turns() -> None:
    short = estimate_savings(prefix_tokens=1000, tail_tokens=200, turns=2, ttl=CacheTTL.ONE_HOUR)
    long = estimate_savings(prefix_tokens=1000, tail_tokens=200, turns=20, ttl=CacheTTL.ONE_HOUR)
    assert short < 0  # a 1h write cannot repay itself in two turns
    assert long > 0
    assert long > short


def test_estimate_savings_is_zero_for_a_single_turn() -> None:
    assert estimate_savings(prefix_tokens=1000, tail_tokens=100, turns=1, ttl=CacheTTL.FIVE_MINUTES) == 0


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def test_stats_show_a_saving_on_repeat_use() -> None:
    """Model a realistic shape: a large stable prefix, small changing tail."""
    stats = CacheStats()
    # First turn writes the cache; later turns read it. The uncached input is
    # the changing tail only -- the cached prefix is billed separately.
    stats.record(read=0, write=9_000, input_tokens=1_000)
    for _ in range(9):
        stats.record(read=9_000, write=0, input_tokens=1_000)

    assert stats.hit_rate > 0.85
    # Hand-checked: writes 9000*1.25 + reads 81000*0.1 + fresh 10000, against a
    # baseline of 100000 full-price tokens, gives 0.2935.
    assert stats.effective_cost_ratio == pytest.approx(0.2935, abs=0.001)


def test_stats_show_a_loss_when_the_cache_is_never_reused() -> None:
    """Writing and never reading must be reported as a loss, not a win."""
    stats = CacheStats()
    stats.record(read=0, write=5_000, input_tokens=5_000)

    assert stats.hit_rate == 0.0
    assert stats.effective_cost_ratio > 1.0


def test_missing_cache_metrics_do_not_inflate_savings() -> None:
    """A backend that reports no cache usage must look like caching did nothing."""
    stats = CacheStats()
    for _ in range(5):
        stats.record(read=None, write=None, input_tokens=1_000)
    assert stats.cache_read_tokens == 0
    assert stats.effective_cost_ratio == pytest.approx(1.0)


def test_stats_summary_is_serialisable() -> None:
    stats = CacheStats()
    stats.record(read=100, write=0, input_tokens=1_000)
    summary = stats.summary()
    assert summary["turns"] == 1
    json.dumps(summary)  # must not raise


# --------------------------------------------------------------------------
# Breakpoint placement
# --------------------------------------------------------------------------


def test_only_the_last_tool_is_marked() -> None:
    """Marking several tools would spend the breakpoint budget for no gain."""
    tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    marked = mark_tools(tools)
    assert [tool.get("cache_control") for tool in marked] == [None, None, CACHE_CONTROL_EPHEMERAL]


def test_system_becomes_blocks_because_a_plain_string_cannot_be_marked() -> None:
    marked = mark_system("you are helpful")
    assert marked == [
        {"type": "text", "text": "you are helpful", "cache_control": CACHE_CONTROL_EPHEMERAL}
    ]


def test_blank_system_is_not_marked() -> None:
    assert mark_system("   ") is None


def test_ttl_is_carried_on_the_marker() -> None:
    marked = mark_system("sys", ttl="1h")
    assert marked[0]["cache_control"]["ttl"] == "1h"


def test_history_marker_skips_the_most_recent_turns() -> None:
    """Marking a message that changes next turn would invalidate the cache."""
    messages = [
        message("user", "old question"),
        message("assistant", "old answer"),
        message("user", "recent question"),
        message("assistant", "recent answer"),
    ]
    marked = mark_history(messages, turns_from_end=2)
    assert marked[1]["content"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # The volatile tail carries no marker.
    assert "cache_control" not in marked[2]["content"]
    assert "cache_control" not in marked[3]["content"]


def test_history_marker_is_skipped_when_everything_is_recent() -> None:
    messages = [message("user", "q"), message("assistant", "a")]
    assert mark_history(messages, turns_from_end=2) == messages


def test_history_marker_preserves_message_order() -> None:
    """Reordering the transcript would change the cached prefix entirely."""
    messages = [message("user", f"m{i}") for i in range(6)]
    marked = mark_history(messages, turns_from_end=2)
    assert [m["role"] for m in marked] == [m["role"] for m in messages]
    for original, result in zip(messages, marked, strict=True):
        text = original["content"]
        if isinstance(result["content"], list):
            assert result["content"][0]["text"] == text


def test_marker_does_not_mutate_the_input() -> None:
    """The caller may reuse the payload, so markers must land on a copy."""
    messages = [message("user", f"m{i}") for i in range(6)]
    mark_history(messages, turns_from_end=2)
    assert all(isinstance(m["content"], str) for m in messages)


def test_marking_is_skipped_entirely_without_a_breakpoint_worth_taking() -> None:
    payload = {"messages": [message("user", "q")], "system": "sys"}
    result = apply_cache_markers(
        payload=payload,
        messages=payload["messages"],
        policy=CachePolicy(enabled=True),
        expected_turns=1,
    )
    assert result["system"] == "sys"  # unchanged
    assert result["messages"] == payload["messages"]


def test_breakpoint_count_stays_within_the_api_limit() -> None:
    """Exceeding four breakpoints is a hard 400 from the API."""
    messages = [message("user", f"m{i}") for i in range(40)]
    payload = {
        "system": "sys",
        "messages": messages,
        "tools": [{"name": "t1"}, {"name": "t2"}],
    }
    result = apply_cache_markers(
        payload=payload,
        messages=messages,
        policy=CachePolicy(enabled=True),
        expected_turns=20,
    )

    def count(node) -> int:
        if isinstance(node, dict):
            return (1 if "cache_control" in node else 0) + sum(
                count(value) for value in node.values()
            )
        if isinstance(node, list):
            return sum(count(item) for item in node)
        return 0

    assert count(result) <= MAX_BREAKPOINTS


# --------------------------------------------------------------------------
# End to end through the provider
# --------------------------------------------------------------------------


def build_provider(policy: CachePolicy, expected_turns: int = 20) -> AnthropicProvider:
    return AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        cache_policy=policy,
        expected_turns=expected_turns,
    )


SPEC = ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})
LONG_HISTORY = [
    message("user", "q1"),
    message("assistant", "a1"),
    message("user", "q2"),
    message("assistant", "a2"),
    message("user", "q3"),
    message("assistant", "a3"),
]


def test_provider_marks_system_and_tools_when_enabled() -> None:
    provider = build_provider(CachePolicy(enabled=True))
    payload = provider._build_payload(LONG_HISTORY, [SPEC], "sys", stream=False)

    assert payload["system"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    assert payload["tools"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_provider_leaves_the_payload_untouched_when_disabled() -> None:
    provider = build_provider(CachePolicy(enabled=False))
    payload = provider._build_payload(LONG_HISTORY, [SPEC], "sys", stream=False)

    assert payload["system"] == "sys"
    assert "cache_control" not in payload["tools"][0]


def test_provider_leaves_the_payload_untouched_for_short_conversations() -> None:
    """Enabled but below break-even must behave as if off."""
    provider = build_provider(CachePolicy(enabled=True), expected_turns=1)
    payload = provider._build_payload(LONG_HISTORY, [SPEC], "sys", stream=False)
    assert payload["system"] == "sys"


def test_cached_payload_is_still_valid_anthropic_json() -> None:
    provider = build_provider(CachePolicy(enabled=True))
    payload = provider._build_payload(LONG_HISTORY, [SPEC], "sys", stream=False)
    # Must serialise and still carry a non-empty message list.
    encoded = json.dumps(payload)
    assert json.loads(encoded)["messages"]


async def test_cache_stats_accumulate_across_turns() -> None:
    """Realised usage is what decides whether the policy was worth it."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        first = calls["n"] == 1
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                # input_tokens is the *uncached* remainder; cache tokens are
                # billed and reported separately.
                "usage": {
                    "input_tokens": 1_000,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 0 if first else 9_500,
                    "cache_creation_input_tokens": 9_500 if first else 0,
                },
            },
        )

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        cache_policy=CachePolicy(enabled=True),
        expected_turns=10,
    )

    for _ in range(6):
        await provider.complete(LONG_HISTORY, [SPEC], system="sys")

    summary = provider.cache_stats.summary()
    assert summary["turns"] == 6
    assert summary["cache_write_tokens"] == 9_500
    assert summary["cache_read_tokens"] == 9_500 * 5
    # Five reads at 0.1x repay the 1.25x write several times over, leaving the
    # uncached tail to dominate the bill.
    assert summary["cost_ratio"] < 0.4
    assert summary["hit_rate"] > 0.85


async def test_usage_carries_cache_fields_to_the_caller() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await provider.complete(LONG_HISTORY, [], system="s")
    assert response.usage.cache_read_tokens == 80
    assert response.usage.cache_write_tokens == 0


async def test_streaming_reports_cache_usage_too() -> None:
    """A streamed turn must report cache hits the same way a buffered one does."""
    body = (
        'data: {"type":"message_start","message":{"usage":'
        '{"input_tokens":500,"output_tokens":0,"cache_read_input_tokens":400}}}\n\n'
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"hi"}}\n\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":3,"cache_read_input_tokens":400}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await provider.complete(LONG_HISTORY, [], system="s", stream=True)
    assert response.usage.cache_read_tokens == 400
    assert provider.cache_stats.cache_read_tokens == 400


def test_cache_policy_is_ignored_by_openai_compatible_backends() -> None:
    """OpenAI caches automatically; sending markers there would be wrong."""
    from vela.providers.factory import create_provider

    provider = create_provider(
        "openai",
        model="gpt-4o-mini",
        api_key="k",
        cache_policy=CachePolicy(enabled=True),
        expected_turns=20,
    )
    assert not hasattr(provider, "cache_policy")