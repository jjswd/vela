"""Prompt caching for the Anthropic Messages API.

Anthropic caches an exact byte prefix of the request. Placing a
``cache_control`` breakpoint marks where that prefix may end; everything up to
and including that block is cached, and the next request replays it at a
discount if the prefix matches.

Two rules drive the placement logic here:

* **Breakpoints on volatile content waste everything before them.** Marking the
  last message -- which changes every turn -- would invalidate the cache each
  time. The useful breakpoint is the end of the stable prefix: tools, system,
  and history old enough to have stopped changing.
* **Only four breakpoints are allowed.** Exceeding the limit is a 400, so the
  plan is built centrally and capped rather than set ad hoc.
"""

from __future__ import annotations

from typing import Any

from vela.providers.caching import CachePolicy, plan

#: Anthropic rejects requests with more than this many cache breakpoints.
MAX_BREAKPOINTS = 4

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


def _with_cache_control(block: dict[str, Any], ttl: str) -> dict[str, Any]:
    """Attach a cache breakpoint to a content block or tool definition."""
    marked = dict(block)
    marked["cache_control"] = {"type": "ephemeral"}
    if ttl != "5m":
        marked["cache_control"]["ttl"] = ttl
    return marked


def mark_tools(tools: list[dict[str, Any]], *, ttl: str = "5m") -> list[dict[str, Any]]:
    """Mark the final tool definition.

    The tool set is constant for a given registry, so it forms a large, fully
    stable prefix -- usually the biggest single win available.
    """
    if not tools:
        return tools
    marked = [dict(tool) for tool in tools]
    marked[-1] = _with_cache_control(marked[-1], ttl)
    return marked


def mark_system(system: str, *, ttl: str = "5m") -> list[dict[str, Any]] | None:
    """Render the system prompt as blocks with a breakpoint.

    A bare string cannot carry a breakpoint; caching requires the block form.
    """
    if not system.strip():
        return None
    return [_with_cache_control({"type": "text", "text": system}, ttl)]


def mark_history(
    messages: list[dict[str, Any]],
    *,
    turns_from_end: int = 2,
    ttl: str = "5m",
) -> list[dict[str, Any]]:
    """Mark the end of the stable history prefix.

    Walks back from the end, skipping the turns that changed most recently, and
    marks the message just before them. Returns the list unchanged when every
    message is too recent to cache safely.
    """
    if not messages or turns_from_end <= 0:
        return messages

    target_index = len(messages) - turns_from_end - 1
    if target_index < 0:
        return messages

    marked = [dict(message) for message in messages]
    candidate = marked[target_index]

    content = candidate.get("content")
    if isinstance(content, str):
        candidate["content"] = [
            _with_cache_control({"type": "text", "text": content}, ttl)
        ]
    elif isinstance(content, list) and content:
        blocks = [dict(block) for block in content]
        blocks[-1] = _with_cache_control(blocks[-1], ttl)
        candidate["content"] = blocks
    else:
        return messages

    return marked


def apply_cache_markers(
    *,
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    policy: CachePolicy,
    expected_turns: int,
) -> dict[str, Any]:
    """Insert breakpoints into a payload according to the policy.

    Returns a new payload; the caller keeps ownership of the original. Breaks
    down rather than raising when caching was requested but cannot be applied --
    a missing cache is slower, while a rejected request fails outright.
    """
    resolved = plan(
        policy, message_count=len(messages), expected_turns=expected_turns
    )
    if not resolved.has_breakpoints:
        return payload

    result = dict(payload)
    used = 0

    if resolved.mark_system and isinstance(result.get("system"), str):
        marked_system = mark_system(result["system"], ttl=policy.ttl.value)
        if marked_system is not None:
            result["system"] = marked_system
            used += 1

    if resolved.mark_tools and result.get("tools"):
        result["tools"] = mark_tools(result["tools"], ttl=policy.ttl.value)
        used += 1

    if resolved.mark_last_stable and used < MAX_BREAKPOINTS:
        result["messages"] = mark_history(
            result["messages"],
            turns_from_end=max(1, resolved.history_turns),
            ttl=policy.ttl.value,
        )

    return result