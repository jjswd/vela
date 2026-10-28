"""Reproduce the cache numbers quoted in the README.

The point of this script is to be honest about what produced them. The backend
here is a *simulation* of Anthropic's caching behaviour, not the real API:
it decides a hit by comparing a fingerprint of the cacheable prefix against
the previous request, and reports the same usage fields the real API reports.

That distinction matters. What this script measures is the cost model and the
breakpoint placement -- whether the code marks the right prefixes, and whether
the arithmetic in ``CacheStats`` is right. It does not measure what Anthropic
actually charges. A real-API run would replace the simulated backend and
leave everything else unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from vela.providers.anthropic import AnthropicProvider
from vela.providers.base import GenerationConfig, ToolSpec
from vela.providers.caching import CachePolicy, CacheTTL

#: Shape of the simulated session: a large stable prefix and a small tail
#: that changes every turn.
PREFIX_TOKENS = 9_000
TAIL_TOKENS = 1_000
TURNS = 10

#: Anthropic's published multipliers, relative to a normal input token.
WRITE_5M = 1.25
READ = 0.1


def simulate_backend(state: dict):
    """A mock that mimics Anthropic's prefix caching.

    A request is a cache hit when the cacheable prefix -- the system prompt
    plus the tool definitions -- is byte-identical to the previous one. That
    is the real rule, simplified to something a mock can express.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        fingerprint_source = json.dumps(
            {
                "system": body.get("system"),
                "tools": body.get("tools"),
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()

        if state.get("fingerprint") == fingerprint:
            # A hit: the prefix is served from cache, and only the tokens
            # after the last breakpoint are fresh.
            read = PREFIX_TOKENS
            fresh = TAIL_TOKENS
            write = 0
        else:
            # A miss that writes the cache. Anthropic reports the written
            # tokens separately: input_tokens counts only what came after the
            # breakpoint, so the prefix is not counted twice.
            read = 0
            fresh = TAIL_TOKENS
            write = PREFIX_TOKENS
            state["fingerprint"] = fingerprint

        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    # The uncached remainder. Anthropic reports cached tokens
                    # separately and excludes them from input_tokens.
                    "input_tokens": fresh,
                    "output_tokens": 60,
                    "cache_read_input_tokens": read,
                    "cache_creation_input_tokens": write,
                },
            },
        )

    return handler


async def run(cache_enabled: bool, ttl: CacheTTL) -> dict:
    state: dict = {}
    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(simulate_backend(state))
        ),
        cache_policy=(
            CachePolicy(enabled=True, ttl=ttl) if cache_enabled else CachePolicy(enabled=False)
        ),
        expected_turns=TURNS,
    )

    # A stable system prompt and a fixed tool set: the cacheable prefix.
    system = "You are a careful assistant. " * 200
    # Tool definitions are part of the cacheable prefix, so they are sent on
    # every request -- that is what makes them worth caching.
    tools = [
        ToolSpec(
            name="read_file",
            description="read a file from the workspace",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
    ]

    for _ in range(TURNS):
        await provider.complete(
            [{"role": "user", "content": "a question"}], tools, system=system
        )
    return provider.cache_stats.summary()


def compute_expected(prefix: int, tail: int, turns: int) -> dict:
    """The same arithmetic, derived independently of the implementation.

    The denominators follow Anthropic's definition: total input is
    ``cache_read + cache_creation + input_tokens``. A write is not a hit, so
    the written tokens sit in the denominator only.
    """
    writes = prefix
    reads = prefix * (turns - 1)
    fresh = tail * turns

    total = writes + reads + fresh
    with_cache = writes * WRITE_5M + reads * READ + fresh
    return {
        "hit_rate": reads / total,
        "cost_ratio": with_cache / total,
    }


async def benchmark() -> dict:
    """Run the simulated session and return the headline figures."""
    on = await run(True, CacheTTL.FIVE_MINUTES)
    expected = compute_expected(PREFIX_TOKENS, TAIL_TOKENS, TURNS)
    return {
        "hit_rate": on["hit_rate"],
        "cost_ratio": on["cost_ratio"],
        "read_tokens": on["cache_read_tokens"],
        "write_tokens": on["cache_write_tokens"],
        "expected_hit_rate": expected["hit_rate"],
        "expected_cost_ratio": expected["cost_ratio"],
        "matches_hand_calculation": (
            abs(expected["hit_rate"] - on["hit_rate"]) < 0.001
            and abs(expected["cost_ratio"] - on["cost_ratio"]) < 0.001
        ),
    }


async def main() -> int:
    print("Simulated backend: a stable 9,000-token prefix, a 1,000-token tail, 10 turns.\n")

    off = await run(False, CacheTTL.FIVE_MINUTES)
    on = await run(True, CacheTTL.FIVE_MINUTES)

    print(f"{'':22}{'caching off':>16}{'caching on':>16}")
    print(f"{'turns':22}{off['turns']:>16}{on['turns']:>16}")
    print(f"{'cache read tokens':22}{off['cache_read_tokens']:>16}{on['cache_read_tokens']:>16}")
    print(f"{'cache write tokens':22}{off['cache_write_tokens']:>16}{on['cache_write_tokens']:>16}")
    print(f"{'hit rate':22}{off['hit_rate']:>15.1%}{on['hit_rate']:>16.1%}")
    print(f"{'cost ratio':22}{off['cost_ratio']:>16.3f}{on['cost_ratio']:>16.3f}")

    print(f"\ninput token cost reduction: {(1 - on['cost_ratio']) * 100:.1f}%")

    expected = compute_expected(PREFIX_TOKENS, TAIL_TOKENS, TURNS)
    print("\nindependent arithmetic:")
    print(f"  hit rate    {expected['hit_rate']:.1%}")
    print(f"  cost ratio  {expected['cost_ratio']:.3f}")

    matched = (
        abs(expected["hit_rate"] - on["hit_rate"]) < 0.001
        and abs(expected["cost_ratio"] - on["cost_ratio"]) < 0.001
    )
    print(f"\nimplementation matches the hand calculation: {matched}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
