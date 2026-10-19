"""Prompt caching: trade a write premium for a read discount.

Caching is not a free win. Backends charge more to *create* a cache entry than
to read it, so caching only pays off when the same prefix is reused enough times:

    cost(n) = write_price * prefix + read_price * (n - 1) * prefix
             vs
    cost(n) = price * n * prefix

Anthropic's numbers are 1.25x for a 5-minute write, 2x for a 1-hour write, and
0.1x for a read. The 1-hour entry breaks even at three uses; the 5-minute one
needs four. Below that, turning caching on makes every request *more* expensive.

So this module does three things rather than just stamping markers:

1. Decide whether caching is worth it for the expected conversation length.
2. Place breakpoints only where the prefix is genuinely stable, since marking a
   point that changes每 turn invalidates everything before it.
3. Report realised savings, so the decision can be checked against reality
   instead of trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Relative prices against a normal input token, from Anthropic's published
#: rates. Kept here rather than hard-coded in the provider so the cost model can
#: be reasoned about and tested without touching a backend.
PRICE_INPUT = 1.0
PRICE_CACHE_WRITE_5MIN = 1.25
PRICE_CACHE_WRITE_1H = 2.0
PRICE_CACHE_READ = 0.1


class CacheTTL(str, Enum):  # noqa: UP042
    # Subclassing str keeps the value directly JSON-serialisable and comparable
    # to a plain string, which the cache_control payload requires. StrEnum would
    # be tidier but needs 3.11+, and the values must survive json.dumps either way.
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"

    @property
    def write_multiplier(self) -> float:
        return {
            CacheTTL.FIVE_MINUTES: PRICE_CACHE_WRITE_5MIN,
            CacheTTL.ONE_HOUR: PRICE_CACHE_WRITE_1H,
        }[self]

    @property
    def break_even_uses(self) -> int:
        """Smallest number of requests that costs no more than not caching.

        Caching ``n`` times costs ``write + (n-1) * read`` while the uncached
        path costs ``n``, so the first request always loses (the write premium
        is unrecovered) and later reads repay it. Solving
        ``write + (n-1)*read < n`` gives the threshold below.
        """
        write = self.write_multiplier
        read = PRICE_CACHE_READ
        uses = 1
        while write + (uses - 1) * read >= uses * PRICE_INPUT:
            uses += 1
            if uses > 1000:  # defensive; unreachable with realistic prices
                break
        return uses


@dataclass(frozen=True)
class CachePolicy:
    """How to mark a request, and whether to bother."""

    enabled: bool = False
    ttl: CacheTTL = CacheTTL.FIVE_MINUTES
    #: Cache the tool definitions too. Worth it when the tool set is stable,
    #: which for a fixed registry it is -- they only change on reconfiguration.
    cache_tools: bool = True
    #: Mark the system prompt. Always stable within a deployment.
    cache_system: bool = True
    #: Mark the start of the conversation prefix (everything up to the last
    #: couple of turns). This is the breakpoint that pays off across turns,
    #: because the recent tail changes every request.
    cache_history: bool = True

    def should_cache(self, expected_turns: int) -> bool:
        """Whether caching is expected to save money for this conversation."""
        if not self.enabled:
            return False
        return expected_turns >= self.ttl.break_even_uses

    def break_even_uses(self) -> int:
        return self.ttl.break_even_uses


@dataclass
class CacheStats:
    """Realised cache behaviour, accumulated across turns.

    A hit is only a saving when it actually displaces a full-price read. The
    running totals let a caller verify that the policy paid for the write, which
    is the only way to know whether the marker placement was useful.
    """

    turns: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    fresh_input_tokens: int = 0

    def record(self, *, read: int | None, write: int | None, input_tokens: int) -> None:
        """Fold one turn's usage into the running totals.

        ``input_tokens`` is the uncached portion only: Anthropic reports cached
        tokens in separate ``cache_read``/``cache_creation`` fields and excludes
        them from the input count. Subtracting the cache hit here would
        understate the uncached baseline and make the cost ratio look better
        than the invoice actually was.
        """
        self.turns += 1
        self.cache_read_tokens += read or 0
        self.cache_write_tokens += write or 0
        self.fresh_input_tokens += input_tokens

    @property
    def hit_rate(self) -> float:
        """Share of total input tokens served from cache.

        The denominator is every input token the model saw -- cached or not --
        since that is what a hit actually saves work on.
        """
        total = self.cache_read_tokens + self.fresh_input_tokens
        return self.cache_read_tokens / total if total else 0.0

    @property
    def effective_cost_ratio(self) -> float:
        """Actual spend as a fraction of the same usage without caching.

        Below 1.0 means caching saved money; above 1.0 means the write premium
        has not been repaid yet.

        The baseline counts every input token at full price, including the ones
        written to cache: without caching those tokens would still have been
        sent and paid for. Leaving them out would make the write look free and
        overstate the saving.
        """
        baseline_tokens = (
            self.fresh_input_tokens + self.cache_read_tokens + self.cache_write_tokens
        )
        if baseline_tokens == 0:
            return 1.0
        with_cache = (
            self.fresh_input_tokens * PRICE_INPUT
            + self.cache_read_tokens * PRICE_CACHE_READ
            + self.cache_write_tokens * PRICE_CACHE_WRITE_5MIN
        )
        return with_cache / baseline_tokens

    def summary(self) -> dict[str, float | int]:
        return {
            "turns": self.turns,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "hit_rate": round(self.hit_rate, 4),
            "cost_ratio": round(self.effective_cost_ratio, 4),
        }


@dataclass
class CachePlan:
    """Concrete instruction for where to place breakpoints."""

    mark_tools: bool = False
    mark_system: bool = False
    #: How many trailing messages to leave unmarked, since they change every turn.
    history_turns: int = 0
    mark_last_stable: bool = False

    @property
    def has_breakpoints(self) -> bool:
        return self.mark_tools or self.mark_system or self.mark_last_stable


def plan(policy: CachePolicy, *, message_count: int, expected_turns: int) -> CachePlan:
    """Decide where to place breakpoints for one request.

    Placing a breakpoint after content that changes next turn wastes the prefix
    before it, so the plan deliberately leaves the recent tail unmarked. The
    history breakpoint therefore sits ``history_turns`` back from the end.
    """
    if not policy.should_cache(expected_turns):
        return CachePlan()

    return CachePlan(
        mark_tools=policy.cache_tools,
        mark_system=policy.cache_system,
        # Two turns is enough to cover a tool result plus the follow-up that
        # depended on it, which is the pair most likely to recur verbatim.
        history_turns=2,
        # Only worth a breakpoint once there is a prefix to reuse.
        mark_last_stable=message_count > policy_history_minimum(),
    )


def policy_history_minimum() -> int:
    """Message count below which a history breakpoint cannot pay for itself."""
    return 4


def estimate_savings(
    *, prefix_tokens: int, tail_tokens: int, turns: int, ttl: CacheTTL
) -> float:
    """Project the saving for a stable prefix over ``turns`` requests.

    Useful for deciding a policy before running anything: if the projection is
    negative, the conversation is too short for caching to help.
    """
    cached_turns = max(0, turns - 1)
    if cached_turns <= 0:
        return 0.0
    uncached = turns * (prefix_tokens + tail_tokens)
    cached = (
        prefix_tokens * ttl.write_multiplier
        + cached_turns * prefix_tokens * PRICE_CACHE_READ
        + turns * tail_tokens
    )
    return uncached - cached