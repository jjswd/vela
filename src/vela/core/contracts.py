"""Core data contracts shared across the runtime.

Every layer (loop, runner, providers, tools) speaks in these types. Keeping them
in one dependency-free module avoids import cycles and makes the boundaries
between layers explicit: nothing here may import from a higher layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------
# Tool contracts
# --------------------------------------------------------------------------

#: How a tool invocation ended. ``error`` means the tool ran and failed in a way
#: the model can potentially recover from; the message goes back into context.
ToolStatus = Literal["ok", "error"]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Whether this call is executable.

        A call with a missing or non-string name cannot be dispatched. Persisting
        such a call into the transcript would replay it on every later turn and
        permanently break the conversation, so callers must validate *before*
        appending to history. See ``AgentRunner._drop_malformed_tool_calls``.
        """
        return isinstance(self.name, str) and bool(self.name.strip())

    def to_message(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible assistant tool_call fragment."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": _dumps(self.arguments),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing one tool call."""

    call_id: str
    name: str
    content: str
    status: ToolStatus = "ok"

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def to_message(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible ``role="tool"`` message.

        ``call_id`` must match the originating tool call exactly; providers reject
        the whole request when an id is unmatched.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.content,
        }


# --------------------------------------------------------------------------
# LLM response contracts
# --------------------------------------------------------------------------

FinishReason = Literal["stop", "length", "tool_calls", "error", "refusal"]

#: Finish reasons that represent an abnormal termination. An empty ``content``
#: with one of these reasons is expected behaviour, not a model malfunction, so
#: the empty-response retry path must not engage.
ABNORMAL_FINISH_REASONS: frozenset[str] = frozenset(
    {"error", "length", "refusal", "content_filter"}
)


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single request.

    ``reported`` distinguishes a provider-confirmed count from a local estimate.
    Some OpenAI-compatible endpoints omit usage entirely; silently treating an
    estimate as fact would make cost reporting untrustworthy, so the source is
    carried alongside the numbers.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    #: Defaults to ``True``: an empty record carries no evidence of measurement,
    #: so treating it as provider-confirmed would overstate how much we know.
    estimated: bool = True

    @classmethod
    def zero(cls) -> Usage:
        return cls()

    def __add__(self, other: Usage) -> Usage:
        """Sum two usage records, preserving the missing-vs-zero distinction.

        ``None`` means "the provider did not report this metric"; ``0`` means
        "reported as zero". Collapsing them would turn an absent metric into a
        misleading number, so addition propagates ``None`` unless one side has a
        real value.
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=_sum_optional(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_optional(self.cache_write_tokens, other.cache_write_tokens),
            # The sum is an estimate if *either* side was.
            estimated=self.estimated or other.estimated,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


@dataclass(frozen=True)
class ModelResponse:
    """One model reply, normalised across provider shapes."""

    #: Defaults to empty so a streaming chunk can be built up incrementally
    #: without having to name every field.
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = "stop"
    usage: Usage = field(default_factory=Usage.zero)
    reasoning: str = ""
    error: str = ""

    @property
    def wants_tools(self) -> bool:
        """Whether the runner should dispatch tools for this response.

        A response can carry tool calls *and* an abnormal finish reason (for
        example ``length`` mid-argument). Executing a partially streamed
        argument list is unsafe, so the caller checks this instead of merely
        testing ``tool_calls``.
        """
        if not self.tool_calls:
            return False
        return self.finish_reason not in ABNORMAL_FINISH_REASONS

    @property
    def is_error(self) -> bool:
        return self.finish_reason == "error"

    def error_message(self, default: str) -> str:
        """Pick the most specific description of a failed request.

        ``error`` wins over ``content``: when a backend fails it often echoes
        the request in ``content`` alongside the real reason, so preferring
        ``content`` would surface an unhelpful fragment to the user.
        """
        return self.error or self.content or default


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
