"""Provider abstraction: one contract, many LLM backends.

The runner only ever sees :class:`Provider`. Everything backend-specific --
message shape, tool-call parsing, error semantics -- lives behind that boundary.

Two ideas from studying production agent runtimes shaped this module:

1. **Model output is untrusted input.** Parse defensively, and never let a
   malformed artefact reach persisted state.
2. **Not all failures are retryable.** Conflating "rate limited" with "out of
   credit" with "network blip" leads to either infinite retries or needless
   give-ups, so errors are classified explicitly.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from vela.core.contracts import ModelResponse, ToolCall, Usage


class ProviderError(RuntimeError):
    """Base class for backend failures."""

    #: Drives retry/fallback policy in the runner.
    kind: str = "unknown"


class TransientError(ProviderError):
    """A temporary failure worth retrying: timeout, 5xx, connection reset."""

    kind = "transient"


class RateLimitError(ProviderError):
    """The backend asked us to slow down. Retrying sooner is counterproductive."""

    kind = "rate_limit"

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BillingError(ProviderError):
    """The account is out of credit.

    Retrying cannot help. The caller should surface a clear message or switch
    to a different model -- treating this as transient produces a hang.
    """

    kind = "billing"


class AuthError(ProviderError):
    """The credential is missing, malformed, or rejected."""

    kind = "auth"


class ContextWindowError(ProviderError):
    """The request does not fit in the model's context window."""

    kind = "context_window"


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

_STATUS_KIND: dict[int, type[ProviderError]] = {
    401: AuthError,
    402: BillingError,
    429: RateLimitError,
    500: TransientError,
    502: TransientError,
    503: TransientError,
    504: TransientError,
}

#: Substrings that mean "out of quota" even when the status code is unhelpful.
_BILLING_HINTS = (
    "insufficient_quota",
    "insufficient credit",
    "billing",
    "payment required",
    "account is suspended",
    "exceeded your current quota",
)

_AUTH_HINTS = ("invalid api key", "incorrect api key", "unauthorized", "invalid_api_key")
_RATE_HINTS = ("rate limit", "too many requests", "rate_limit_exceeded")
_CONTEXT_HINTS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


def classify_error(status_code: int | None, body: str) -> ProviderError:
    """Map an HTTP status plus response body onto a typed error.

    Status codes are consulted first because they are authoritative; body text is
    only a fallback for backends that report billing problems as 400.
    """
    text = (body or "").lower()

    if status_code is not None:
        if status_code == 401:
            return AuthError(body or "authentication failed")
        if status_code == 402:
            return BillingError(body or "out of credit")

    if any(hint in text for hint in _BILLING_HINTS):
        return BillingError(body or "out of credit")
    if any(hint in text for hint in _AUTH_HINTS):
        return AuthError(body or "authentication failed")
    if status_code == 429 or any(hint in text for hint in _RATE_HINTS):
        return RateLimitError(body or "rate limited")
    if any(hint in text for hint in _CONTEXT_HINTS):
        return ContextWindowError(body or "context window exceeded")
    if status_code is not None and status_code in _STATUS_KIND:
        return _STATUS_KIND[status_code](body or f"HTTP {status_code}")
    return ProviderError(body or f"unexpected error (status={status_code})")


def is_retryable(error: BaseException) -> bool:
    """Whether retrying the same request could plausibly succeed."""
    if isinstance(error, (TransientError, RateLimitError)):
        return True
    # Timeouts surface as httpx errors; treat them as transient.
    name = type(error).__name__.lower()
    return "timeout" in name or "connect" in name


# --------------------------------------------------------------------------
# Tool definition helpers
# --------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """How a tool is advertised to a backend."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop structurally invalid messages before they reach a backend.

    严格的API会因单条错误消息而拒绝整个请求，因此格式错误的否则，发言权可能阻碍整个对话。这反映了防御性的
     通过生产运行时需求：空闲助手内容被填充为占位符（某些后端拒绝空内容），以及空工具结果成为明确的标记而非空字符串。
    """
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and not _has_payload(message):
            message = {**message, "content": "(no content)"}
        elif message.get("role") == "tool" and not message.get("content"):
            message = {**message, "content": "(empty result)"}
        cleaned.append(message)
    return cleaned


def _has_payload(message: dict[str, Any]) -> bool:
    if message.get("content"):
        return True
    return bool(message.get("tool_calls"))


# --------------------------------------------------------------------------
# The provider contract
# --------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Per-request generation parameters."""

    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    context_window: int = 128_000


class Provider(ABC):
    """One LLM backend."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        *,
        system: str = "",
        stream: bool = False,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> ModelResponse:
        """Run one completion. Must not raise for ordinary model failures.

        ``on_delta`` is invoked as each text fragment is parsed, *before* the
        full response exists. Streaming through a return value alone would only
        deliver the finished text, which defeats the purpose of streaming: the
        caller should be able to render tokens as they arrive.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release transport resources."""


class StreamEmitter:
    """Receives incremental text during streaming.

    Kept as a tiny protocol so the runner can stay agnostic to whether the
    backend streams at all.
    """

    #: Whether the consumer actually wants deltas. Streaming adds reassembly
    #: bookkeeping and a separate code path, so it stays opt-in: a caller that
    #: only needs the final text must not pay for incremental assembly.
    wants_stream: bool = False

    #: Set once any text has been delivered. The runner uses it to tell an
    #: incremental stream apart from a backend that ignored the callback, so a
    #: catch-up emission can be made without risking a duplicate.
    streamed_content: bool = False

    async def on_text(self, delta: str) -> None:  # pragma: no cover - default no-op
        """Handle an incremental text chunk."""

    async def on_stream_end(self) -> None:  # pragma: no cover - default no-op
        """Handle stream completion."""


def parse_tool_call_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool call's ``arguments`` payload into a dict.

    Models routinely emit near-JSON: trailing commas, single quotes, markdown
    fences. Rejecting all of those loses otherwise-usable calls, so this
    attempts a lenient repair before giving up.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _repair_json(text)
    if isinstance(parsed, dict):
        return parsed
    # A bare scalar or list is not a valid argument object.
    return {}


def _repair_json(text: str) -> Any:
    candidate = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    attempts = (candidate, _drop_trailing_commas(candidate), _close_brackets(candidate))
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return {}


def _drop_trailing_commas(text: str) -> str:
    """Remove ``,`` before ``}``/``]`` outside of string literals."""
    output: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            output.append(char)
            continue
        if char == "\\":
            escaped = True
            output.append(char)
            continue
        if char == '"':
            in_string = not in_string
            output.append(char)
            continue
        if not in_string and char == ",":
            following = text[index + 1:].lstrip()
            if following[:1] in {"}", "]"}:
                continue  # drop this comma
        output.append(char)
    return "".join(output)


def _close_brackets(text: str) -> str:
    """Append missing closing braces/brackets, ignoring brackets in strings."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()
    return text + "".join(reversed(stack))


def extract_usage(payload: dict[str, Any]) -> Usage:
    """Read token counts from a backend response, flagging absent data.

    When ``usage`` is missing the record is marked ``estimated`` even though the
    caller fills in rough numbers -- callers must be able to tell a measured
    count from a guess.
    """
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return Usage(estimated=True)

    def _int(key: str) -> int:
        value = raw.get(key)
        return value if isinstance(value, int) else 0

    def _opt(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) else None

    details = raw.get("prompt_tokens_details")
    cache_read = None
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        cache_read = details["cached_tokens"]

    return Usage(
        input_tokens=_int("prompt_tokens"),
        output_tokens=_int("completion_tokens"),
        cache_read_tokens=cache_read,
        cache_write_tokens=_opt("cache_creation_input_tokens"),
        estimated=False,
    )


def parse_openai_response(payload: dict[str, Any]) -> ModelResponse:
    """Normalise an OpenAI-shaped chat completion into :class:`ModelResponse`."""
    import time

    choices = payload.get("choices") or []
    if not choices:
        return ModelResponse(
            content="",
            finish_reason="error",
            error=payload.get("error", {}).get("message", "no choices returned"),
            usage=extract_usage(payload),
        )

    choice = choices[0]
    message = choice.get("message") or {}
    raw_reason = choice.get("finish_reason") or "stop"
    finish_reason = raw_reason if raw_reason in {
        "stop", "length", "tool_calls", "content_filter", "error"
    } else "stop"

    tool_calls: list[ToolCall] = []
    for index, item in enumerate(message.get("tool_calls") or []):
        function = item.get("function") or {}
        name = function.get("name")
        # Keep the call only if the name is usable. The runner re-validates, but
        # dropping the obvious garbage here keeps every consumer simple.
        if not isinstance(name, str) or not name.strip():
            continue
        tool_calls.append(
            ToolCall(
                id=item.get("id") or f"call_{index}",
                name=name,
                arguments=parse_tool_call_arguments(function.get("arguments", "")),
            )
        )

    return ModelResponse(
        content=message.get("content") or "",
        tool_calls=tuple(tool_calls),
        finish_reason=finish_reason,  # type: ignore[arg-type]
        usage=extract_usage(payload),
        reasoning=message.get("reasoning_content") or "",
        error=payload.get("error", {}).get("message", ""),
    )


async def noop_stream() -> AsyncIterator[str]:  # pragma: no cover - helper
    if False:  # pragma: no cover
        yield ""
