"""The execution loop: model -> tools -> model, until an answer exists.

This is the component that has to survive an unreliable model. A production
agent cannot assume the backend behaves: it may return empty text, truncate
mid-sentence, emit tool calls with no name, or loop forever calling tools. Each
of those gets an explicit, bounded recovery path below. Every bound exists --
retry counts, recovery segments, injection cycles -- because an unbounded
recovery against untrusted input is just a way to burn tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from vela.core.contracts import (
    ABNORMAL_FINISH_REASONS,
    ModelResponse,
    ToolCall,
    ToolResult,
    Usage,
)
from vela.providers.base import Provider, ProviderError, StreamEmitter
from vela.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# --- Recovery budgets -------------------------------------------------------
#: How many times to retry a response that came back empty.
MAX_EMPTY_RETRIES = 2
#: How many continuation segments to accept after output truncation.
MAX_LENGTH_RECOVERIES = 3
#: How many injected messages may arrive within one turn.
MAX_INJECTIONS_PER_TURN = 3
#: How many times injected content may restart the loop within one turn.
MAX_INJECTION_CYCLES = 5

DEFAULT_ERROR_MESSAGE = "The model call failed."
TRUNCATED_OUTPUT_MESSAGE = (
    "[Your previous reply was cut off by the output limit. Continue exactly where "
    "it stopped, without repeating any text and without restating the question.]"
)


class StopReason:
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    EMPTY = "empty_final_response"
    CANCELLED = "cancelled"


@dataclass
class RunSpec:
    """Everything one run needs. Immutable input to the loop."""

    system: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_iterations: int = 12
    max_tool_output: int = 8_000
    concurrent_tools: bool = True
    tool_timeout: float | None = 60.0
    finalize_on_limit: bool = True
    session_id: str = "default"


@dataclass
class RunResult:
    """Everything one run produced."""

    content: str
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage.zero)
    stop_reason: str = StopReason.COMPLETED
    error: str = ""
    iterations: int = 0
    had_injections: bool = False


class AgentRunner:
    """Owns the model-facing loop. Knows nothing about channels or transports."""

    def __init__(self, provider: Provider, tools: ToolRegistry) -> None:
        self._provider = provider
        self._tools = tools

    async def run(
        self,
        spec: RunSpec,
        *,
        emitter: StreamEmitter | None = None,
        drain_injections: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    ) -> RunResult:
        """Execute one turn and return its result.

        ``drain_injections`` is polled at each safe boundary rather than
        mid-request: appending to the message array while a request is in flight
        would mutate state the provider may still be reading.
        """
        emitter = emitter or StreamEmitter()
        messages = list(spec.messages)
        state = _RunState(
            spec=spec,
            messages=messages,
            emitter=emitter,
            drain_source=drain_injections,
        )

        try:
            for iteration in range(spec.max_iterations):
                state.iteration = iteration
                if await self._step(state):
                    return state.result(iteration + 1)
        except _Cancelled:
            return state.result(state.iteration, stop_reason=StopReason.CANCELLED)

        # The loop finished without ``break``: iteration budget exhausted.
        return await self._handle_exhausted(state)

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    async def _step(self, state: _RunState) -> bool:
        """Run one model call. Returns ``True`` when the turn is finished."""
        response = await self._request(state)

        if response.wants_tools:
            return await self._handle_tools(state, response)

        # A response can carry tool calls *and* an abnormal finish reason (output
        # truncated mid-argument). Executing partially parsed arguments is
        # unsafe, so record and drop them rather than dispatching.
        if response.tool_calls:
            logger.warning(
                "discarding %d tool call(s): finish_reason=%s",
                len(response.tool_calls),
                response.finish_reason,
            )

        content = response.content

        # An empty reply is a malfunction *only* when the model claims it
        # finished normally. Empty output after truncation or refusal is
        # expected, so those finish reasons must not trigger a retry.
        if not _has_text(content) and response.finish_reason not in ABNORMAL_FINISH_REASONS:
            return await self._handle_empty(state, response)

        if response.finish_reason == "length":
            return await self._handle_truncated(state, response, content)

        if response.is_error:
            return await self._handle_error(state, response)

        if not _has_text(content):
            return await self._handle_empty(state, response, final=True)

        state.append_assistant(content)
        # Join the accepted segments with the final reply. Without this the
        # earlier segments are silently dropped and the user sees only the tail
        # of an answer the model already spent tokens producing.
        state.final_content = (
            "".join(state.length_segments) + content
            if state.length_segments
            else content
        )
        state.stop_reason = StopReason.COMPLETED

        if await state.drain():
            state.length_segments.clear()
            return False
        return True

    # ------------------------------------------------------------------
    # Recovery paths
    # ------------------------------------------------------------------

    async def _handle_tools(self, state: _RunState, response: ModelResponse) -> bool:
        """Record the assistant turn, run tools, append results, continue."""
        await state.emitter.on_stream_end()

        state.messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [call.to_message() for call in response.tool_calls],
            }
        )

        records = await self._tools.dispatch(
            response.tool_calls,
            concurrent=state.spec.concurrent_tools,
            timeout=state.spec.tool_timeout,
        )

        for record in records:
            state.tools_used.append(record.call.name)
            result = self._clamp(record.result, state.spec.max_tool_output)
            state.messages.append(result.to_message())

        # A tool that actually ran means the model is making progress; reset the
        # budgets so recovery state from an earlier bad turn cannot suppress
        # legitimate retries later.
        state.empty_retries = 0
        state.length_segments.clear()

        if await state.drain():
            state.had_injections = True
        return False

    async def _handle_empty(self, state: _RunState, response: ModelResponse, *, final: bool = False) -> bool:
        """Recover from an empty reply: retry a bounded number of times."""
        if final:
            state.final_content = "The model returned an empty response."
            state.stop_reason = StopReason.EMPTY
            state.error = state.final_content
            state.append_assistant(state.final_content)
            return not await state.drain()

        state.empty_retries += 1
        if state.empty_retries < MAX_EMPTY_RETRIES:
            logger.warning(
                "empty response (attempt %d/%d), retrying",
                state.empty_retries,
                MAX_EMPTY_RETRIES,
            )
            await state.emitter.on_stream_end()
            if await state.drain():
                state.had_injections = True
            return False

        logger.warning("empty response after %d retries, finalizing", state.empty_retries)
        await state.emitter.on_stream_end()
        retry = await self._provider.complete(
            state.messages + [{"role": "user", "content": TRUNCATED_OUTPUT_MESSAGE}],
            [],
            system=state.spec.system,
        )
        state.merge_usage(retry.usage)
        if _has_text(retry.content) and not retry.is_error:
            state.append_assistant(retry.content)
            state.final_content = retry.content
            state.stop_reason = StopReason.COMPLETED
            return not await state.drain()

        state.final_content = "The model returned an empty response."
        state.stop_reason = StopReason.EMPTY
        state.error = state.final_content
        return not await state.drain()

    async def _handle_truncated(self, state: _RunState, response: ModelResponse, content: str) -> bool:
        """Accept one truncated segment and ask the model to continue.

        Segments accumulate instead of streaming straight to the user, then join
        at the end. Emitting partial text on every retry produces duplicated or
        visibly stuttering output.
        """
        if len(state.length_segments) < MAX_LENGTH_RECOVERIES:
            state.length_segments.append(content)
            logger.info(
                "output truncated (segment %d/%d), continuing",
                len(state.length_segments),
                MAX_LENGTH_RECOVERIES,
            )
            state.messages.append({"role": "assistant", "content": content})
            # The continuation prompt carries the text produced so far so the
            # model resumes rather than restarting its answer.
            state.messages.append({"role": "user", "content": TRUNCATED_OUTPUT_MESSAGE})
            await state.emitter.on_stream_end()
            if await state.drain():
                state.had_injections = True
            return False

        state.final_content = "".join(state.length_segments).strip() or DEFAULT_ERROR_MESSAGE
        state.stop_reason = StopReason.COMPLETED
        return True

    async def _handle_error(self, state: _RunState, response: ModelResponse) -> bool:
        """Record a model error, then honour any queued input before exiting."""
        message = response.error_message(DEFAULT_ERROR_MESSAGE)
        state.final_content = message
        state.stop_reason = StopReason.ERROR
        state.error = message
        # Persist a neutral placeholder rather than the error text: an error is
        # not something the assistant said, and future turns must not read it as
        # a prior answer.
        state.messages.append({"role": "assistant", "content": "(the model call failed)"})
        return not await state.drain()

    async def _handle_exhausted(self, state: _RunState) -> RunResult:
        """Iteration budget spent: ask for a final answer without tools.

        Hard-stopping leaves the user with whatever mid-thought text happened to
        exist. One extra tool-free request usually converts that into a usable
        summary, and removing the tool definitions makes it structurally
        impossible for the model to start new work it cannot finish.
        """
        await state.drain()

        terminal = ""
        if state.spec.finalize_on_limit:
            try:
                response = await self._provider.complete(
                    state.messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "You have reached the step limit for this task. Using "
                                "only what you already gathered, give the user a final "
                                "answer now. Do not call any tools."
                            ),
                        }
                    ],
                    [],
                    system=state.spec.system,
                )
                state.merge_usage(response.usage)
                if not response.is_error and not response.tool_calls and _has_text(response.content):
                    terminal = response.content
            except ProviderError as exc:
                logger.warning("finalization after limit failed: %s", exc)

        if not terminal:
            used = ", ".join(dict.fromkeys(state.tools_used)) or "none"
            terminal = (
                "I reached the step limit for this task before finishing. "
                f"Tools used: {used}."
            )

        state.final_content = (
            "".join(state.length_segments).rstrip() + "\n\n" + terminal.lstrip()
        ).strip() if state.length_segments else terminal
        state.stop_reason = StopReason.MAX_ITERATIONS
        state.append_assistant(terminal)
        return state.result(state.iteration + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _request(self, state: _RunState) -> ModelResponse:
        """One provider call, with streaming and retries for transient errors."""
        from vela.providers.base import is_retryable

        stream = bool(state.emitter.wants_stream)
        # A reused emitter still holds the previous turn's delivery state; clear
        # it so the catch-up check below judges only this request.
        reset = getattr(state.emitter, "reset", None)
        if callable(reset):
            reset()
        for attempt in range(3):
            try:
                response = await self._provider.complete(
                    state.messages,
                    self._tools.specs(),
                    system=state.spec.system,
                    stream=stream,
                    # Hand the emitter straight to the backend so fragments
                    # render as they arrive. Waiting for the returned response
                    # would deliver the whole answer at once, which is not
                    # streaming in any sense a user could observe.
                    on_delta=state.emitter.on_text if stream else None,
                )
            except _Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                if is_retryable(exc) and attempt < 2:
                    logger.warning("transient provider error, retry %d: %s", attempt + 1, exc)
                    continue
                return ModelResponse(
                    content="", finish_reason="error", error=f"{type(exc).__name__}: {exc}"
                )

            state.merge_usage(response.usage)
            if stream:
                # Deltas were already forwarded by the backend. Only a provider
                # that ignored the callback still needs a catch-up, otherwise
                # the answer would be rendered twice.
                if not state.emitter.streamed_content and _has_text(response.content):
                    await state.emitter.on_text(response.content)
                await state.emitter.on_stream_end()
            return response

        return ModelResponse(content="", finish_reason="error", error=DEFAULT_ERROR_MESSAGE)

    def _clamp(self, result: ToolResult, limit: int) -> ToolResult:
        """Bound tool output so one verbose call cannot exhaust the context."""
        if len(result.content) <= limit:
            return result
        dropped = len(result.content) - limit
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            content=f"{result.content[:limit]}\n\n[truncated {dropped} characters]",
            status=result.status,
        )


class _Cancelled(Exception):
    """Internal signal that the turn was cancelled."""


@dataclass
class _RunState:
    """Mutable per-turn state, grouped so the runner methods stay readable."""

    spec: RunSpec
    messages: list[dict[str, Any]]
    emitter: StreamEmitter
    #: Named ``drain_source`` rather than ``drain``: a dataclass field would
    #: shadow the ``drain()`` method below, since both live on this class.
    drain_source: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None

    iteration: int = 0
    usage: Usage = field(default_factory=Usage.zero)
    tools_used: list[str] = field(default_factory=list)
    empty_retries: int = 0
    length_segments: list[str] = field(default_factory=list)
    final_content: str = ""
    stop_reason: str = StopReason.COMPLETED
    error: str = ""
    had_injections: bool = False

    async def drain(self) -> bool:
        """Pull queued user input, if any, into the transcript.

        Called only at iteration boundaries. Returns whether new input arrived,
        meaning the loop should continue rather than finish.
        """
        if self.drain_source is None:
            return False
        pending = await self.drain_source()
        if not pending:
            return False
        # Accept a single message or a batch: callers naturally have one or the
        # other depending on their queue, and normalising here avoids a
        # shape mismatch that would surface as an opaque slicing error.
        injected = [pending] if isinstance(pending, dict) else list(pending)
        capped = injected[:MAX_INJECTIONS_PER_TURN]
        if len(injected) > MAX_INJECTIONS_PER_TURN:
            logger.warning(
                "dropped %d injected message(s) beyond the per-turn cap",
                len(injected) - MAX_INJECTIONS_PER_TURN,
            )
        self.messages.extend(capped)
        return True

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def merge_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def result(self, iterations: int, *, stop_reason: str | None = None) -> RunResult:
        return RunResult(
            content=self.final_content,
            messages=self.messages,
            tools_used=list(self.tools_used),
            usage=self.usage,
            stop_reason=stop_reason or self.stop_reason,
            error=self.error,
            iterations=iterations,
            had_injections=self.had_injections,
        )


class StreamingEmitter(StreamEmitter):
    """Pushes incremental text to a callback as fragments arrive."""

    def __init__(self, on_text: Callable[[str], Any] | None = None) -> None:
        self._on_text = on_text
        self.buffer = ""
        self.wants_stream = on_text is not None
        #: Trackers for this turn. Reset by the runner before each request so a
        #: reused emitter does not carry stale state into the next turn.
        self.streamed_content = False
        self.ended = False

    def reset(self) -> None:
        """Clear per-turn delivery state before a new request."""
        self.streamed_content = False
        self.ended = False

    async def on_text(self, delta: str) -> None:
        self.buffer += delta
        self.streamed_content = True
        if self._on_text is not None:
            self._on_text(delta)

    async def on_stream_end(self) -> None:
        self.ended = True


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())
