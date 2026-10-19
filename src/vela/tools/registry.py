"""Tool registry and dispatch.

The registry is the single place that knows which tools exist. Dispatch is
strict about unknown names: silently ignoring a call would leave the model
believing the action succeeded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vela.core.contracts import ToolCall, ToolResult
from vela.providers.base import ToolSpec
from vela.tools.base import Tool


@dataclass(frozen=True)
class DispatchRecord:
    """What actually happened for one dispatched call."""

    call: ToolCall
    result: ToolResult


class ToolRegistry:
    """Holds tools and runs them, optionally in parallel.

    A tool may declare itself blocking, meaning its ``run`` performs synchronous
    IO such as walking a directory tree. Those are executed on a worker thread so
    a slow tool cannot stall the event loop and every other in-flight request
    with it. A tool that blocks without declaring it would be a latent outage,
    which is why the flag exists rather than dispatching everything off-thread.
    """

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache: list[ToolSpec] | None = None
        # Shared across instances is fine: the work is IO-bound, so the pool
        # scales on waiting rather than on CPU.
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="vela-tool"
        )
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cache = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cache = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Advertised tool schemas, cached because they are request-invariant."""
        if self._cache is None:
            self._cache = [self._tools[name].spec() for name in self.names]
        return list(self._cache)

    async def dispatch(
        self,
        calls: Sequence[ToolCall],
        *,
        concurrent: bool = True,
        timeout: float | None = 60.0,
    ) -> list[DispatchRecord]:
        """Run every call, always returning one record per input call.

        Order is preserved regardless of concurrency: models emit tool calls
        whose results map back positionally via ``tool_call_id``, so reordering
        would scramble the transcript.
        """
        if not calls:
            return []
        if concurrent and len(calls) > 1:
            results = await asyncio.gather(
                *(self._run_one(call, timeout) for call in calls),
                return_exceptions=True,
            )
        else:
            results = [await self._run_one(call, timeout) for call in calls]

        records: list[DispatchRecord] = []
        # strict=True: a length mismatch would mean a tool result was dropped,
        # leaving an unmatched tool_call id in the transcript. Fail loudly
        # instead of silently zipping the shorter list.
        for call, outcome in zip(calls, results, strict=True):
            if isinstance(outcome, BaseException):
                # A timeout or an internal crash still owes the model a tool
                # message; omitting it would leave an unmatched tool_call id.
                outcome = ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=f"Tool execution failed: {type(outcome).__name__}: {outcome}",
                    status="error",
                )
            records.append(DispatchRecord(call=call, result=outcome))
        return records

    async def _run_one(self, call: ToolCall, timeout: float | None) -> ToolResult:
        if not call.is_valid():
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content="Tool call has no usable name.",
                status="error",
            )

        tool = self._tools.get(call.name)
        if tool is not None and getattr(tool, "blocking", False):
            return await self._run_blocking(tool, call, timeout)

        if tool is None:
            suggestion = self._suggest(call.name)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            available = ", ".join(self.names)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=(
                    f"Unknown tool '{call.name}'.{hint} Available tools: {available}"
                ),
                status="error",
            )

        payload = dict(call.arguments)
        coro = tool.execute(payload, call_id=call.id)
        try:
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except TimeoutError:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Tool '{call.name}' exceeded {timeout:.0f}s and was cancelled.",
                status="error",
            )
        except Exception as exc:  # noqa: BLE001 - boundary
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"{type(exc).__name__}: {exc}",
                status="error",
            )

    async def _run_blocking(self, tool: Tool, call: ToolCall, timeout: float | None) -> ToolResult:
        """Execute a synchronous tool on a worker thread.

        The validation-and-run sequence that :meth:`Tool.execute` performs is
        reproduced here rather than awaited inside the thread: ``execute`` is a
        coroutine, and awaiting one from a worker thread yields a coroutine that
        never runs. Tool ``run`` implementations are synchronous by contract for
        exactly this reason.
        """
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, self._execute_blocking_sync, tool, call)
            if timeout is not None:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        except TimeoutError:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Tool '{call.name}' exceeded {timeout:.0f}s and was cancelled.",
                status="error",
            )
        except Exception as exc:  # noqa: BLE001 - boundary
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"{type(exc).__name__}: {exc}",
                status="error",
            )

    @staticmethod
    def _execute_blocking_sync(tool: Tool, call: ToolCall) -> ToolResult:
        """Execute a blocking tool to completion, from a worker thread.

        Tools are coroutines by contract, but a blocking one cannot be awaited
        on the shared loop without stalling it. So the coroutine runs on a
        private loop owned by the worker thread. Awaiting it directly from the
        thread would yield a coroutine that never executes, and calling the
        underlying ``run`` would bypass validation and error wrapping.
        """
        import asyncio as _asyncio

        from vela.tools.base import validate_arguments

        errors = validate_arguments(tool.parameters, call.arguments)
        if errors:
            return ToolResult(
                call_id=call.id,
                name=tool.name,
                content=f"Invalid parameters for '{tool.name}': {'; '.join(errors)}",
                status="error",
            )

        private_loop = _asyncio.new_event_loop()
        try:
            content = private_loop.run_until_complete(tool.run(call.arguments))
        except Exception as exc:  # noqa: BLE001 - boundary
            return ToolResult(
                call_id=call.id,
                name=tool.name,
                content=f"{type(exc).__name__}: {exc}",
                status="error",
            )
        finally:
            private_loop.close()

        # ``run`` yields the model-facing text; the envelope is added here so a
        # blocking tool's result is shaped exactly like every other tool's.
        return ToolResult(
            call_id=call.id,
            name=tool.name,
            content=content if isinstance(content, str) else str(content),
        )

    def shutdown(self) -> None:
        """Release the worker threads. Call when the registry is discarded."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _suggest(self, name: str) -> str | None:
        """Offer a correction when a tool name looks like a typo.

        Models misspell, mis-case, or mis-separate tool names often enough that a
        plain unknown-name error wastes a whole turn. Matching is tolerant of
        case and separators, and one edit of distance, but returns a suggestion
        only when it is unambiguous -- guessing between two candidates would be
        worse than saying nothing.
        """
        target = _normalize(name)
        if not target:
            return None

        exact = [known for known in self._tools if _normalize(known) == target]
        if len(exact) == 1:
            return exact[0]

        close = [known for known in self._tools if _within_one_edit(target, _normalize(known))]
        return close[0] if len(close) == 1 else None


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _within_one_edit(left: str, right: str) -> bool:
    """Whether two names differ by at most one insertion, deletion, or swap."""
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) <= 1

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    # The shorter string is a prefix of the longer one except at a single
    # position, which is exactly what a one-character insertion looks like.
    for index in range(len(shorter)):
        if longer[index] != shorter[index]:
            return longer[index + 1:] == shorter[index:]
    # Every compared position matched, so the difference is a trailing
    # character in the longer string.
    return True


def load_builtin_tools(workspace: str) -> list[Tool]:
    """Construct the default tool set bound to a workspace."""
    from vela.tools.builtin import builtin_tools

    return builtin_tools(workspace)
