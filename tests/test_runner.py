"""Tests for the runner's recovery paths.

Every recovery branch exists because model output is untrusted. These tests
lock in the guarantees that justify each branch, so a later refactor cannot
quietly remove a safeguard.
"""

from __future__ import annotations

from typing import Any

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.core.runner import (
    AgentRunner,
    RunSpec,
    StopReason,
    StreamingEmitter,
)
from vela.providers.base import GenerationConfig, Provider
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


class ScriptedProvider(Provider):
    """Returns a canned response per call, recording what it was asked."""

    def __init__(self, script: list[ModelResponse], finalizer: ModelResponse | None = None) -> None:
        super().__init__(GenerationConfig(model="test"))
        self.script = script
        # Response for the extra tool-free request the runner makes after the
        # iteration budget is spent. Keeping it explicit avoids the script being
        # consumed by that unrelated bookkeeping call.
        self.finalizer = finalizer
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, messages, tools, *, system="", stream=False, on_delta=None
    ) -> ModelResponse:
        self.calls.append({"messages": list(messages), "tools": tools, "system": system})
        # A tool-free request is the post-budget finalization: the runner strips
        # tool definitions there so the model cannot start new work. Recognising
        # it by shape keeps the scripted turns independent of how many extra
        # bookkeeping requests the runner makes.
        if self.finalizer is not None and not tools:
            response, self.finalizer = self.finalizer, None
            return response
        if not self.script:
            return ModelResponse(content="script exhausted", finish_reason="stop")
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


    async def aclose(self) -> None:
        return None


class ToolStub(Tool):
    """Base for the tiny tools these tests register."""


class Echo(ToolStub):
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return f"echo:{arguments.get('text', '')}"


class Boom(ToolStub):
    name = "boom"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict) -> str:
        raise RuntimeError("tool exploded")


def registry(*tools: Any) -> ToolRegistry:
    return ToolRegistry(list(tools))


async def test_returns_final_answer_without_tools() -> None:
    provider = ScriptedProvider([ModelResponse(content="done", finish_reason="stop")])
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.content == "done"
    assert result.stop_reason == StopReason.COMPLETED
    assert result.usage.total_tokens == 0


async def test_executes_tool_then_answers() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            ModelResponse(content="all good", finish_reason="stop", usage=Usage(input_tokens=8, output_tokens=2)),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))

    assert result.content == "all good"
    assert result.tools_used == ["echo"]
    assert result.iterations == 2
    # Usage accumulates across both requests.
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 7


async def test_tool_error_does_not_abort_the_turn() -> None:
    """A crashing tool must return an error result, not kill the conversation."""
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="boom", arguments={}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="recovered", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Boom()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}]))

    assert result.content == "recovered"
    assert result.stop_reason == StopReason.COMPLETED
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert "tool exploded" in tool_messages[0]["content"]


async def test_unknown_tool_name_returns_error_to_model() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="nope", arguments={}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="ok", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}]))
    tool_message = [m for m in result.messages if m.get("role") == "tool"][0]
    assert "Unknown tool" in tool_message["content"]


async def test_empty_response_is_retried_then_answered() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(content="", finish_reason="stop"),
            ModelResponse(content="recovered", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.content == "recovered"
    assert result.iterations == 2


async def test_persistently_empty_response_gives_clear_message() -> None:
    provider = ScriptedProvider([ModelResponse(content="   ", finish_reason="stop")])
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.stop_reason == StopReason.EMPTY
    assert "empty" in result.content.lower()


async def test_truncated_output_is_continued_not_duplicated() -> None:
    """Truncation must resume the answer, and segments join without repeats."""
    provider = ScriptedProvider(
        [
            ModelResponse(content="The answer is", finish_reason="length"),
            ModelResponse(content=" forty-two.", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))

    assert result.content == "The answer is forty-two."
    assert result.content.count("The answer is") == 1
    # The continuation prompt must reference the partial text.
    assert any("cut off" in str(m.get("content", "")) for m in result.messages)


async def test_empty_content_after_truncation_is_not_retried() -> None:
    """Empty output caused by truncation is expected, not a malfunction."""
    provider = ScriptedProvider(
        [
            ModelResponse(content="partial", finish_reason="length"),
            ModelResponse(content="", finish_reason="length"),
            ModelResponse(content="partial finished", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.stop_reason == StopReason.COMPLETED


async def test_tool_calls_with_abnormal_finish_reason_are_not_executed() -> None:
    """Arguments truncated mid-JSON must never be dispatched."""
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "half"}),),
                finish_reason="length",
            ),
            ModelResponse(content="final", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.tools_used == []


async def test_iteration_limit_triggers_tool_free_finalization() -> None:
    """Exhausting the budget yields a summary, not a hard stop."""
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": str(i)}),),
                finish_reason="tool_calls",
            )
            for i in range(10)
        ],
        finalizer=ModelResponse(content="here is the summary", finish_reason="stop"),
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}], max_iterations=3))

    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert "here is the summary" in result.content
    # The finalization request must not offer tools, so no new work can start.
    assert provider.calls[-1]["tools"] == []


async def test_finalization_failure_falls_back_to_local_message() -> None:
    """If finalization fails, the user still gets an explanation."""
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            )
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "go"}], max_iterations=2))
    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert "step limit" in result.content


async def test_model_error_is_reported_with_placeholder_history() -> None:
    """An error must not be persisted as if the assistant had said it."""
    provider = ScriptedProvider(
        [ModelResponse(content="upstream exploded", finish_reason="error", error="500 boom")]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))

    assert result.stop_reason == StopReason.ERROR
    assert "500 boom" in result.content
    assistant_messages = [m for m in result.messages if m.get("role") == "assistant"]
    assert "500 boom" not in assistant_messages[-1]["content"]
    assert "failed" in assistant_messages[-1]["content"]


async def test_injected_message_is_picked_up_mid_turn() -> None:
    """Input arriving during a turn joins the transcript instead of being lost."""
    queued = [{"role": "user", "content": "actually use blue"}]

    async def drain():
        return queued.pop(0) if queued else []

    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="switched to blue", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Echo()))
    result = await runner.run(
        RunSpec(messages=[{"role": "user", "content": "make a logo"}]), drain_injections=drain
    )
    assert result.had_injections
    assert any("blue" in str(m.get("content")) for m in result.messages)


async def test_tool_output_is_clamped() -> None:
    """A verbose tool result cannot blow the context budget."""

    class Chatty(ToolStub):
        name = "chatty"
        description = "verbose"

        async def run(self, arguments: dict) -> str:
            return "x" * 5000

    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="chatty", arguments={}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="ok", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Chatty()))
    result = await runner.run(
        RunSpec(messages=[{"role": "user", "content": "go"}], max_tool_output=100)
    )
    tool_message = [m for m in result.messages if m.get("role") == "tool"][0]
    assert "truncated" in tool_message["content"]
    assert len(tool_message["content"]) < 200


async def test_concurrent_tools_preserve_order() -> None:
    """Tool results must stay aligned with their call ids even when parallel."""
    import asyncio

    class Slow(ToolStub):
        name = "slow"
        description = "variable delay"

        async def run(self, arguments: dict) -> str:
            await asyncio.sleep(0.01 * (3 - int(arguments["n"])))
            return str(arguments["n"])

    provider = ScriptedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(id="a", name="slow", arguments={"n": 1}),
                    ToolCall(id="b", name="slow", arguments={"n": 2}),
                    ToolCall(id="c", name="slow", arguments={"n": 3}),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="ok", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, registry(Slow()))
    result = await runner.run(
        RunSpec(messages=[{"role": "user", "content": "go"}], concurrent_tools=True)
    )
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b", "c"]


async def test_streaming_emitter_receives_text() -> None:
    chunks: list[str] = []
    provider = ScriptedProvider([ModelResponse(content="streamed answer", finish_reason="stop")])
    runner = AgentRunner(provider, registry(Echo()))
    emitter = StreamingEmitter(on_text=chunks.append)

    result = await runner.run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    assert result.content == "streamed answer"
