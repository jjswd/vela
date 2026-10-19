"""Runner-level streaming tests.

The provider tests prove fragments are reassembled correctly. These prove the
runtime actually *uses* that streaming: that text reaches the caller
incrementally, that a stream is not declared finished while tool work is still
to come, and that a caller who does not want deltas never pays for them.
"""

from __future__ import annotations

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.core.loop import AgentLoop, SessionStore
from vela.core.runner import AgentRunner, RunSpec, StopReason, StreamingEmitter
from vela.providers.base import GenerationConfig, Provider
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


class RecordingEmitter(StreamingEmitter):
    """Captures the callback order so interleaving can be asserted.

    Overrides the hooks to record them, but delegates to the base class so
    buffer and streamed_content stay consistent with what was actually
    delivered -- skipping the super call would make delivery look like it never
    happened.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        super().__init__(on_text=None)
        # This subclass implements on_text itself, so the base class cannot infer
        # opt-in from a callback argument. Enable streaming explicitly.
        self.wants_stream = True

    async def on_text(self, delta: str) -> None:
        await super().on_text(delta)
        self.events.append(f"text:{delta}")

    async def on_stream_end(self) -> None:
        await super().on_stream_end()
        self.events.append("end")


class Echo(Tool):
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


class NotedProvider(Provider):
    """Records whether streaming was requested, and streams like a real backend.

    A faithful double must honour the ``on_delta`` callback: the runtime's whole
    point of passing it is that fragments reach the caller before the response is
    assembled, so a double that ignores it would test nothing.
    """

    def __init__(self, script: list[ModelResponse]) -> None:
        super().__init__(GenerationConfig(model="test"))
        self.script = list(script)
        self.stream_flags: list[bool] = []
        #: Fragments per response; the text is delivered in these pieces.
        self.granularity = 3

    async def complete(
        self, messages, tools, *, system="", stream=False, on_delta=None
    ) -> ModelResponse:
        self.stream_flags.append(stream)
        response = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if stream and on_delta is not None and response.content:
            for piece in _split(response.content, self.granularity):
                await on_delta(piece)
        return response

    async def aclose(self) -> None:
        return None


def _split(text: str, pieces: int) -> list[str]:
    if not text or pieces <= 0:
        return [text] if text else []
    size = max(1, len(text) // pieces)
    return [text[index:index + size] for index in range(0, len(text), size)]


def registry() -> ToolRegistry:
    return ToolRegistry([Echo()])


async def test_text_reaches_the_emitter_when_streaming() -> None:
    provider = NotedProvider([ModelResponse(content="hello there", finish_reason="stop")])
    emitter = RecordingEmitter()
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    # Delivered in fragments, not as one completed blob: that is the whole
    # point of streaming.
    texts = [event for event in emitter.events if event.startswith("text:")]
    assert len(texts) > 1
    assert "".join(event[len("text:"):] for event in texts) == "hello there"
    assert result.content == "hello there"
    assert provider.stream_flags == [True]


async def test_streaming_is_not_requested_without_an_emitter() -> None:
    """A caller wanting only the final answer must not trigger the streaming path."""
    provider = NotedProvider([ModelResponse(content="done", finish_reason="stop")])
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}])
    )
    assert provider.stream_flags == [False]
    assert result.content == "done"


async def test_a_default_emitter_never_streams() -> None:
    from vela.providers.base import StreamEmitter

    assert StreamEmitter().wants_stream is False


async def test_stream_is_not_closed_before_tool_work() -> None:
    """Closing the stream early would make a UI commit a partial answer.

    The model calls a tool on the first turn, so the stream must still be open
    when that happens; only the turn's real end may close it.
    """
    provider = NotedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="final answer", finish_reason="stop"),
        ]
    )
    emitter = RecordingEmitter()
    await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )

    ends = [index for index, event in enumerate(emitter.events) if event == "end"]
    texts = [index for index, event in enumerate(emitter.events) if event.startswith("text:")]
    # The closing event must come after the final text, never before it.
    assert ends, "the stream should be closed at least once"
    assert max(texts) < max(ends)


async def test_emitter_receives_nothing_for_a_tool_only_turn() -> None:
    """The tool-only round produces no text; only the final round does.

    Emitting something during tool execution would make a UI show a message
    card before any answer exists.
    """
    provider = NotedProvider(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="done", finish_reason="stop"),
        ]
    )
    emitter = RecordingEmitter()
    await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    # Text arrives only for the final round, and the fragments reassemble into
    # exactly that answer -- nothing leaked from the tool-only round.
    texts = [event[len("text:"):] for event in emitter.events if event.startswith("text:")]
    assert texts
    assert "".join(texts) == "done"


async def test_streaming_survives_a_multi_segment_recovery() -> None:
    """Recovering from truncation must keep the stream usable."""
    provider = NotedProvider(
        [
            ModelResponse(content="first half ", finish_reason="length"),
            ModelResponse(content="second half", finish_reason="stop"),
        ]
    )
    emitter = RecordingEmitter()
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    assert result.content == "first half second half"
    assert emitter.events.count("end") >= 2  # closed once per recovery segment


async def test_loop_forwards_the_emitter(tmp_path) -> None:
    """The orchestration layer must pass the emitter through untouched."""
    provider = NotedProvider([ModelResponse(content="loop answer", finish_reason="stop")])
    emitter = RecordingEmitter()
    loop = AgentLoop(
        provider,
        registry(),
        SessionStore(tmp_path),
        workspace=str(tmp_path),
    )
    result = await loop.run("s1", "hi", emitter=emitter, ephemeral=True)

    assert result.content == "loop answer"
    texts = [event[len("text:"):] for event in emitter.events if event.startswith("text:")]
    assert "".join(texts) == "loop answer"


async def test_emitter_buffer_equals_the_result(tmp_path) -> None:
    """What the user saw must equal what the turn returned."""
    provider = NotedProvider([ModelResponse(content="consistent", finish_reason="stop")])
    emitter = RecordingEmitter()
    loop = AgentLoop(
        provider, registry(), SessionStore(tmp_path), workspace=str(tmp_path)
    )
    result = await loop.run("s1", "hi", emitter=emitter, ephemeral=True)
    assert emitter.buffer == result.content


async def test_error_turn_still_closes_the_stream() -> None:
    """A failed turn must not leave a consumer waiting forever."""
    provider = NotedProvider(
        [ModelResponse(content="", finish_reason="error", error="upstream down")]
    )
    emitter = RecordingEmitter()
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    assert result.stop_reason == StopReason.ERROR
    assert "end" in emitter.events


async def test_usage_is_reported_for_streamed_turns() -> None:
    provider = NotedProvider(
        [ModelResponse(content="hi", finish_reason="stop", usage=Usage(input_tokens=9, output_tokens=2))]
    )
    emitter = RecordingEmitter()
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    assert result.usage.input_tokens == 9
    assert result.usage.output_tokens == 2

async def test_provider_that_ignores_the_callback_still_delivers_text() -> None:
    """A backend that streams nothing must not produce an empty reply.

    The runner re-checks the completed response and catches up when no fragment
    was forwarded, so an older or stub backend degrades to a single delivery
    instead of silence.
    """

    class SilentProvider(NotedProvider):
        async def complete(self, messages, tools, *, system="", stream=False, on_delta=None):
            self.stream_flags.append(stream)
            return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    provider = SilentProvider([ModelResponse(content="no callbacks here", finish_reason="stop")])
    emitter = RecordingEmitter()
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    assert result.content == "no callbacks here"
    assert "".join(e[len("text:"):] for e in emitter.events if e.startswith("text:")) == "no callbacks here"


async def test_text_is_not_delivered_twice() -> None:
    """Catch-up must not duplicate text the backend already streamed."""
    provider = NotedProvider([ModelResponse(content="counted once", finish_reason="stop")])
    emitter = RecordingEmitter()
    await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
    )
    joined = "".join(e[len("text:"):] for e in emitter.events if e.startswith("text:"))
    assert joined == "counted once"


async def test_emitter_state_resets_between_turns() -> None:
    """A reused emitter must not make turn two look like it already streamed."""
    provider = NotedProvider(
        [
            ModelResponse(content="turn one", finish_reason="stop"),
            ModelResponse(content="turn two", finish_reason="stop"),
        ]
    )
    emitter = RecordingEmitter()
    runner = AgentRunner(provider, registry())
    spec = RunSpec(messages=[{"role": "user", "content": "hi"}])

    await runner.run(spec, emitter=emitter)
    assert emitter.streamed_content is True

    await runner.run(spec, emitter=emitter)
    texts = "".join(e[len("text:"):] for e in emitter.events if e.startswith("text:"))
    # Two turns, two answers, no missed catch-up.
    assert texts.count("turn one") == 1
    assert texts.count("turn two") == 1
