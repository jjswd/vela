"""Offline verification: exercise every subsystem without an API key.

    python examples/verify.py

This exists because a project that claims working features should be able to
demonstrate them on demand. It runs the tool loop, a recovery path, streaming,
session persistence, compaction, protocol conversion, and the caching cost
model, then reports pass or fail per item.

Exit code is non-zero if anything fails, so it doubles as a smoke test in CI.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# The benchmark lives beside this script; importing it keeps the figure quoted
# in the README tied to the code that produces it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from cache_benchmark import benchmark as run_cache_benchmark

from vela import (
    AgentLoop,
    AgentRunner,
    CachePolicy,
    CacheStats,
    CacheTTL,
    GenerationConfig,
    OpenAICompatProvider,
    RunSpec,
    SessionStore,
    StreamingEmitter,
    Tool,
    ToolRegistry,
    create_provider,
)
from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.providers.anthropic import AnthropicProvider
from vela.providers.base import Provider
from vela.providers.caching import estimate_savings

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok    {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class Echo(Tool):
    name = "echo"
    description = "echo the given text"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


class Scripted(Provider):
    """A provider that replays a fixed plan, honouring the streaming contract.

    Holding the double to the same contract as a real backend is deliberate: a
    double that skipped ``on_delta`` would let a broken streaming path pass.
    """

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.script: list[ModelResponse] = []
        self.finalizer: str | None = None
        self.flags: list[bool] = []

    async def complete(self, messages, tools, *, system="", stream=False, on_delta=None):
        self.flags.append(stream)
        # A tool-free request is the post-budget finalisation the runner makes.
        if self.finalizer is not None and not tools:
            text, self.finalizer = self.finalizer, None
            return ModelResponse(content=text, finish_reason="stop")
        response = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if stream and on_delta is not None and response.content:
            for index in range(0, len(response.content), 4):
                await on_delta(response.content[index:index + 4])
        return response

    async def aclose(self) -> None:
        return None


def make_scripted(
    script: list[ModelResponse], final: str | None = None
) -> Scripted:
    provider = Scripted(GenerationConfig(model="verify"))
    provider.script = script
    provider.finalizer = final
    return provider


def registry() -> ToolRegistry:
    return ToolRegistry([Echo()])


async def verify_tool_loop() -> None:
    section("1. Tool loop and usage accounting")
    provider = make_scripted(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "ping"}),),
                finish_reason="tool_calls",
                usage=Usage(input_tokens=10, output_tokens=3),
            ),
            ModelResponse(
                content="replied: ping",
                finish_reason="stop",
                usage=Usage(input_tokens=12, output_tokens=4),
            ),
        ]
    )
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "go"}])
    )
    check("tool executed", result.tools_used == ["echo"], str(result.tools_used))
    check("final answer returned", result.content == "replied: ping")
    check("usage accumulated", result.usage.input_tokens == 22)
    check("two iterations", result.iterations == 2, str(result.iterations))


async def verify_recovery() -> None:
    section("2. Recovery: truncated output is continued, not duplicated")
    provider = make_scripted(
        [
            ModelResponse(content="The answer is", finish_reason="length"),
            ModelResponse(content=" forty-two.", finish_reason="stop"),
        ]
    )
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "q"}])
    )
    check("segments joined", result.content == "The answer is forty-two.", repr(result.content))
    check("no duplication", result.content.count("The answer is") == 1)

    section("3. Recovery: iteration budget triggers a tool-free summary")
    provider = make_scripted(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            )
        ],
        final="here is the summary",
    )
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "q"}], max_iterations=3)
    )
    check("stop reason is max_iterations", result.stop_reason == "max_iterations")
    check("summary delivered", "here is the summary" in result.content)


async def verify_streaming() -> None:
    section("4. Streaming delivers fragments incrementally")
    provider = make_scripted([ModelResponse(content="streamed reply here", finish_reason="stop")])
    seen: list[str] = []
    emitter = StreamingEmitter(on_text=seen.append)
    result = await AgentRunner(provider, registry()).run(
        RunSpec(messages=[{"role": "user", "content": "q"}]), emitter=emitter
    )
    check("multiple fragments", len(seen) > 1, f"{len(seen)} deltas")
    check("fragments equal result", "".join(seen) == result.content)
    check("streaming requested", provider.flags == [True])

    section("5. SSE parsing over a fragmented byte stream")
    frames = [
        f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'
        for t in ["你", "好", "，", "世", "界"]
    ]
    frames.append('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    frames.append("data: [DONE]\n\n")
    body = "".join(frames).encode()

    class Chunked(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(body), 7):
                await asyncio.sleep(0)
                yield body[offset:offset + 7]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunked()
        )

    provider = OpenAICompatProvider(
        GenerationConfig(model="gpt-4o-mini"),
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    deltas: list[str] = []

    async def on_delta(text: str) -> None:
        deltas.append(text)

    response = await provider.complete(
        [{"role": "user", "content": "hi"}], [], stream=True, on_delta=on_delta
    )
    check("reassembled text", response.content == "你好，世界", repr(response.content))
    check("one delta per frame", deltas == ["你", "好", "，", "世", "界"], str(deltas))


async def verify_protocols() -> None:
    section("6. Same transcript survives both wire protocols")
    transcript = [
        {"role": "user", "content": "weather in Beijing?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "echo", "arguments": '{"text":"Beijing"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Beijing: 22C"},
        {"role": "assistant", "content": "It is 22C."},
    ]

    openai = OpenAICompatProvider(GenerationConfig(model="m"), api_key="k")._build_payload(
        transcript, [Echo().spec()], "sys", stream=False
    )["messages"]
    anthropic = AnthropicProvider(
        GenerationConfig(model="m"), api_key="k"
    )._build_payload(transcript, [Echo().spec()], "sys", stream=False)["messages"]

    check("OpenAI uses a tool role", any(m["role"] == "tool" for m in openai))
    check("Anthropic has no tool role", all(m["role"] in {"user", "assistant"} for m in anthropic))
    anthropic_result = [
        block
        for message in anthropic
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if block.get("type") == "tool_result"
    ]
    check("result id preserved", anthropic_result and anthropic_result[0]["tool_use_id"] == "c1")
    check("result content preserved", anthropic_result and anthropic_result[0]["content"] == "Beijing: 22C")


async def verify_sessions() -> None:
    section("7. Sessions: isolation, ordering, and persistence")
    with tempfile.TemporaryDirectory() as directory:
        store = SessionStore(directory)

        provider_a = make_scripted(
            [ModelResponse(content=f"echo:{i}", finish_reason="stop") for i in range(3)]
        )
        loop = AgentLoop(provider_a, registry(), SessionStore(directory), workspace=directory)
        results = await asyncio.gather(
            loop.submit("s", "one"), loop.submit("s", "two"), loop.submit("s", "three")
        )
        check("queued in order", [r.content for r in results] == ["echo:0", "echo:1", "echo:2"])

        messages = store.load("s").messages
        check("history persisted", len(messages) >= 6, f"{len(messages)} messages")

        loop_b = AgentLoop(
            make_scripted([ModelResponse(content="x", finish_reason="stop")]),
            registry(),
            SessionStore(directory),
            workspace=directory,
        )
        await loop_b.run("other", "different session")
        check(
            "sessions isolated",
            not any("different session" in str(m.get("content")) for m in store.load("s").messages),
        )
        await loop.aclose()
        await loop_b.aclose()


async def verify_compaction() -> None:
    section("8. Compaction replaces old turns with a summary")
    with tempfile.TemporaryDirectory() as directory:
        loop = AgentLoop(
            make_scripted([ModelResponse(content="summary line", finish_reason="stop")]),
            registry(),
            SessionStore(directory),
            workspace=directory,
            compact_at_tokens=1,
        )
        for index in range(5):
            await loop.run("c", f"question {index} " + "padding " * 40)

        session = SessionStore(directory).load("c")
        check("summary produced", bool(session.summary), repr(session.summary[:40]))
        check(
            "old turns replaced",
            any("Summary of earlier conversation" in str(m.get("content")) for m in session.messages),
        )
        await loop.aclose()


async def verify_caching() -> None:
    section("9. Caching cost model")
    check("5m break-even is 2 requests", CacheTTL.FIVE_MINUTES.break_even_uses == 2)
    check("1h break-even is 3 requests", CacheTTL.ONE_HOUR.break_even_uses == 3)
    check("disabled by default", CachePolicy().should_cache(100) is False)
    check(
        "refuses when too short",
        CachePolicy(enabled=True, ttl=CacheTTL.ONE_HOUR).should_cache(1) is False,
    )

    savings = estimate_savings(
        prefix_tokens=9000, tail_tokens=1000, turns=10, ttl=CacheTTL.FIVE_MINUTES
    )
    check("long conversation saves", savings > 0, f"{savings:,.0f} tokens-equivalent")

    stats = CacheStats()
    stats.record(read=0, write=9000, input_tokens=1000)
    for _ in range(9):
        stats.record(read=9000, write=0, input_tokens=1000)
    # 81,000 reads out of 100,000 total input tokens. A write is not a hit,
    # so the tokens stored on the first turn are in the denominator only.
    check(
        "hit rate is 81%",
        abs(stats.hit_rate - 0.81) < 0.001,
        f"{stats.hit_rate:.1%}",
    )
    check(
        "input cost drops to 29%",
        abs(stats.effective_cost_ratio - 0.2935) < 0.001,
        f"{stats.effective_cost_ratio:.1%}",
    )
    check(
        "cost ratio below 0.35",
        stats.effective_cost_ratio < 0.35,
        f"{stats.effective_cost_ratio:.3f}",
    )

    # The end-to-end path: the same numbers through a provider and a simulated
    # backend, so the figure quoted in the README cannot drift away from what
    # the code actually does.
    section("9b. End-to-end cache benchmark")
    benchmark = await run_cache_benchmark()
    check(
        "benchmark hit rate is 81%",
        abs(benchmark["hit_rate"] - 0.81) < 0.01,
        f"{benchmark['hit_rate']:.1%}",
    )
    check(
        "benchmark cost ratio is 0.293",
        abs(benchmark["cost_ratio"] - 0.293) < 0.01,
        f"{benchmark['cost_ratio']:.3f}",
    )
    check(
        "benchmark reduces input cost by 70.7%",
        abs((1 - benchmark["cost_ratio"]) * 100 - 70.7) < 0.5,
        f"{(1 - benchmark['cost_ratio']) * 100:.1f}%",
    )

    section("10. Cache markers only on stable prefixes")
    provider = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"),
        api_key="k",
        cache_policy=CachePolicy(enabled=True),
        expected_turns=20,
    )
    payload = provider._build_payload(
        [{"role": "user", "content": f"m{i}"} for i in range(6)],
        [Echo().spec()],
        "a stable system prompt",
        stream=False,
    )
    check("system marked", isinstance(payload["system"], list))
    check("tools marked", "cache_control" in payload["tools"][-1])

    off = AnthropicProvider(
        GenerationConfig(model="claude-sonnet-4-5"), api_key="k"
    )._build_payload([{"role": "user", "content": "m"}], [Echo().spec()], "sys", stream=False)
    check("no markers when disabled", off["system"] == "sys")


async def verify_registry() -> None:
    section("11. Tool registry")
    from vela.tools.builtin import builtin_tools

    with tempfile.TemporaryDirectory() as directory:
        tools = ToolRegistry(builtin_tools(directory))
        check("builtin tools registered", len(tools) >= 5, f"{len(tools)} tools")

        # "calculatr" is one edit from "calculate", so the registry should
        # offer it rather than just reporting the name as unknown.
        record = await tools.dispatch(
            [ToolCall(id="c1", name="calculatr", arguments={"expression": "1"})]
        )
        check(
            "typo corrected in the error",
            "Did you mean 'calculate'?" in record[0].result.content,
            record[0].result.content[:60],
        )

        record = await tools.dispatch(
            [ToolCall(id="c2", name="calculate", arguments={"expression": "(12+8)*3"})]
        )
        check("tool executed", record[0].result.content == "60", record[0].result.content)

        # The evaluator only accepts digits and operators, so an escape attempt
        # must come back as an error result rather than executing anything.
        record = await tools.dispatch(
            [ToolCall(id="c3", name="calculate", arguments={"expression": "__import__('os')"})]
        )
        check(
            "code execution refused as an error",
            record[0].result.is_error and "unsupported characters" in record[0].result.content.lower(),
            record[0].result.content[:52],
        )

        record = await tools.dispatch(
            [ToolCall(id="c5", name="calculate", arguments={"expression": "1/0"})]
        )
        check(
            "division by zero is an error, not a value",
            record[0].result.is_error,
            record[0].result.content[:44],
        )

        record = await tools.dispatch(
            [ToolCall(id="c4", name="read_file", arguments={"path": "../../../etc/passwd"})]
        )
        check("path escape refused", record[0].result.is_error, record[0].result.content[:40])


async def verify_backends() -> None:
    section("12. Backend registry")
    from vela import available_providers

    names = available_providers()
    check("multiple backends registered", len(names) >= 5, ", ".join(names))
    for name in names:
        provider = create_provider(name, model="m", api_key="k")
        check(f"  {name} instantiates", type(provider).__name__.endswith("Provider"))
    try:
        create_provider("openiae", model="m", api_key="k")
        check("unknown backend rejected", False)
    except Exception:
        check("unknown backend rejected", True)


async def main() -> int:
    print("Vela offline verification -- no API key required\n")
    for step in (
        verify_tool_loop,
        verify_recovery,
        verify_streaming,
        verify_protocols,
        verify_sessions,
        verify_compaction,
        verify_caching,
        verify_registry,
        verify_backends,
    ):
        await step()

    print(f"\n{'=' * 46}")
    print(f"passed: {PASSED}    failed: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
