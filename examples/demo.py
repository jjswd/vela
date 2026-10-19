"""Runnable end-to-end demo. No API key required.

    python examples/demo.py

Uses a scripted provider so the tool loop, recovery paths, sessions, and
compaction all execute deterministically. Swap ``ScriptedProvider`` for
``build_default_loop`` to talk to a real model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vela import AgentLoop, SessionStore, StreamingEmitter, ToolRegistry, build_default_loop
from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.core.loop import load_builtin_tools
from vela.providers.base import GenerationConfig, Provider


class ScriptedProvider(Provider):
    """Pretends to be a model, following a fixed plan."""

    def __init__(self) -> None:
        super().__init__(GenerationConfig(model="demo"))
        self.step = 0

    async def complete(
        self, messages, tools, *, system="", stream=False, on_delta=None
    ) -> ModelResponse:
        self.step += 1
        if self.step == 1:
            return ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="calculate", arguments={"expression": "(12+8)*3"}),
                    ToolCall(id="c2", name="current_time", arguments={}),
                ),
                finish_reason="tool_calls",
                usage=Usage(input_tokens=180, output_tokens=42),
            )
        if self.step == 2:
            return ModelResponse(
                content="60 ", finish_reason="length", usage=Usage(input_tokens=240, output_tokens=2)
            )
        if self.step == 3:
            return ModelResponse(
                content="and the time is available above.",
                finish_reason="stop",
                usage=Usage(input_tokens=260, output_tokens=11),
            )
        return ModelResponse(content="Anything else?", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def banner(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


async def part_one_tool_loop() -> None:
    banner("1. Tool loop, truncated output recovery, usage accounting")
    provider = ScriptedProvider()
    tools = ToolRegistry(load_builtin_tools("./.demo-workspace"))
    loop = AgentLoop(provider, tools, SessionStore("./.demo/sessions"), workspace="./.demo-workspace")

    result = await loop.run("demo:1", "What is (12+8)*3 and what time is it?")

    print(f"answer      : {result.content}")
    print(f"stop reason : {result.stop_reason}")
    print(f"iterations  : {result.iterations}")
    print(f"tools used  : {result.tools_used}")
    print(f"usage       : {result.usage.input_tokens} in / "
          f"{result.usage.output_tokens} out (estimated={result.usage.estimated})")

    banner("1b. The same conversation continues")
    second = await loop.run("demo:1", "thanks")
    print(f"answer      : {second.content}")
    print(f"history     : {len(second.messages)} messages persisted to ./.demo/sessions")

    banner("1c. Streaming the final reply")
    chunks: list[str] = []
    emitter = StreamingEmitter(on_text=chunks.append)
    await loop.run("demo:stream", "hello", emitter=emitter, ephemeral=True)
    print(f"deltas      : {len(chunks)} chunk(s) -> {''.join(chunks)!r}")


async def part_two_isolation() -> None:
    banner("2. Session isolation and serialised ordering")
    provider = ScriptedProvider()
    tools = ToolRegistry(load_builtin_tools("./.demo-workspace"))
    loop = AgentLoop(provider, tools, SessionStore("./.demo/sessions"), workspace="./.demo-workspace")

    results = await asyncio.gather(
        loop.submit("user:alice", "alice speaks first"),
        loop.submit("user:bob", "bob speaks second"),
    )
    for key, result in zip(("alice", "bob"), results, strict=True):
        store = SessionStore("./.demo/sessions")
        session = store.load(f"user:{key}")
        first = session.messages[0]["content"]
        print(f"{key:6}: first message = {first!r}, turns = {len(result.messages)}")


async def part_three_real_model() -> None:
    """Same loop, different backends. Nothing above the provider layer changes."""
    import os

    from vela import available_providers, create_provider

    banner("3. Backend selection is configuration only")
    print(f"registered backends: {', '.join(available_providers())}\n")

    # Which backend is configured:
    backend = os.environ.get("VELA_BACKEND", "")
    if not backend:
        print("Set VELA_BACKEND + the matching key (ANTHROPIC_API_KEY / OPENAI_API_KEY / ...)")
        print("to run this part against a real model. Skipping.\n")
        return

    key_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(backend, "OPENAI_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        print(f"{key_env} not set; skipping.\n")
        return

    provider = create_provider(
        backend, model=os.environ.get("VELA_MODEL", "gpt-4o-mini"), api_key=api_key
    )
    print(f"instantiated: {type(provider).__name__} for {backend}\n")

    loop = build_default_loop(
        backend=backend,
        api_key=api_key,
        model=os.environ.get("VELA_MODEL", "gpt-4o-mini"),
        root="./.demo",
        workspace="./.demo-workspace",
        system="You are a concise assistant.",
    )
    try:
        emitter = StreamingEmitter(on_text=lambda delta: print(delta, end="", flush=True))
        result = await loop.run("live", "用一句话介绍你自己", emitter=emitter, ephemeral=True)
        print(f"\n\nusage: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
    finally:
        await loop.aclose()


async def main() -> None:
    await part_one_tool_loop()
    await part_two_isolation()
    await part_three_real_model()
    print("\nDone. Artefacts written to ./.demo/\n")


if __name__ == "__main__":
    asyncio.run(main())
