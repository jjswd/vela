"""Tests for the orchestration layer: sessions, compaction, isolation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vela.core.contracts import ModelResponse, ToolCall
from vela.core.loop import AgentLoop, SessionStore
from vela.providers.base import GenerationConfig, Provider
from vela.tools.registry import ToolRegistry


class EchoProvider(Provider):
    """Answers with a tool call the first time, then a final answer."""

    def __init__(self) -> None:
        super().__init__(GenerationConfig(model="test"))
        self.seen: list[list[dict[str, Any]]] = []
        self.pending: list[str] = []

    async def complete(
        self, messages, tools, *, system="", stream=False, on_delta=None
    ) -> ModelResponse:
        self.seen.append(list(messages))
        if self.pending:
            return ModelResponse(content=self.pending.pop(0), finish_reason="stop")
        if len(self.seen) % 2 == 1:
            return ModelResponse(
                content="",
                tool_calls=(ToolCall(id=f"c{len(self.seen)}", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            )
        return ModelResponse(content="final answer", finish_reason="stop")

    async def aclose(self) -> None:
        return None


async def test_turn_persists_to_the_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    loop = AgentLoop(EchoProvider(), ToolRegistry(), store, workspace=str(tmp_path))

    result = await loop.run("s1", "hello")
    assert result.content == "final answer"

    reloaded = store.load("s1")
    assert reloaded.messages[0]["content"] == "hello"
    assert reloaded.messages[-1]["role"] == "assistant"


async def test_ephemeral_turn_does_not_persist(tmp_path) -> None:
    store = SessionStore(tmp_path)
    loop = AgentLoop(EchoProvider(), ToolRegistry(), store, workspace=str(tmp_path))
    await loop.run("s1", "hello", ephemeral=True)
    assert store.load("s1").messages == []


async def test_sessions_are_isolated(tmp_path) -> None:
    store = SessionStore(tmp_path)
    loop = AgentLoop(EchoProvider(), ToolRegistry(), store, workspace=str(tmp_path))

    await loop.run("alice", "alice's question")
    await loop.run("bob", "bob's question")

    alice = store.load("alice")
    bob = store.load("bob")
    assert any("alice's question" in str(m.get("content")) for m in alice.messages)
    assert not any("bob's question" in str(m.get("content")) for m in alice.messages)
    assert any("bob's question" in str(m.get("content")) for m in bob.messages)


async def test_queued_messages_run_in_order(tmp_path) -> None:
    """Per-session serialisation keeps the transcript consistent."""
    store = SessionStore(tmp_path)
    provider = EchoProvider()
    provider.pending = ["first", "second", "third"]
    loop = AgentLoop(provider, ToolRegistry(), store, workspace=str(tmp_path))

    results = await asyncio.gather(*[loop.submit("s1", text) for text in ("a", "b", "c")])
    assert [r.content for r in results] == ["first", "second", "third"]

    messages = store.load("s1").messages
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    assert user_turns == ["a", "b", "c"]


async def test_compaction_replaces_old_turns_with_a_summary(tmp_path) -> None:
    store = SessionStore(tmp_path)
    provider = EchoProvider()
    provider.pending = ["summary of earlier turns"] * 20
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        store,
        workspace=str(tmp_path),
        compact_at_tokens=1,  # force compaction on every turn
    )

    for index in range(6):
        await loop.run("s1", f"question {index} " + "padding " * 40)

    session = store.load("s1")
    assert session.summary, "a summary should have been produced"
    assert any("Summary of earlier conversation" in str(m.get("content")) for m in session.messages)


async def test_compaction_failure_keeps_history(tmp_path) -> None:
    """A failing summariser must never destroy the transcript."""
    store = SessionStore(tmp_path)
    async def failing(messages):
        raise RuntimeError("summariser down")

    loop = AgentLoop(
        EchoProvider(),
        ToolRegistry(),
        store,
        workspace=str(tmp_path),
        compact_at_tokens=1,
        summarize=failing,
    )
    for index in range(4):
        await loop.run("s1", f"q{index} " + "x" * 200)

    session = store.load("s1")
    # Content must survive even though compaction failed.
    assert any("q0" in str(m.get("content")) for m in session.messages)


def test_corrupt_session_is_quarantined_not_fatal(tmp_path) -> None:
    store = SessionStore(tmp_path)
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", "utf-8")

    session = store.load("broken")
    assert session.messages == []
    assert path.with_suffix(".corrupt").exists()


def test_session_key_is_sanitised_for_the_filename(tmp_path) -> None:
    """A key containing path separators must not escape the session directory."""
    store = SessionStore(tmp_path)
    session = store.load("../../escape")
    session.messages = [{"role": "user", "content": "hi"}]
    store.save(session)

    # Sanitisation applies to the filename only; the in-memory key is untouched.
    assert session.key == "../../escape"
    written = list(tmp_path.glob("*.json"))
    # Every unsafe character is replaced, so the name can no longer contain a
    # separator or a dot-segment.
    assert len(written) == 1
    assert ".." not in written[0].name
    assert written[0].parent == tmp_path


def test_session_roundtrip_preserves_unicode(tmp_path) -> None:
    store = SessionStore(tmp_path)
    original = store.load("s1")
    original.messages = [{"role": "user", "content": "中文内容 with emoji 🌊"}]
    store.save(original)

    assert store.load("s1").messages[0]["content"] == "中文内容 with emoji 🌊"


async def test_close_cancels_workers(tmp_path) -> None:
    loop = AgentLoop(EchoProvider(), ToolRegistry(), SessionStore(tmp_path), workspace=str(tmp_path))
    await loop.submit("s1", "hi")
    await loop.aclose()
    with pytest.raises(RuntimeError):
        await loop.submit("s1", "hi again")
