"""The orchestration layer: one turn from the channel's point of view.

Separated from :mod:`vela.core.runner` on purpose. The runner answers "how do I
talk to the model?"; this module answers "what does it mean to handle a message
from a chat platform?". The split keeps both sides replaceable: a new transport
never touches the model loop, and a new backend never touches routing.

Responsibilities live here and nowhere else: session identity, transcript
storage, compaction, workspace scoping, and lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vela.core.contracts import Usage
from vela.core.runner import AgentRunner, RunResult, RunSpec, StreamEmitter
from vela.observability import Instrumentor, Tracer
from vela.providers.base import Provider
from vela.tools.registry import ToolRegistry, load_builtin_tools

logger = logging.getLogger(__name__)

#: Rough characters-per-token ratio for English + CJK mixed text. Used only for
#: estimating whether compaction is due; exact counting needs the backend
#: tokenizer, which varies per provider and is not worth the coupling.
CHARS_PER_TOKEN = 3.5


@dataclass
class Session:
    """Conversation state for one conversation key."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=lambda: __import__("time").time())

    def approx_tokens(self) -> int:
        total = sum(len(str(m.get("content") or "")) for m in self.messages)
        total += len(self.summary)
        return int(total / CHARS_PER_TOKEN)


#: One lock per session directory, shared across every store instance in the
#: process. Keying per instance would not help: the HTTP server builds a new
#: store per request, so two concurrent requests would take different locks and
#: still collide on the same file. os.replace raises on Windows when the
#: destination is open elsewhere -- measured at 4 of 20 concurrent replaces --
#: so this is a correctness guard, not a nicety.
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class SessionStore:
    """Persists sessions as one JSON file per key.

    Reads and writes for one directory are serialised. A crash mid-write must
    not leave an unparseable file, so the write goes to a temporary path and is
    renamed into place; the lock is what keeps concurrent writers from tripping
    over each other during that rename.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        resolved = str(self.root.resolve())
        with _PATH_LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(resolved, threading.RLock())

    def _lock_for(self, key: str) -> threading.RLock:
        # Signature kept for call sites that pass a key; the lock covers the
        # whole directory, which is stricter and still allows different
        # directories to proceed in parallel.
        return self._lock

    def load(self, key: str) -> Session:
        path = self._path(key)
        with self._lock_for(key):
            if not path.exists():
                return Session(key=key)
            return self._read_locked(path, key)

    def _read_locked(self, path: Path, key: str) -> Session:
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt session must not break the conversation; start fresh and
            # keep the damaged file for inspection.
            logger.warning("unreadable session %s, starting new: %s", key, exc)
            self._quarantine(path)
            return Session(key=key)
        return Session(
            key=key,
            messages=payload.get("messages") or [],
            summary=payload.get("summary") or "",
            created_at=payload.get("created_at") or 0.0,
        )

    def save(self, session: Session) -> None:
        path = self._path(session.key)
        payload = {
            "key": session.key,
            "created_at": session.created_at,
            "summary": session.summary,
            "messages": session.messages,
        }
        # Write-then-rename: a crash mid-write must not leave a half-written
        # session that fails to parse on the next load. The temporary name is
        # unique per write so two concurrent writers cannot target one file --
        # a shared ".tmp" makes the loser's rename fail with a permission error
        # on Windows and silently discard data elsewhere.
        unique = f".{uuid.uuid4().hex[:12]}.tmp"
        temporary = path.with_name(path.name + unique)
        # os.replace fails on Windows when the destination is open elsewhere, so
        # writers for one session must not overlap. Verified necessary: 4 of 20
        # unsynchronised replaces raised PermissionError.
        with self._lock_for(session.key):
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
                )
                temporary.replace(path)
            except OSError:
                temporary.unlink(missing_ok=True)
                raise

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
        return self.root / f"{safe or 'default'}.json"

    def _quarantine(self, path: Path) -> None:
        # Best effort: if the rename fails there is still a usable session, just
        # without the damaged file preserved for inspection.
        with suppress(OSError):
            path.replace(path.with_suffix(".corrupt"))


class AgentLoop:
    """Turn-oriented orchestration over a runner."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        store: SessionStore,
        *,
        workspace: str | Path = ".",
        system: str = "",
        max_iterations: int = 12,
        compact_at_tokens: int = 60_000,
        summarize: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None,
        instrumentor: Instrumentor | None = None,
        trace: bool = False,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._store = store
        #: Optional. Without it the runtime emits nothing, which keeps the core
        #: independent of any observability stack.
        self.instrumentor = instrumentor
        self._trace = trace
        self._runner = AgentRunner(provider, tools, instrumentor=instrumentor)
        self.workspace = Path(workspace)
        self.system = system
        self.max_iterations = max_iterations
        self.compact_at_tokens = compact_at_tokens
        self._summarize = summarize
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        # Per-session lock around load -> run -> save. Without it, two
        # concurrent callers for one session both load the same transcript and
        # the second save discards the first turn's work. Per-session rather than
        # global so unrelated sessions stay parallel.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._closed = False
        #: Spans from the most recent turn, for callers that want to inspect a
        #: turn without attaching a tracing backend.
        self.last_trace = Tracer(enabled=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, session_key: str, text: str) -> RunResult:
        """Queue a message and wait for its turn.

        One worker per session processes messages in order. Running turns
        concurrently within a session would interleave tool results into a
        transcript whose order no longer matches the requests, so serialisation
        is a correctness requirement rather than a performance choice.
        """
        if self._closed:
            raise RuntimeError("loop is closed")

        queue = self._queues.get(session_key)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[session_key] = queue
            self._workers[session_key] = asyncio.create_task(
                self._worker(session_key), name=f"vela-worker-{session_key}"
            )

        done: asyncio.Future[RunResult] = asyncio.get_running_loop().create_future()
        await queue.put({"text": text, "future": done})
        return await done

    async def run(
        self,
        session_key: str,
        text: str,
        *,
        emitter: StreamEmitter | None = None,
        ephemeral: bool = False,
    ) -> RunResult:
        """Process one message directly, bypassing the per-session queue.

        Serialised per session so concurrent callers cannot interleave a
        load-modify-save cycle on the same transcript.
        """
        async with self._session_lock(session_key):
            session = self._store.load(session_key)
            return await self._execute(
                session, text, emitter=emitter, ephemeral=ephemeral
            )

    def _session_lock(self, key: str) -> asyncio.Lock:
        """Lock for one session's load-run-save cycle.

        Lazily created under a guard because several threads may reach a fresh
        loop at once: each request in the HTTP server runs on its own thread.
        """
        with self._session_locks_guard:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[key] = lock
            return lock

    def queue_message(self, session_key: str, text: str) -> None:
        """Enqueue input for injection into an in-flight turn.

        Lets a user add context mid-task. The runner picks it up at the next
        iteration boundary rather than mutating the transcript mid-request.
        """
        self._pending.setdefault(session_key, []).append({"role": "user", "content": text})

    async def aclose(self) -> None:
        """Cancel in-flight work and release the provider."""
        self._closed = True
        for task in self._workers.values():
            task.cancel()
        for task in self._workers.values():
            # Shutdown is best effort: a worker that was already failing must
            # not prevent the remaining ones from stopping.
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._workers.clear()
        self._queues.clear()
        await self._provider.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _worker(self, session_key: str) -> None:
        queue = self._queues[session_key]
        while True:
            item = await queue.get()
            future: asyncio.Future[RunResult] = item["future"]
            if future.cancelled():
                continue
            try:
                result = await self.run(session_key, item["text"])
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)

    async def _execute(
        self,
        session: Session,
        text: str,
        *,
        emitter: StreamEmitter | None = None,
        ephemeral: bool = False,
    ) -> RunResult:
        """Load context, run one turn, persist. The heart of a single turn."""
        session.messages.append({"role": "user", "content": text})
        await self._compact_if_needed(session)

        drain = lambda: self._collect_pending(session.key)  # noqa: E731
        tracer = Tracer(enabled=self._trace)
        spec = RunSpec(
            system=self._build_system(session),
            messages=list(session.messages),
            max_iterations=self.max_iterations,
            session_id=session.key,
            tracer=tracer,
        )
        started = time.perf_counter()
        try:
            result = await self._runner.run(spec, emitter=emitter, drain_injections=drain)
        finally:
            self.last_trace = tracer

        session.messages = result.messages
        if not ephemeral:
            self._store.save(session)
        logger.info(
            "turn done session=%s stop=%s tools=%s usage=%s",
            session.key,
            result.stop_reason,
            len(result.tools_used),
            result.usage.total_tokens,
        )
        if self.instrumentor is not None:
            self.instrumentor.record_turn(
                session=session.key,
                result=result,
                duration_s=time.perf_counter() - started,
                tracer=tracer,
            )
        return result

    async def _collect_pending(self, session_key: str) -> list[dict[str, Any]]:
        pending = self._pending.pop(session_key, [])
        if pending:
            logger.info("injected %d message(s) into %s", len(pending), session_key)
        return pending

    def _build_system(self, session: Session) -> str:
        parts = [self.system]
        if session.summary:
            parts.append(f"## Conversation summary so far\n{session.summary}")
        return "\n\n".join(part for part in parts if part)

    async def _compact_if_needed(self, session: Session) -> None:
        """Summarise older turns once the transcript approaches the limit.

        Keeps the most recent turns verbatim -- they carry the immediate task --
        and replaces everything older with a summary.
        """
        if session.approx_tokens() < self.compact_at_tokens:
            return
        keep = 8
        if len(session.messages) <= keep:
            return

        older, recent = session.messages[:-keep], session.messages[-keep:]
        logger.info(
            "compacting session=%s: %d messages -> summary + %d",
            session.key,
            len(session.messages),
            len(recent),
        )

        summary = await self._summarize_older(older)
        session.summary = summary
        session.messages = [
            {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}
        ] + recent

    async def _summarize_older(self, messages: list[dict[str, Any]]) -> str:
        """Produce a summary, degrading to a local extract if the model fails.

        Compaction runs on every long conversation, so it must never be able to
        break a turn: a summariser outage has to cost detail, not history.
        """
        transcript = "\n".join(
            f"{m.get('role')}: {str(m.get('content'))[:500]}" for m in messages
        )

        if self._summarize is not None:
            try:
                summary = await self._summarize(messages)
                if summary.strip():
                    return summary.strip()
                logger.warning("custom summariser returned nothing; using fallback")
            except Exception as exc:  # noqa: BLE001
                logger.warning("custom summariser failed, using fallback: %s", exc)
            return self._extract_fallback(messages)

        try:
            response = await self._provider.complete(
                [
                    {
                        "role": "user",
                        "content": (
                            "Summarise this conversation, preserving decisions, "
                            "facts, file paths, and open tasks. Be terse.\n\n" + transcript
                        ),
                    }
                ],
                [],
            )
            if not response.is_error and response.content.strip():
                return response.content.strip()
            logger.warning("summarisation returned an error response; using fallback")
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarisation failed, using fallback: %s", exc)

        return self._extract_fallback(messages)

    @staticmethod
    def _extract_fallback(messages: list[dict[str, Any]]) -> str:
        """Last-resort summary that keeps a readable trace of the conversation."""
        return "\n".join(
            f"{m.get('role')}: {str(m.get('content'))[:200]}" for m in messages[-10:]
        )


def build_default_loop(
    *,
    backend: str = "openai",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    root: str | Path = "./.vela",
    workspace: str | Path = ".",
    system: str = "",
    extra_tools: list | None = None,
    max_iterations: int = 12,
) -> AgentLoop:
    """Wire a working loop with sensible defaults.

    ``backend`` selects the protocol implementation; everything above the
    provider layer is unchanged regardless of which one is chosen.
    """
    from vela.providers.factory import create_provider

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry(load_builtin_tools(str(workspace)))
    for tool in extra_tools or []:
        tools.register(tool)

    provider = create_provider(
        backend, model=model, api_key=api_key, base_url=base_url
    )
    return AgentLoop(
        provider,
        tools,
        SessionStore(root / "sessions"),
        workspace=workspace,
        system=system,
        max_iterations=max_iterations,
    )


__all__ = [
    "AgentLoop",
    "Session",
    "SessionStore",
    "build_default_loop",
    "Usage",
    "RunResult",
    "RunSpec",
    "StreamEmitter",
]
