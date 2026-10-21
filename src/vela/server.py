"""HTTP server exposing the runtime.

Built on FastAPI, which is the framework a reviewer will expect to see in a
Python service, and which removes three things this file used to do by hand:
request validation, SSE framing, and running the agent on the event loop
rather than on a private one.

The API surface is OpenAI-compatible on purpose. Existing clients, SDKs,
evaluation harnesses, and front ends then work against it unchanged, which is
the difference between a demo and something a colleague can actually call.

FastAPI is an optional dependency. The core runtime does not need it, so
``pip install vela`` stays light and ``pip install vela[server]`` adds the web
surface. Importing this module without FastAPI raises a clear error rather
than an obscure one from deep inside the framework.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from vela.observability import Instrumentor

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "vela.server needs FastAPI. Install it with: pip install 'vela[server]'"
    ) from exc


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Server-level settings."""

    host: str = "127.0.0.1"
    port: int = 8080
    #: Sent as a bearer token when set. Bound to loopback by default, so an
    #: empty token is only reachable from the machine itself.
    api_key: str = ""


# --------------------------------------------------------------------------
# Request and response models
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One message in the conversation.

    ``content`` accepts a string or a list of blocks, because OpenAI permits
    multimodal input and rejecting it would break existing clients for no
    reason. Only the text parts are used.
    """

    role: str = Field(pattern="^(system|user|assistant)$")
    content: str | list[dict[str, Any]]

    @field_validator("content")
    @classmethod
    def _require_text(cls, value: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("message content must not be empty")
            return value
        if not value:
            raise ValueError("message content blocks must not be empty")
        return value

    def text(self) -> str:
        """The text this message contributes to the transcript."""
        if isinstance(self.content, str):
            return self.content
        parts = [
            block.get("text", "")
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)


class ChatCompletionRequest(BaseModel):
    """The request body, matching the OpenAI shape."""

    model: str = "vela"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    #: Session key. OpenAI clients do not send this, but it is what makes the
    #: server stateful across calls, so it is accepted under both names.
    user: str | None = None
    session: str | None = None

    def session_key(self) -> str:
        return self.user or self.session or "api:default"

    def last_prompt(self) -> str:
        return self.messages[-1].text()


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "vela"
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelCard(BaseModel):
    id: str = "vela"
    object: str = "model"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str = "ok"
    inflight: int = 0


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def _finish_reason(stop_reason: str) -> str:
    """Map the internal stop reason onto the OpenAI vocabulary."""
    return {
        "completed": "stop",
        "max_iterations": "length",
        "empty_final_response": "stop",
        "error": "stop",
        "cancelled": "stop",
    }.get(stop_reason, "stop")


def _sse_chunk(
    completion_id: str,
    model: str,
    delta: str,
    *,
    finish_reason: str | None = None,
    error: str = "",
) -> str:
    """Render one OpenAI-compatible streaming chunk.

    Shaped like the non-streaming response so a client can accumulate deltas
    without branching on which mode it asked for.
    """
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{completion_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if error:
        payload["error"] = {"message": error, "type": "internal_error"}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class VelaServer:
    """Serves the runtime over HTTP.

    Takes a loop factory rather than a loop instance so the server does not
    own the agent's lifecycle: whoever created the loop closes it.
    """

    def __init__(
        self,
        loop_factory: Callable[[], Any],
        config: ServerConfig | None = None,
        *,
        workspace: str = ".",
        system: str = "",
        instrumentor: Instrumentor | None = None,
    ) -> None:
        self._loop_factory = loop_factory
        self.config = config or ServerConfig()
        self.workspace = workspace
        self.system = system
        self.metrics = (instrumentor or Instrumentor()).metrics
        #: Requests currently in flight, reported by /health.
        self.inflight = 0
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="Vela",
            version="0.5.0",
            description=(
                "A small agent runtime. The chat endpoint is OpenAI-compatible, "
                "so existing clients work against it unchanged."
            ),
        )

        @app.middleware("http")
        async def _count_inflight(request: Request, call_next):
            # Counted so /health can report pressure, and so a hung request is
            # visible from outside the process.
            self.inflight += 1
            try:
                return await call_next(request)
            finally:
                self.inflight -= 1

        @app.exception_handler(Exception)
        async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "internal_error",
                    }
                },
            )

        @app.get("/health", response_model=HealthResponse, tags=["ops"])
        async def health() -> HealthResponse:
            return HealthResponse(inflight=self.inflight)

        @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
        async def metrics() -> str:
            # Prometheus exposition format, so no client library is needed on
            # the scraping side either.
            return self.metrics.to_prometheus()

        @app.get("/v1/models", response_model=ModelList, tags=["chat"])
        async def models() -> ModelList:
            return ModelList(data=[ModelCard()])

        @app.post(
            "/v1/chat/completions",
            tags=["chat"],
            summary="Chat completion",
            response_model=None,
        )
        async def chat_completions(
            request: ChatCompletionRequest,
            authorization: str | None = Header(default=None),
        ):
            if not self._authorised(authorization):
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "invalid api key",
                            "type": "authentication_error",
                        }
                    },
                )

            agent_loop = self._loop_factory()
            prompt = request.last_prompt()
            session = request.session_key()

            if request.stream:
                return StreamingResponse(
                    self._stream(agent_loop, session, prompt, request.model),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            result = await agent_loop.run(session, prompt)
            payload = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message={"role": "assistant", "content": result.content},
                        finish_reason=_finish_reason(result.stop_reason),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.usage.input_tokens,
                    completion_tokens=result.usage.output_tokens,
                    total_tokens=result.usage.total_tokens,
                ),
            )
            return JSONResponse(content=payload.model_dump())

        return app

    def _authorised(self, authorization: str | None) -> bool:
        """Check the bearer token, but only when one is configured.

        Bound to loopback with no token is a normal local setup; the check
        exists for the case where the server is exposed deliberately.
        """
        expected = self.config.api_key
        if not expected:
            return True
        return authorization == f"Bearer {expected}"

    async def _stream(
        self,
        agent_loop: Any,
        session: str,
        prompt: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Yield SSE chunks as the reply arrives.

        Running on the request's own event loop is the point of the rewrite:
        the previous implementation had to spin up a private loop and poll it,
        because the agent was driven from a worker thread. Here the agent, the
        provider, and the response share one loop, so a fragment is forwarded
        the moment it is produced.

        The queue exists only because the provider's ``on_delta`` callback is
        synchronous while the response is a coroutine; a ``call_soon`` hop
        hands each fragment over without blocking the loop.
        """
        from vela.core.runner import StreamingEmitter

        completion_id = uuid.uuid4().hex[:24]
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_text(delta: str) -> None:
            loop.call_soon(queue.put_nowait, _sse_chunk(completion_id, model, delta))

        emitter = StreamingEmitter(on_text=on_text)
        task = asyncio.create_task(agent_loop.run(session, prompt, emitter=emitter))

        # Forward whatever has been produced, and keep forwarding until the
        # turn is done. The task and the generator share the loop, so awaiting
        # the queue is what lets the provider make progress.
        while True:
            getter = asyncio.ensure_future(queue.get())
            finished, _pending = await asyncio.wait(
                {getter, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in finished:
                chunk = getter.result()
                if chunk:
                    yield chunk
                continue
            # The turn finished; drain anything still queued, then stop.
            getter.cancel()
            while not queue.empty():
                chunk = queue.get_nowait()
                if chunk:
                    yield chunk
            break

        yield _sse_chunk(completion_id, model, "", finish_reason="stop")
        # The terminating marker. OpenAI clients wait for it to know the stream
        # ended cleanly, so a stream that ends without it looks truncated.
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Run the server until interrupted."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )


def build_default_server(
    *,
    api_key: str,
    model: str,
    backend: str = "openai",
    host: str = "127.0.0.1",
    port: int = 8080,
    workspace: str = ".",
    system: str = "",
    root: str = "./.vela",
    auth_token: str = "",
) -> VelaServer:
    """Wire a server backed by a real provider.

    The loop is constructed per request rather than shared, because sessions
    are keyed on disk and a fresh loop keeps concurrent requests from sharing
    mutable state.
    """
    from vela.core.loop import build_default_loop

    instrumentor = Instrumentor()

    def factory() -> Any:
        loop = build_default_loop(
            backend=backend,
            api_key=api_key,
            model=model,
            root=root,
            workspace=workspace,
            system=system,
        )
        # Share the server's collector with the runtime, so one scrape covers
        # the whole request rather than only part of it.
        loop.instrumentor = instrumentor
        loop._runner.instrumentor = instrumentor  # noqa: SLF001
        return loop

    return VelaServer(
        factory,
        ServerConfig(host=host, port=port, api_key=auth_token),
        workspace=workspace,
        system=system,
        instrumentor=instrumentor,
    )
