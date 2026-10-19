"""HTTP server exposing the runtime, with no third-party dependency.

Two reasons it is built on the standard library rather than a framework. First,
the runtime's value proposition is being small enough to read, and a web
framework would be the heaviest dependency in it. Second, the surface here is
three endpoints; a framework would add more concepts than it removes.

The API is OpenAI-compatible on purpose. Existing clients, SDKs, evaluation
harnesses, and front ends then work against it unchanged, which is the
difference between a demo and something a colleague can actually call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vela.observability import Instrumentor

logger = logging.getLogger(__name__)

#: Refuse bodies larger than this. The model decides how much text arrives, so an
#: unbounded body is an easy way to be handed an out-of-memory payload.
MAX_BODY_BYTES = 4 * 1024 * 1024


@dataclass
class ServerConfig:
    """Server-level settings."""

    host: str = "127.0.0.1"
    port: int = 8080
    #: Sent as a bearer token when set. Bound to loopback by default, so an
    #: empty token is only reachable from the machine itself.
    api_key: str = ""


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
        self._httpd: ThreadingHTTPServer | None = None
        #: Requests currently in flight, for graceful shutdown and for /metrics.
        self.inflight = 0

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def handle_chat(self, body: dict[str, Any], stream: bool) -> dict[str, Any] | str:
        """Run one chat completion.

        Returns either a response dict or, when streaming, an iterator of SSE
        payload strings. Kept synchronous and called through ``run_coroutine``
        so the async loop stays owned by the loop layer.
        """
        messages = _parse_messages(body.get("messages"))
        if not messages:
            raise _BadRequest("'messages' must be a non-empty array")

        session = body.get("user") or body.get("session") or "api:default"
        prompt = messages[-1]["content"]

        agent_loop = self._loop_factory()

        if stream:
            return self._stream_response(agent_loop, session, prompt)
        return self._blocking_response(agent_loop, session, prompt)

    def _blocking_response(self, agent_loop, session: str, prompt: str) -> dict[str, Any]:
        result = asyncio.run(agent_loop.run(session, prompt))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body_model(),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": _finish_reason(result.stop_reason),
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.input_tokens,
                "completion_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

    def _stream_response(self, agent_loop, session: str, prompt: str):
        """Yield SSE chunks as the reply arrives.

        The handler runs on a request thread, not the thread that owns an event
        loop, so this drives a loop of its own. Fragments are forwarded with
        ``call_soon_threadsafe`` as the provider produces them, then the loop is
        stepped until the agent finishes, draining whatever it queued.
        """
        loop = asyncio.new_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        completion_id = uuid.uuid4().hex[:24]
        model = body_model()

        def offer(chunk: str) -> None:
            # The producer coroutine and this generator share the loop, so a
            # direct put is safe; the threadsafe path is only needed because the
            # agent may also be driven from elsewhere.
            queue.put_nowait(chunk)

        async def drive() -> None:
            from vela.core.runner import StreamingEmitter

            def on_text(delta: str) -> None:
                offer(_sse_chunk(completion_id, model, delta))

            try:
                await agent_loop.run(session, prompt, emitter=StreamingEmitter(on_text=on_text))
            except Exception as exc:  # noqa: BLE001 - reported to the client
                offer(_sse_chunk(completion_id, model, "", error=str(exc)))
            finally:
                offer("")

        try:
            task = loop.create_task(drive())
            # Step the loop, forwarding fragments, until the agent signals done.
            while not task.done():
                loop.run_until_complete(asyncio.sleep(0))
                while not queue.empty():
                    chunk = queue.get_nowait()
                    if chunk:
                        yield chunk
            # Drain anything queued on the final iteration.
            loop.run_until_complete(asyncio.sleep(0))
            while not queue.empty():
                chunk = queue.get_nowait()
                if chunk:
                    yield chunk
        finally:
            loop.close()

        yield _sse_chunk(completion_id, model, "", finish_reason="stop")

    def handle_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": body_model(), "object": "model"}]}

    def metrics_text(self) -> str:
        return self.metrics.to_prometheus()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        server = ThreadingHTTPServer((self.config.host, self.config.port), _Handler)
        server.vela = self  # type: ignore[attr-defined]
        self._httpd = server
        logger.info(
            "listening on http://%s:%d (POST /v1/chat/completions)",
            self.config.host,
            self.config.port,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()


def body_model() -> str:
    return "vela"


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
            {"index": 0, "delta": {"content": delta} if delta else {}, "finish_reason": finish_reason}
        ],
    }
    if error:
        payload["error"] = {"message": error, "type": "internal_error"}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class _BadRequest(ValueError):
    """A client error, mapped to 400."""


# --------------------------------------------------------------------------
# Message parsing
# --------------------------------------------------------------------------


def _parse_messages(raw: Any) -> list[dict[str, str]]:
    """Accept both the OpenAI message array and a bare string prompt."""
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if not isinstance(raw, list):
        raise _BadRequest("'messages' must be an array or a string")

    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise _BadRequest("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise _BadRequest(f"unsupported role: {role!r}")
        # Content may arrive as a list of blocks, which OpenAI permits for
        # multimodal input. Flatten the text parts rather than rejecting.
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "".join(parts)
        if not isinstance(content, str):
            raise _BadRequest(f"message {role!r} must have string or block content")
        messages.append({"role": role, "content": content})
    return messages


def _finish_reason(stop_reason: str) -> str:
    """Map the internal stop reason onto the OpenAI vocabulary."""
    return {
        "completed": "stop",
        "max_iterations": "length",
        "empty_final_response": "stop",
        "error": "stop",
        "cancelled": "stop",
    }.get(stop_reason, "stop")


def _sse(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Maps HTTP onto the runtime. Kept thin on purpose."""

    protocol_version = "HTTP/1.1"
    server_version = "Vela/0.4"

    @property
    def vela(self) -> VelaServer:
        return self.server.vela  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Route through logging rather than stderr so it lands wherever the
        # application's logs are configured to go.
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- responses --------------------------------------------------

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        """Check the bearer token, but only when one is configured.

        Bound to loopback with no token is a normal local setup; the check
        exists for the case where the server is exposed deliberately.
        """
        expected = self.vela.config.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {expected}"

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise _BadRequest("request body too large")
        if length <= 0:
            raise _BadRequest("empty request body")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise _BadRequest(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise _BadRequest("request body must be a JSON object")
        return payload

    # -- routes -----------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0]
        if path in {"/health", "/healthz"}:
            self._send_json(HTTPStatus.OK, {"status": "ok", "inflight": self.vela.inflight})
            return
        if path == "/metrics":
            self._send_text(HTTPStatus.OK, self.vela.metrics_text(), "text/plain")
            return
        if path == "/v1/models":
            self._send_json(HTTPStatus.OK, self.vela.handle_models())
            return
        self._send_json(HTTPStatus.NOT_FOUND, _error("unknown endpoint", "not_found_error"))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, _error("unknown endpoint", "not_found_error"))
            return

        if not self._authorised():
            self._send_json(
                HTTPStatus.UNAUTHORIZED, _error("invalid api key", "authentication_error")
            )
            return

        try:
            body = self._read_body()
            self.vela.inflight += 1
            try:
                result = self.vela.handle_chat(body, stream=bool(body.get("stream")))
            finally:
                self.vela.inflight -= 1
        except _BadRequest as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, _error(str(exc), "invalid_request_error"))
            return
        except Exception as exc:  # noqa: BLE001 - boundary
            logger.exception("request failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error(f"{type(exc).__name__}: {exc}", "internal_error"),
            )
            return

        if isinstance(result, str):
            self._send_json(HTTPStatus.OK, json.loads(result))
            return
        if isinstance(result, dict):
            self._send_json(HTTPStatus.OK, result)
            return

        # Iterator: stream it as server-sent events.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in result:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _error(message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind}}


def build_default_server(
    *,
    api_key: str,
    model: str,
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

    def factory():
        return build_default_loop(
            backend=api_key.get("_backend", "openai") if isinstance(api_key, dict) else "openai",
            api_key=None if isinstance(api_key, dict) else api_key,
            model=model,
            root=root,
            workspace=workspace,
            system=system,
        )

    return VelaServer(
        factory,
        ServerConfig(host=host, port=port, api_key=auth_token),
        workspace=workspace,
        system=system,
    )
