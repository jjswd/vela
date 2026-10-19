"""Tests for the HTTP surface.

These start a real server on an ephemeral port and talk to it over a socket,
because the interesting failures in an HTTP layer are in the plumbing —
headers, status codes, chunk framing — and those never show up when the handler
is called directly.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus

import pytest

from vela.core.contracts import ModelResponse
from vela.core.loop import AgentLoop, SessionStore
from vela.core.runner import AgentRunner, RunSpec, StreamingEmitter
from vela.observability import Instrumentor
from vela.providers.base import GenerationConfig, Provider
from vela.server import ServerConfig, VelaServer, _finish_reason, _parse_messages
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


class Echo(Tool):
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


class Scripted(Provider):
    """Replays a fixed script, honouring the streaming contract."""

    def __init__(self, reply: str = "hello from vela") -> None:
        super().__init__(GenerationConfig(model="test"))
        self.reply = reply

    async def complete(self, messages, tools, *, system="", stream=False, on_delta=None):
        if stream and on_delta is not None:
            for index in range(0, len(self.reply), 4):
                await on_delta(self.reply[index:index + 4])
        return ModelResponse(content=self.reply, finish_reason="stop")

    async def aclose(self) -> None:
        return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Harness:
    """A live server plus a small request helper."""

    def __init__(self, reply: str = "hello from vela", api_key: str = "") -> None:
        self.root: str | None = None
        self.port = free_port()
        self.api_key = api_key
        # One Instrumentor shared by the server and every loop it builds, so a
        # scrape reflects the turns the server actually served -- which is the
        # only way the endpoint is useful in production too.
        self.instrumentor = Instrumentor()

        def factory():
            # A fresh temp dir shared across requests, so a session started by
            # one request is visible to the next.
            import tempfile

            if self.root is None:
                self.root = tempfile.mkdtemp()
            return AgentLoop(
                Scripted(reply),
                ToolRegistry([Echo()]),
                SessionStore(self.root),
                workspace=self.root,
                instrumentor=self.instrumentor,
            )

        self.server = VelaServer(
            factory,
            ServerConfig(host="127.0.0.1", port=self.port, api_key=api_key),
            instrumentor=self.instrumentor,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Harness:
        self.thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc: object) -> bool:
        self.server._httpd.shutdown()  # noqa: SLF001 - test teardown
        return False

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path: str, *, headers: dict | None = None):
        request = urllib.request.Request(self.url(path), headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    def post(self, path: str, payload: dict, *, headers: dict | None = None):
        request = urllib.request.Request(
            self.url(path),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


def test_health_endpoint() -> None:
    with Harness() as h:
        status, body = h.get("/health")
        assert status == HTTPStatus.OK
        assert json.loads(body)["status"] == "ok"


def test_models_endpoint_lists_a_model() -> None:
    with Harness() as h:
        status, body = h.get("/v1/models")
        assert status == HTTPStatus.OK
        payload = json.loads(body)
        assert payload["object"] == "list"
        assert payload["data"][0]["object"] == "model"


def test_chat_completion_returns_a_reply() -> None:
    with Harness() as h:
        status, body = h.post(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert status == HTTPStatus.OK
        payload = json.loads(body)
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["content"] == "hello from vela"
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert "usage" in payload


def test_bare_string_prompt_is_accepted() -> None:
    with Harness() as h:
        status, _ = h.post("/v1/chat/completions", {"messages": "just a string"})
        assert status == HTTPStatus.OK


def test_metrics_endpoint_exposes_counters() -> None:
    with Harness() as h:
        h.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        status, body = h.get("/metrics")
        assert status == HTTPStatus.OK
        assert "vela_counter" in body
        # The turn must actually have been counted.
        assert "turns_total" in body


def test_streaming_returns_sse_chunks() -> None:
    with Harness(reply="streamed reply") as h:
        request = urllib.request.Request(
            h.url("/v1/chat/completions"),
            data=json.dumps(
                {"messages": [{"role": "user", "content": "hi"}], "stream": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            raw = response.read().decode()

    assert raw.rstrip().endswith("data: [DONE]")
    chunks = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert chunks, "expected at least one chunk"
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert text == "streamed reply"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_unknown_endpoint_returns_404() -> None:
    with Harness() as h:
        status, body = h.get("/nope")
        assert status == HTTPStatus.NOT_FOUND
        assert json.loads(body)["error"]["type"] == "not_found_error"


def test_malformed_body_returns_400() -> None:
    with Harness() as h:
        request = urllib.request.Request(
            h.url("/v1/chat/completions"),
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            assert json.loads(exc.read().decode())["error"]["type"] == "invalid_request_error"
        assert status == HTTPStatus.BAD_REQUEST


def test_missing_messages_returns_400() -> None:
    with Harness() as h:
        status, body = h.post("/v1/chat/completions", {"model": "x"})
        assert status == HTTPStatus.BAD_REQUEST
        assert "messages" in json.loads(body)["error"]["message"]


def test_api_key_is_enforced_when_configured() -> None:
    with Harness(api_key="secret-token") as h:
        status, body = h.post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong"},
        )
        assert status == HTTPStatus.UNAUTHORIZED
        assert json.loads(body)["error"]["type"] == "authentication_error"

        status, _ = h.post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert status == HTTPStatus.OK


def test_open_endpoint_allows_anonymous_calls() -> None:
    """Loopback with no token is a normal local setup."""
    with Harness(api_key="") as h:
        status, _ = h.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert status == HTTPStatus.OK


def test_concurrent_requests_are_served() -> None:
    """The threading server must handle overlapping calls independently."""
    with Harness() as h:
        results: list[int] = []
        errors: list[str] = []

        def call() -> None:
            try:
                status, _ = h.post(
                    "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
                )
                results.append(status)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        threads = [threading.Thread(target=call) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors, errors
        assert results == [HTTPStatus.OK] * 6


# --------------------------------------------------------------------------
# Request parsing, without a socket
# --------------------------------------------------------------------------


def test_parse_messages_handles_block_content() -> None:
    """OpenAI permits multimodal blocks; flatten the text rather than reject."""
    messages = _parse_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at "},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "this"},
                ],
            }
        ]
    )
    assert messages[0]["content"] == "look at this"


def test_parse_messages_rejects_unknown_roles() -> None:
    with pytest.raises(ValueError):
        _parse_messages([{"role": "tool", "content": "x"}])


def test_finish_reason_mapping() -> None:
    assert _finish_reason("completed") == "stop"
    assert _finish_reason("max_iterations") == "length"
    assert _finish_reason("error") == "stop"
    assert _finish_reason("something_new") == "stop"


def test_agent_runs_without_the_server() -> None:
    """The HTTP layer is additive; the library API must remain sufficient."""
    import asyncio

    async def main() -> None:
        runner = AgentRunner(Scripted("direct"), ToolRegistry([Echo()]))
        result = await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
        assert result.content == "direct"

        emitter = StreamingEmitter(on_text=lambda delta: None)
        streamed = await runner.run(
            RunSpec(messages=[{"role": "user", "content": "hi"}]), emitter=emitter
        )
        assert streamed.content == "direct"

    asyncio.run(main())
