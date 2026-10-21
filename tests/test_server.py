"""Tests for the HTTP surface.

Driven through FastAPI's ``TestClient``, which exercises the real ASGI stack
-- routing, validation, middleware, serialisation -- without binding a socket.
That matters because the failures worth catching in an HTTP layer live in the
plumbing, and an in-process client reaches the same code a socket would while
staying fast and deterministic.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from vela.core.contracts import ModelResponse
from vela.core.loop import AgentLoop, SessionStore
from vela.observability import Instrumentor
from vela.providers.base import GenerationConfig, Provider
from vela.server import ServerConfig, VelaServer, _finish_reason
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


class Echo(Tool):
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


class Scripted(Provider):
    """Replays a fixed reply, honouring the streaming contract."""

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


def build_server(tmp_path, reply: str = "hello from vela", api_key: str = "") -> VelaServer:
    """A server whose agent always answers with a fixed string."""
    instrumentor = Instrumentor()

    def factory():
        return AgentLoop(
            Scripted(reply),
            ToolRegistry([Echo()]),
            SessionStore(tmp_path / "sessions"),
            workspace=str(tmp_path),
            instrumentor=instrumentor,
        )

    server = VelaServer(
        factory,
        ServerConfig(host="127.0.0.1", port=0, api_key=api_key),
        instrumentor=instrumentor,
    )
    return server


@pytest.fixture
def client(tmp_path):
    server = build_server(tmp_path)
    with TestClient(server.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inflight_count_returns_to_zero(tmp_path) -> None:
    """A counter that only ever grows is worse than none.

    It is read from inside a request, so while /health is being served the
    count includes that very request; what matters is that it settles back to
    zero once the traffic stops.
    """
    server = build_server(tmp_path)
    with TestClient(server.app) as client:
        for _ in range(3):
            client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
            )
        assert server.inflight == 0
        # The health request counts itself, so one is expected while it runs.
        assert client.get("/health").json()["inflight"] == 1
        assert server.inflight == 0


def test_models_endpoint(client) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["object"] == "model"


def test_chat_completion_returns_a_reply(client) -> None:
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello from vela"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert "usage" in payload


def test_openai_shape_is_preserved(client) -> None:
    """A client written against the OpenAI SDK must find what it expects."""
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    payload = response.json()
    for field in ("id", "object", "created", "model", "choices", "usage"):
        assert field in payload, field
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert field in payload["usage"], field


def test_metrics_endpoint_exposes_counters(client) -> None:
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "vela_counter" in response.text
    # The turn must actually have been counted.
    assert "turns_total" in response.text


def test_streaming_returns_sse_chunks(tmp_path) -> None:
    server = build_server(tmp_path, reply="streamed reply")
    with TestClient(server.app) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

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


def test_streaming_fragments_arrive_one_by_one(tmp_path) -> None:
    """A single blob would pass a content check while defeating streaming."""
    server = build_server(tmp_path, reply="abcdefghij")
    with TestClient(server.app) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        raw = "".join(response.iter_text())

    chunks = [
        json.loads(line[len("data: "):])
        for line in raw.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    deltas = [c["choices"][0]["delta"].get("content", "") for c in chunks]
    assert len([d for d in deltas if d]) > 1, deltas


# --------------------------------------------------------------------------
# Validation, handled by the framework
# --------------------------------------------------------------------------


def test_missing_messages_is_rejected(client) -> None:
    response = client.post("/v1/chat/completions", json={"model": "x"})
    assert response.status_code == 422


def test_empty_messages_is_rejected(client) -> None:
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422


def test_unknown_role_is_rejected(client) -> None:
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "tool", "content": "x"}]}
    )
    assert response.status_code == 422


def test_empty_content_is_rejected(client) -> None:
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "   "}]}
    )
    assert response.status_code == 422


def test_malformed_body_is_rejected(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_block_content_is_accepted(client) -> None:
    """OpenAI permits multimodal blocks; rejecting them breaks existing clients."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at "},
                        {"type": "image_url", "image_url": {"url": "https://example/x.png"}},
                        {"type": "text", "text": "this"},
                    ],
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello from vela"


def test_unknown_endpoint_returns_404(client) -> None:
    response = client.get("/nope")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def test_api_key_is_enforced_when_configured(tmp_path) -> None:
    server = build_server(tmp_path, api_key="secret-token")
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 401

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200


def test_open_endpoint_allows_anonymous_calls(client) -> None:
    """Loopback with no token is a normal local setup."""
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_concurrent_requests_are_served(tmp_path) -> None:
    """Overlapping calls must not interfere with each other."""
    import threading

    server = build_server(tmp_path)
    results: list[int] = []
    errors: list[str] = []
    lock = threading.Lock()

    def call() -> None:
        try:
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
            with lock:
                results.append(response.status_code)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert results == [200] * 6


# --------------------------------------------------------------------------
# OpenAPI, which the framework provides
# --------------------------------------------------------------------------


def test_openapi_schema_is_served(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/v1/chat/completions" in schema["paths"]
    assert "post" in schema["paths"]["/v1/chat/completions"]


def test_interactive_docs_are_served(client) -> None:
    """A browsable API page is a concrete answer to 'how do I call this?'."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_finish_reason_mapping() -> None:
    assert _finish_reason("completed") == "stop"
    assert _finish_reason("max_iterations") == "length"
    assert _finish_reason("error") == "stop"
    assert _finish_reason("something_new") == "stop"


#: Import hook that hides the web stack, so the optional-dependency guarantee
#: can be checked on any machine regardless of what happens to be installed.
_BLOCK_FASTAPI = """
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('fastapi', 'pydantic', 'starlette'):
            raise ImportError('blocked for the test')
        return None


sys.meta_path.insert(0, Blocker())
"""


def _run_without_web_stack(body: str):
    """Run a snippet in a subprocess where the web stack cannot be imported."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    source_root = Path(__file__).resolve().parent.parent / "src"
    env = {**os.environ, "PYTHONPATH": str(source_root)}
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_FASTAPI + body],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_core_works_without_fastapi_installed() -> None:
    """The runtime must not require the web stack.

    FastAPI is an optional dependency, so the core package stays installable
    and usable on its own. This is the property that makes the split worth
    having; without it, the extra would be a tax on every embedder.
    """
    result = _run_without_web_stack(
        "import vela"
        + chr(10)
        + "assert vela.available_providers()"
        + chr(10)
        + "print('core-ok')"
        + chr(10)
    )
    assert "core-ok" in result.stdout, result.stderr[-500:]


def test_server_module_explains_a_missing_extra() -> None:
    """A missing optional dependency should say how to install it."""
    result = _run_without_web_stack(
        "try:"
        + chr(10)
        + "    import vela.server"
        + chr(10)
        + "except ImportError as exc:"
        + chr(10)
        + "    print('hint-ok' if 'vela[server]' in str(exc) else 'no-hint')"
        + chr(10)
    )
    assert "hint-ok" in result.stdout, result.stdout + result.stderr[-500:]


def test_agent_runs_without_the_server() -> None:
    """The HTTP layer is additive; the library API must remain sufficient."""
    import asyncio

    from vela.core.runner import AgentRunner, RunSpec, StreamingEmitter

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
