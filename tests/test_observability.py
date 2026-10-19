"""Tests for metrics and tracing.

The recovery paths exist to keep a turn succeeding through a misbehaving model,
which means a run of quietly recovered turns is invisible unless it is counted.
These tests pin down that the counting actually happens, since a metric that
only fires on the happy path is worse than no metric: it looks healthy.
"""

from __future__ import annotations

import json

import pytest

from vela.core.contracts import ModelResponse, ToolCall, Usage
from vela.core.loop import AgentLoop, SessionStore
from vela.core.runner import AgentRunner, RunSpec, StopReason
from vela.observability import Instrumentor, Metrics, Tracer
from vela.providers.base import GenerationConfig, Provider
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


class Echo(Tool):
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, arguments: dict) -> str:
        return str(arguments.get("text", ""))


class Strict(Tool):
    """Requires an argument, so calling it without one exercises the error path."""

    name = "strict"
    description = "requires text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
    }

    async def run(self, arguments: dict) -> str:
        return str(arguments["text"])


class Scripted(Provider):
    def __init__(self, script, finalizer=None):
        super().__init__(GenerationConfig(model="test"))
        self.script = list(script)
        self.finalizer = finalizer

    async def complete(self, messages, tools, *, system="", stream=False, on_delta=None):
        if self.finalizer is not None and not tools:
            # A tool-free request is the post-budget finalisation. Returning the
            # string itself would violate the provider contract.
            text, self.finalizer = self.finalizer, None
            return ModelResponse(content=text, finish_reason="stop")
        if not self.script:
            return ModelResponse(content="exhausted", finish_reason="stop")
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    async def aclose(self) -> None:
        return None


def value(metrics: Metrics, name: str, **labels: str) -> int:
    key = (name, tuple(sorted(labels.items())))
    return metrics.counter_values.get(key, 0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_counters_accumulate_by_label() -> None:
    metrics = Metrics()
    metrics.increment("tool_calls_total", tool="search")
    metrics.increment("tool_calls_total", tool="search")
    metrics.increment("tool_calls_total", tool="read")

    assert value(metrics, "tool_calls_total", tool="search") == 2
    assert value(metrics, "tool_calls_total", tool="read") == 1


def test_unlabelled_and_labelled_counters_stay_separate() -> None:
    metrics = Metrics()
    metrics.increment("turns_total")
    metrics.increment("turns_total", reason="completed")
    assert value(metrics, "turns_total") == 1
    assert value(metrics, "turns_total", reason="completed") == 1


def test_label_values_are_bounded() -> None:
    """Unbounded label values would grow memory without limit in a long process."""
    metrics = Metrics()
    for index in range(Metrics.MAX_LABEL_VALUES + 50):
        metrics.increment("calls_total", session=f"session-{index}")

    sessions = {
        value
        for (name, labels), _count in metrics.counter_values.items()
        if name == "calls_total"
        for key, value in labels
        if key == "session"
    }
    assert "other" in sessions
    # Values past the cap collapse rather than accumulating.
    assert len(sessions) <= Metrics.MAX_LABEL_VALUES + 1
    assert len(metrics.counter_values) <= Metrics.MAX_LABEL_VALUES + 1


def test_label_pools_are_per_metric() -> None:
    """A high-cardinality series must not starve other metrics into "other"."""
    metrics = Metrics()
    for index in range(Metrics.MAX_LABEL_VALUES + 20):
        metrics.increment("noisy_total", session=f"s{index}")
    # A different metric keeps its own resolution.
    metrics.increment("quiet_total", session="only-one")
    sessions = {
        value
        for (name, labels), _count in metrics.counter_values.items()
        if name == "quiet_total"
        for _key, value in labels
    }
    assert sessions == {"only-one"}


def test_timers_report_percentiles() -> None:
    metrics = Metrics()
    for value_ms in (10, 20, 30, 40, 1000):
        metrics.observe("latency_seconds", value_ms / 1000)

    snapshot = metrics.snapshot()
    timer = snapshot["timers"]["latency_seconds"]
    assert timer["count"] == 5
    assert timer["p50_ms"] == pytest.approx(30.0, abs=1)
    assert timer["p95_ms"] == pytest.approx(1000.0, abs=1)
    assert timer["max_ms"] == pytest.approx(1000.0, abs=1)


def test_timer_samples_are_bounded() -> None:
    """A long-lived process must not accumulate every duration forever."""
    metrics = Metrics()
    for _ in range(10_050):
        metrics.observe("latency_seconds", 0.001)
    assert metrics.snapshot()["timers"]["latency_seconds"]["count"] <= 10_000


def test_prometheus_rendering_is_valid_text() -> None:
    metrics = Metrics()
    metrics.increment("tool_results_total", tool="search", status="ok")
    metrics.observe("latency_seconds", 0.05)

    text = metrics.to_prometheus()
    assert 'name="tool_results_total"' in text
    assert 'tool="search"' in text and 'status="ok"' in text
    assert text.count("vela_counter") == 1
    assert "vela_timer_seconds" in text
    # Every line must be metric value pairs, not prose.
    assert all(len(line.split()) == 2 for line in text.splitlines())


def test_snapshot_is_json_serialisable() -> None:
    metrics = Metrics()
    metrics.increment("a_total", k="v")
    metrics.observe("b_seconds", 0.1)
    metrics.gauge("c", 3)
    json.dumps(metrics.snapshot())


# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------


def test_spans_record_duration() -> None:
    tracer = Tracer()
    with tracer.span("work", label="x"):
        pass
    assert tracer.spans[0].duration_ms is not None
    assert tracer.spans[0].attributes["label"] == "x"


def test_nested_spans_render_as_a_tree() -> None:
    tracer = Tracer()
    with tracer.span("turn"):
        with tracer.span("model_request", iteration=0):
            pass
        with tracer.span("tool_batch", count=2):
            pass

    text = tracer.to_text()
    assert "turn" in text
    # Children are indented under the parent.
    assert any(line.startswith("  ") and "model_request" in line for line in text.splitlines())


def test_exception_is_recorded_on_the_span() -> None:
    tracer = Tracer()
    with pytest.raises(ValueError), tracer.span("failing"):
        raise ValueError("boom")

    events = tracer.spans[0].events
    assert any("boom" in message for _, message in events)


def test_disabled_tracer_records_nothing() -> None:
    tracer = Tracer(enabled=False)
    with tracer.span("ignored"):
        pass
    tracer.event("also_ignored")
    assert tracer.spans == []


def test_event_lands_on_the_innermost_span() -> None:
    tracer = Tracer()
    with tracer.span("outer"), tracer.span("inner"):
        tracer.event("note", "detail")
    inner = [s for s in tracer.spans if s.name == "inner"][0]
    assert inner.events == [("note", "detail")]


# --------------------------------------------------------------------------
# Instrumented runtime
# --------------------------------------------------------------------------


async def test_turn_records_stop_reason_and_tokens() -> None:
    metrics = Metrics()
    runner = AgentRunner(
        Scripted([ModelResponse(content="done", finish_reason="stop", usage=Usage(30, 7))]),
        ToolRegistry([Echo()]),
        instrumentor=Instrumentor(metrics),
    )
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))

    # Turn-level counters belong to the loop; the runner contributes request
    # latency and tool results only. Both layers feeding one Metrics instance
    # is what makes a single scrape cover the whole request.
    assert value(metrics, "turns_total") == 0
    assert "model_request_seconds" in metrics.snapshot()["timers"]


async def test_model_request_latency_is_observed() -> None:
    metrics = Metrics()
    runner = AgentRunner(
        Scripted([ModelResponse(content="x", finish_reason="stop")]),
        ToolRegistry([Echo()]),
        instrumentor=Instrumentor(metrics),
    )
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert "model_request_seconds" in metrics.snapshot()["timers"]


async def test_tool_errors_are_counted_separately() -> None:
    """A tool failing repeatedly means the capability is broken, not bad luck."""
    metrics = Metrics()
    provider = Scripted(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="strict", arguments={}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="recovered", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(
        provider, ToolRegistry([Strict()]), instrumentor=Instrumentor(metrics)
    )
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))

    # 'text' is required, so validation rejected the call.
    assert value(metrics, "tool_results_total", tool="strict", status="error") == 1
    assert value(metrics, "tool_errors_total", tool="strict") == 1
    # A successful call must be counted separately from a failed one.
    assert value(metrics, "tool_results_total", tool="strict", status="ok") == 0


async def test_recovery_paths_are_counted() -> None:
    """Silent recovery is exactly the case a metric exists to expose."""
    metrics = Metrics()

    # Empty reply, then an answer.
    provider = Scripted(
        [
            ModelResponse(content="", finish_reason="stop"),
            ModelResponse(content="answered", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, ToolRegistry([Echo()]), instrumentor=Instrumentor(metrics))
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert value(metrics, "recovery_empty_response_total") == 1

    # Truncation recovery.
    metrics = Metrics()
    provider = Scripted(
        [
            ModelResponse(content="part", finish_reason="length"),
            ModelResponse(content=" rest", finish_reason="stop"),
        ]
    )
    runner = AgentRunner(provider, ToolRegistry([Echo()]), instrumentor=Instrumentor(metrics))
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert value(metrics, "recovery_truncation_total") == 1

    # Budget exhaustion.
    metrics = Metrics()
    provider = Scripted(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                finish_reason="tool_calls",
            )
        ],
        finalizer="summary",
    )
    runner = AgentRunner(provider, ToolRegistry([Echo()]), instrumentor=Instrumentor(metrics))
    result = await runner.run(
        RunSpec(messages=[{"role": "user", "content": "hi"}], max_iterations=2)
    )
    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert value(metrics, "recovery_budget_exhausted_total") == 1


async def test_model_errors_are_counted() -> None:
    metrics = Metrics()
    provider = Scripted(
        [ModelResponse(content="", finish_reason="error", error="upstream down")]
    )
    runner = AgentRunner(provider, ToolRegistry([Echo()]), instrumentor=Instrumentor(metrics))
    await runner.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert value(metrics, "model_errors_total") == 1


async def test_loop_records_the_whole_turn(tmp_path) -> None:
    metrics = Metrics()
    loop = AgentLoop(
        Scripted(
            [ModelResponse(content="hi", finish_reason="stop", usage=Usage(12, 4))]
        ),
        ToolRegistry([Echo()]),
        SessionStore(tmp_path),
        workspace=str(tmp_path),
        instrumentor=Instrumentor(metrics),
    )
    await loop.run("s", "hello", ephemeral=True)

    assert value(metrics, "turns_total") == 1
    assert value(metrics, "turn_stop_reason", reason="completed") == 1
    assert metrics.snapshot()["counters"]["tokens_input_total"] == 12
    await loop.aclose()


async def test_loop_tracing_produces_a_trace(tmp_path) -> None:
    loop = AgentLoop(
        Scripted(
            [
                ModelResponse(
                    content="",
                    tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "x"}),),
                    finish_reason="tool_calls",
                ),
                ModelResponse(content="done", finish_reason="stop"),
            ]
        ),
        ToolRegistry([Echo()]),
        SessionStore(tmp_path),
        workspace=str(tmp_path),
        trace=True,
    )
    await loop.run("s", "hi", ephemeral=True)

    names = [span.name for span in loop.last_trace.spans]
    assert "model_request" in names
    assert "tool_batch" in names
    assert "model_request" in loop.last_trace.to_text()
    await loop.aclose()


async def test_runtime_works_without_instrumentation(tmp_path) -> None:
    """Observability is optional; its absence must change no behaviour."""
    provider = Scripted([ModelResponse(content="ok", finish_reason="stop")])
    plain = AgentRunner(provider, ToolRegistry([Echo()]))
    result = await plain.run(RunSpec(messages=[{"role": "user", "content": "hi"}]))
    assert result.content == "ok"
    assert plain.instrumentor is None


async def test_metrics_survive_a_failing_turn(tmp_path) -> None:
    """A failed turn must still be recorded, or outages look like silence."""
    metrics = Metrics()
    loop = AgentLoop(
        Scripted([ModelResponse(content="", finish_reason="error", error="down")]),
        ToolRegistry([Echo()]),
        SessionStore(tmp_path),
        workspace=str(tmp_path),
        instrumentor=Instrumentor(metrics),
    )
    await loop.run("s", "hi", ephemeral=True)
    assert value(metrics, "turns_total") == 1
    assert value(metrics, "turn_errors_total") == 1
    assert value(metrics, "turn_stop_reason", reason="error") == 1
    await loop.aclose()
