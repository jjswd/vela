"""Observability: metrics and structured tracing for the runtime.

Two problems this solves. Metrics answer "is the system healthy" in aggregate,
which logs cannot: the recovery paths in this runtime are designed to keep
requests succeeding through a misbehaving model, so a run of quietly recovered
turns is invisible without counters. Traces answer "why was this one turn slow",
which counters cannot.

Deliberately dependency-free. A runtime whose whole value proposition is being
small should not pull in an SDK to report numbers, so metrics accumulate in
process and expose as plain text, and a trace is a list of spans that any
exporter can consume.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


class Metrics:
    """In-process counters, timers, and gauges.

    Counters are keyed by label sets rather than pre-declared names, so adding a
    tool or a backend does not require declaring a metric first. The trade-off is
    that nothing stops a caller from inventing unbounded label values, so
    label values are normalised and unknown values collapse into "other" after
    ``MAX_LABEL_VALUES`` distinct entries.
    """

    #: Distinct values kept per label before collapsing into "other".
    MAX_LABEL_VALUES = 40

    def __init__(self) -> None:
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._timers: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._label_pools: dict[str, set[str]] = defaultdict(set)

    def _key(self, name: str, labels: dict[str, str] | None) -> tuple:
        if not labels:
            return (name, ())
        normalised = []
        for key in sorted(labels, key=lambda item: (item, str(labels[item]))):
            value = str(labels[key])
            # Bound distinct values *per metric and per label name*. Sharing one
            # pool across metrics would let a high-cardinality series starve the
            # others into "other".
            pool = self._label_pools[f"{name}.{key}"]
            if value not in pool:
                if len(pool) >= self.MAX_LABEL_VALUES:
                    value = "other"
                else:
                    pool.add(value)
            normalised.append((key, value))
        return (name, tuple(normalised))

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        self._counters[self._key(name, labels)] += amount

    def observe(self, name: str, seconds: float, **labels: str) -> None:
        # Stored under the same tuple key shape as counters so the snapshot and
        # Prometheus renderers can treat both uniformly. Keeping the structured
        # key (rather than a formatted string) means labels survive to output.
        key = self._key(name, labels)
        # Bound memory: a long-lived process must not accumulate every
        # individual duration forever.
        samples = self._timers[key]
        samples.append(seconds)
        if len(samples) > 10_000:
            del samples[: len(samples) // 2]

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    @property
    def counter_values(self) -> dict[tuple[str, tuple], int]:
        return dict(self._counters)

    def snapshot(self) -> dict[str, Any]:
        """A structured view suitable for logging or exporting."""
        counters: dict[str, int] = defaultdict(int)
        for (name, labels), value in self._counters.items():
            suffix = ",".join(f"{k}={v}" for k, v in labels)
            counters[f"{name}{{{suffix}}}" if suffix else name] += value

        timers: dict[str, dict[str, float]] = {}
        for (name, label_pairs), samples in self._timers.items():
            if not samples:
                continue
            ordered = sorted(samples)
            labels = ",".join(f"{k}={v}" for k, v in label_pairs)
            label = f"{name}{{{labels}}}" if labels else name
            timers[label] = {
                "count": len(ordered),
                "sum_s": round(sum(ordered), 4),
                "avg_ms": round(sum(ordered) / len(ordered) * 1000, 3),
                "p50_ms": round(_percentile(ordered, 0.50) * 1000, 3),
                "p95_ms": round(_percentile(ordered, 0.95) * 1000, 3),
                "p99_ms": round(_percentile(ordered, 0.99) * 1000, 3),
                "max_ms": round(ordered[-1] * 1000, 3),
            }

        return {"counters": dict(counters), "timers": timers, "gauges": dict(self._gauges)}

    def to_prometheus(self) -> str:
        """Render in Prometheus text exposition format.

        Exposing the standard format means this integrates with any existing
        dashboard without adding a client library.
        """
        lines: list[str] = []
        for (name, labels), value in self._counters.items():
            label_text = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f'vela_counter{{name="{name}"{"," if label_text else ""}{label_text}}} {value}')
        for (name, _label_pairs), samples in self._timers.items():
            if not samples:
                continue
            ordered = sorted(samples)
            for statistic, value in (
                ("count", len(ordered)),
                ("sum", round(sum(ordered), 4)),
                ("avg", round(sum(ordered) / len(ordered), 4)),
                ("p95", round(_percentile(ordered, 0.95), 4)),
                ("p99", round(_percentile(ordered, 0.99), 4)),
            ):
                lines.append(f'vela_timer_seconds{{name="{name}",stat="{statistic}"}} {value}')
        for name, value in self._gauges.items():
            lines.append(f'vela_gauge{{name="{name}"}} {value}')
        return "\n".join(lines)

    def reset(self) -> None:
        self._counters.clear()
        self._timers.clear()
        self._gauges.clear()
        self._label_pools.clear()


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------


@dataclass
class Span:
    """One timed operation."""

    name: str
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    #: (span name, message) pairs marking points of interest inside a span.
    events: list[tuple[str, str]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms is not None else None,
            "attributes": self.attributes,
            "events": [{"name": name, "message": message} for name, message in self.events],
        }


class Tracer:
    """Collects spans for one turn, with nesting by name.

    A turn produces a tree of work: the model call, each tool, each recovery
    path. Flattening that loses the part that explains a slow turn, which is
    which child dominated. Spans therefore carry their parent and render as a
    tree.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    def span(self, name: str, **attributes: Any) -> _SpanContext:
        if not self.enabled:
            return _SpanContext(None, self)
        span = Span(name=name, started_at=time.perf_counter(), attributes=dict(attributes))
        if self._stack:
            span.attributes["parent"] = self._stack[-1].name
        return _SpanContext(span, self)

    def _push(self, span: Span) -> None:
        self.spans.append(span)
        self._stack.append(span)

    def _pop(self) -> None:
        if self._stack:
            self._stack.pop()

    def event(self, name: str, message: str = "") -> None:
        """Mark a point of interest on the innermost open span."""
        if self._stack:
            self._stack[-1].events.append((name, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [span.to_dict() for span in self.spans],
            "total_ms": round(
                sum(s.duration_ms or 0.0 for s in self.spans if not s.attributes.get("parent")),
                2,
            ),
        }

    def to_text(self) -> str:
        """Render as an indented tree, for logs and terminal output."""
        if not self.spans:
            return "(no spans)"
        by_name: dict[str, list[Span]] = defaultdict(list)
        roots: list[Span] = []
        for span in self.spans:
            parent = span.attributes.get("parent")
            if parent is None:
                roots.append(span)
            else:
                by_name[parent].append(span)

        lines: list[str] = []

        def render(span: Span, depth: int) -> None:
            duration = span.duration_ms or 0.0
            detail = " ".join(f"{k}={v}" for k, v in span.attributes.items() if k != "parent")
            lines.append(
                f"{'  ' * depth}{'└─ ' if depth else ''}{span.name} ({duration:.1f}ms)"
                + (f"  {detail}" if detail else "")
            )
            for name, message in span.events:
                lines.append(f"{'  ' * (depth + 1)}· {name}: {message}")
            for child in by_name.get(span.name, []):
                render(child, depth + 1)

        for root in roots:
            render(root, 0)
        return "\n".join(lines)


class _SpanContext:
    """Context manager returned by :meth:`Tracer.span`."""

    def __init__(self, span: Span | None, tracer: Tracer) -> None:
        self._span = span
        self._tracer = tracer

    def __enter__(self) -> Span | None:
        if self._span is not None:
            self._tracer._push(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._span is not None:
            self._span.ended_at = time.perf_counter()
            if exc is not None:
                self._span.events.append(("error", f"{type(exc).__name__}: {exc}"))
            self._tracer._pop()
        return False


# --------------------------------------------------------------------------
# Runtime instrumentation
# --------------------------------------------------------------------------


class Instrumentor:
    """Wires metrics and tracing into a running loop.

    Attached rather than hard-wired so the runtime does not depend on being
    observed: with no instrumentor attached the hooks below are no-ops, and the
    same code runs unchanged in a script, a test, and a service.
    """

    def __init__(self, metrics: Metrics | None = None) -> None:
        self.metrics = metrics or Metrics()

    def record_turn(
        self,
        *,
        session: str,
        result: Any,
        duration_s: float,
        tracer: Tracer | None = None,
    ) -> None:
        """Summarise a finished turn.

        Stop reason and tool count are the two dimensions worth alerting on: a
        spike in ``max_iterations`` means the model is not converging, and tool
        errors mean a capability is broken.
        """
        metrics = self.metrics
        metrics.increment("turns_total")
        metrics.observe("turn_duration_seconds", duration_s)
        metrics.increment("turn_stop_reason", reason=getattr(result, "stop_reason", "unknown"))

        tools = list(getattr(result, "tools_used", []) or [])
        if tools:
            metrics.increment("turns_with_tools")
            for name in set(tools):
                metrics.increment("tool_calls_total", tool=name)

        error = getattr(result, "error", "") or ""
        if error:
            metrics.increment("turn_errors_total")

        usage = getattr(result, "usage", None)
        if usage is not None:
            metrics.gauge("tokens_input_last", usage.input_tokens)
            metrics.gauge("tokens_output_last", usage.output_tokens)
            metrics.increment("tokens_input_total", amount=usage.input_tokens)
            metrics.increment("tokens_output_total", amount=usage.output_tokens)

        if tracer is not None and tracer.enabled:
            logger.info(
                "turn complete session=%s stop=%s tools=%s duration=%.0fms\n%s",
                session,
                getattr(result, "stop_reason", "?"),
                ",".join(tools) or "-",
                duration_s * 1000,
                tracer.to_text(),
            )
