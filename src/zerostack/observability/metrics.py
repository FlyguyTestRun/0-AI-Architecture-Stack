"""Metrics: counters, gauges and histograms, with a Prometheus text endpoint.

Traces answer "what happened in this one request". Metrics answer "what is
happening across all of them", which is the question an operator actually has at
three in the morning. The two are complements, not alternatives, and a system
with only traces cannot tell you that latency doubled an hour ago.

This is a small in process registry rather than a client library, for the same
reason the rest of the stack ships offline implementations: metrics must work
before anyone has stood up a metrics backend. The exposition format is the
Prometheus text format, so any standard scraper can read it when one appears.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

# Buckets in seconds, chosen to straddle the latencies this stack actually
# produces: sub millisecond for the offline path, hundreds of milliseconds to
# tens of seconds for a local model server.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

Labels = tuple[tuple[str, str], ...]


def _freeze(labels: dict[str, str] | None) -> Labels:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _render_labels(labels: Labels, extra: dict[str, str] | None = None) -> str:
    pairs = dict(labels)
    if extra:
        pairs.update(extra)
    if not pairs:
        return ""
    inner = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(pairs.items()))
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class Histogram:
    """Cumulative bucket counts, the shape Prometheus expects."""

    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[float, int] = field(default_factory=dict)
    total: int = 0
    sum_value: float = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum_value += value
        for bound in self.buckets:
            if value <= bound:
                self.counts[bound] = self.counts.get(bound, 0) + 1

    @property
    def average(self) -> float:
        return self.sum_value / self.total if self.total else 0.0

    def quantile(self, q: float) -> float:
        """Approximate a quantile from the bucket bounds.

        Bucketed data cannot give an exact quantile, only the bound of the bucket
        the quantile falls in. That is the normal trade for cheap histograms and
        is accurate enough to answer "did the tail get worse".
        """
        if not self.total:
            return 0.0
        target = q * self.total
        cumulative = 0
        for bound in self.buckets:
            cumulative = self.counts.get(bound, 0)
            if cumulative >= target:
                return bound
        return self.buckets[-1]


class MetricsRegistry:
    """Process wide metric storage. Safe to use from request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, Labels], float] = defaultdict(float)
        self._gauges: dict[tuple[str, Labels], float] = {}
        self._histograms: dict[tuple[str, Labels], Histogram] = {}
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}

    def describe(self, name: str, help_text: str, metric_type: str) -> None:
        self._help[name] = help_text
        self._types[name] = metric_type

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        with self._lock:
            self._counters[(name, _freeze(labels))] += value
            self._types.setdefault(name, "counter")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[(name, _freeze(labels))] = value
            self._types.setdefault(name, "gauge")

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = (name, _freeze(labels))
            if key not in self._histograms:
                self._histograms[key] = Histogram()
            self._histograms[key].observe(value)
            self._types.setdefault(name, "histogram")

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
        """Observe the duration of a block, including when it raises.

        Timing only the success path hides exactly the latency an operator is
        looking for, because a request that fails slowly is the interesting one.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, labels)

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._counters.get((name, _freeze(labels)), 0.0)

    def gauge_value(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        with self._lock:
            return self._gauges.get((name, _freeze(labels)))

    def histogram(self, name: str, labels: dict[str, str] | None = None) -> Histogram | None:
        with self._lock:
            return self._histograms.get((name, _freeze(labels)))

    def snapshot(self) -> dict[str, object]:
        """A plain dictionary view, for the health endpoint and the UI."""
        with self._lock:
            counters = {
                f"{name}{_render_labels(labels)}": value
                for (name, labels), value in sorted(self._counters.items())
            }
            gauges = {
                f"{name}{_render_labels(labels)}": value
                for (name, labels), value in sorted(self._gauges.items())
            }
            histograms = {
                f"{name}{_render_labels(labels)}": {
                    "count": histogram.total,
                    "sum": round(histogram.sum_value, 6),
                    "avg": round(histogram.average, 6),
                    "p50": histogram.quantile(0.50),
                    "p95": histogram.quantile(0.95),
                    "p99": histogram.quantile(0.99),
                }
                for (name, labels), histogram in sorted(self._histograms.items())
            }
        return {"counters": counters, "gauges": gauges, "histograms": histograms}

    def render_prometheus(self) -> str:
        """Render the registry in the Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            emitted: set[str] = set()

            def header(name: str) -> None:
                if name in emitted:
                    return
                emitted.add(name)
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {self._types.get(name, 'untyped')}")

            for (name, labels), value in sorted(self._counters.items()):
                header(name)
                lines.append(f"{name}{_render_labels(labels)} {value}")

            for (name, labels), value in sorted(self._gauges.items()):
                header(name)
                lines.append(f"{name}{_render_labels(labels)} {value}")

            for (name, labels), histogram in sorted(self._histograms.items()):
                header(name)
                cumulative = 0
                for bound in histogram.buckets:
                    cumulative = max(cumulative, histogram.counts.get(bound, 0))
                    bucket_labels = _render_labels(labels, {"le": _format_bound(bound)})
                    lines.append(f"{name}_bucket{bucket_labels} {cumulative}")
                lines.append(
                    f"{name}_bucket{_render_labels(labels, {'le': '+Inf'})} {histogram.total}"
                )
                lines.append(f"{name}_sum{_render_labels(labels)} {histogram.sum_value}")
                lines.append(f"{name}_count{_render_labels(labels)} {histogram.total}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


def _format_bound(bound: float) -> str:
    return str(int(bound)) if bound == int(bound) else str(bound)


_REGISTRY: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Return the process wide registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = MetricsRegistry()
        _register_defaults(_REGISTRY)
    return _REGISTRY


def _register_defaults(registry: MetricsRegistry) -> None:
    registry.describe("zerostack_requests_total", "Agent runs started.", "counter")
    registry.describe("zerostack_request_errors_total", "Agent runs that raised.", "counter")
    registry.describe("zerostack_request_seconds", "End to end agent run latency.", "histogram")
    registry.describe("zerostack_retrieval_seconds", "Retrieval latency.", "histogram")
    registry.describe("zerostack_llm_seconds", "Model call latency.", "histogram")
    registry.describe("zerostack_llm_tokens_total", "Tokens consumed, by direction.", "counter")
    registry.describe("zerostack_llm_cost_usd_total", "Estimated model spend.", "counter")
    registry.describe("zerostack_cache_events_total", "Semantic cache hits and misses.", "counter")
    registry.describe("zerostack_documents_indexed", "Chunks currently indexed.", "gauge")
    registry.describe("zerostack_tool_calls_total", "Tool invocations, by outcome.", "counter")
