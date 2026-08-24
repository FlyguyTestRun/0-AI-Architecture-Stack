"""Minimal tracing that works with no collector running.

Every span is appended to a JSONL file so a developer can inspect a run without
standing up Phoenix. When ``export_traces`` is enabled and the OpenTelemetry
packages are installed, spans are additionally exported to a Phoenix collector.

The tracer is intentionally tiny. The point of the observability layer in this
baseline is that every LLM call, retrieval and tool invocation is recorded by
default, not that the tracer is feature complete.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zerostack.config import ObservabilitySettings, get_settings

logger = logging.getLogger("zerostack")

_LOGGING_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    resolved = level or get_settings().observability.log_level
    logging.basicConfig(
        level=getattr(logging, resolved.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGING_CONFIGURED = True


@dataclass
class Span:
    """A single unit of traced work."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return round((end - self.start_time) * 1000, 2)

    def set(self, **attributes: Any) -> Span:
        """Attach attributes to the span. Returns self so calls can be chained."""
        self.attributes.update(attributes)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "attributes": self.attributes,
        }


class _TraceState(threading.local):
    """Per thread trace id and span stack.

    The tracer is a process wide singleton, but the API serves each request on a
    worker thread. Holding the current trace id and the open span stack on the
    instance would let concurrent requests overwrite each other's trace id and
    parent spans, which silently misattributes spans to the wrong request and
    stores the wrong trace_id on the run record. Thread local state keeps each
    request's trace independent.
    """

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.stack: list[Span] = []


class Tracer:
    """Records spans to a JSONL file and optionally to a Phoenix collector."""

    def __init__(self, settings: ObservabilitySettings | None = None) -> None:
        self.settings = settings or get_settings().observability
        # Bounded so a long running server cannot grow this without limit. The
        # JSONL log is the durable record; this buffer only serves the UI.
        self.spans: deque[Span] = deque(maxlen=self.settings.max_retained_spans)
        self._state = _TraceState()
        self._lock = threading.Lock()
        self._otel_tracer = self._maybe_build_otel_tracer()

    def _maybe_build_otel_tracer(self) -> Any | None:
        """Build an OTLP exporter only if enabled and the packages are present."""
        if not (self.settings.enabled and self.settings.export_traces):
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            logger.debug("OpenTelemetry not installed, tracing to file only")
            return None

        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=self.settings.phoenix_endpoint))
        )
        trace.set_tracer_provider(provider)
        return trace.get_tracer("zerostack")

    @property
    def trace_id(self) -> str:
        """The trace id for the calling thread."""
        return self._state.trace_id

    def new_trace(self) -> str:
        """Start a fresh trace id for the calling thread. Call once per request."""
        self._state.trace_id = uuid.uuid4().hex
        self._state.stack.clear()
        return self._state.trace_id

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Record a span around a block of work."""
        stack = self._state.stack
        span = Span(
            name=name,
            trace_id=self._state.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=stack[-1].span_id if stack else None,
            attributes=dict(attributes),
        )
        stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_time = time.time()
            stack.pop()
            with self._lock:
                self.spans.append(span)
            self._emit(span)

    def _emit(self, span: Span) -> None:
        if not self.settings.enabled:
            return
        logger.debug("span %s finished in %sms", span.name, span.duration_ms)
        self._write_jsonl(span, self.settings.trace_log_path)
        if self._otel_tracer is not None:
            self._export_otel(span)

    @staticmethod
    def _write_jsonl(span: Span, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(span.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.warning("could not write trace to %s: %s", path, exc)

    def _export_otel(self, span: Span) -> None:
        try:
            with self._otel_tracer.start_as_current_span(span.name) as otel_span:
                for key, value in span.attributes.items():
                    otel_span.set_attribute(key, str(value))
                otel_span.set_attribute("duration_ms", span.duration_ms)
        except Exception as exc:
            logger.warning("trace export failed: %s", exc)

    def summary(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Return recorded spans as plain dictionaries.

        Defaults to the calling thread's current trace so a caller asking for
        "the last run" does not receive every span the process has ever emitted.
        Pass ``trace_id=""`` to get the whole retained buffer.
        """
        wanted = self._state.trace_id if trace_id is None else trace_id
        with self._lock:
            spans = list(self.spans)
        if not wanted:
            return [span.to_dict() for span in spans]
        return [span.to_dict() for span in spans if span.trace_id == wanted]


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    """Return the process wide tracer."""
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer()
    return _TRACER
