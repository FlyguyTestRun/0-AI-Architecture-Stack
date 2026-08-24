"""Tests for the observability layer.

The concurrency tests exist because the API serves each request on a worker
thread while the tracer is a process wide singleton. An earlier version held the
current trace id and the open span stack on the instance, so two simultaneous
requests overwrote each other's trace and parent span. That misattributes spans
and stores the wrong trace_id on the run record, which is worse than no tracing
because it looks correct.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from zerostack.config import ObservabilitySettings
from zerostack.observability.tracing import Span, Tracer


@pytest.fixture
def tracer(tmp_path) -> Tracer:
    return Tracer(
        ObservabilitySettings(
            enabled=False, trace_log_path=tmp_path / "traces.jsonl", max_retained_spans=1000
        )
    )


class TestSpan:
    def test_duration_is_measured(self, tracer):
        with tracer.span("work") as span:
            time.sleep(0.01)
        assert span.duration_ms >= 10

    def test_attributes_can_be_attached(self, tracer):
        with tracer.span("work") as span:
            span.set(hits=3, source="a.md")
        assert span.attributes == {"hits": 3, "source": "a.md"}

    def test_error_is_recorded_and_reraised(self, tracer):
        with pytest.raises(ValueError):
            with tracer.span("boom"):
                raise ValueError("bad")
        assert tracer.spans[-1].error == "ValueError: bad"

    def test_nesting_sets_parent(self, tracer):
        with tracer.span("outer") as outer:
            with tracer.span("inner") as inner:
                pass
        assert inner.parent_id == outer.span_id
        assert outer.parent_id is None

    def test_stack_unwinds_after_an_error(self, tracer):
        """A failed span must not leave itself on the stack as a phantom parent."""
        with pytest.raises(ValueError):
            with tracer.span("boom"):
                raise ValueError("bad")
        with tracer.span("after") as after:
            pass
        assert after.parent_id is None

    def test_span_is_json_serialisable(self, tracer):
        with tracer.span("work") as span:
            span.set(sources=["a.md"])
        json.dumps(span.to_dict(), default=str)


class TestConcurrency:
    def test_traces_do_not_leak_between_threads(self, tracer):
        """Two simultaneous requests must not share a trace id."""
        recorded: dict[str, tuple[str, str]] = {}
        barrier = threading.Barrier(2)

        def request(name: str) -> None:
            trace_id = tracer.new_trace()
            barrier.wait()  # force the two requests to interleave
            with tracer.span(f"{name}.run"):
                time.sleep(0.02)
                with tracer.span(f"{name}.inner") as span:
                    recorded[name] = (trace_id, span.trace_id)
                time.sleep(0.02)

        threads = [threading.Thread(target=request, args=(n,)) for n in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for name, (started, observed) in recorded.items():
            assert started == observed, f"{name} recorded another request's trace id"
        assert recorded["a"][0] != recorded["b"][0]

    def test_parent_spans_do_not_leak_between_threads(self, tracer):
        parents: dict[str, str | None] = {}
        barrier = threading.Barrier(2)

        def request(name: str) -> None:
            tracer.new_trace()
            barrier.wait()
            with tracer.span(f"{name}.run"):
                time.sleep(0.02)
                with tracer.span(f"{name}.inner") as span:
                    parents[name] = span.parent_id

        threads = [threading.Thread(target=request, args=(n,)) for n in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(set(parents.values())) == 2, "nested spans shared one parent"

    def test_concurrent_spans_are_all_retained(self, tracer):
        def emit() -> None:
            for _ in range(50):
                with tracer.span("x"):
                    pass

        threads = [threading.Thread(target=emit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(tracer.spans) == 200


class TestRetention:
    def test_span_buffer_is_bounded(self, tmp_path):
        """A long running server must not grow this buffer without limit."""
        tracer = Tracer(
            ObservabilitySettings(
                enabled=False,
                trace_log_path=tmp_path / "t.jsonl",
                max_retained_spans=50,
            )
        )
        for _ in range(500):
            with tracer.span("x"):
                pass
        assert len(tracer.spans) == 50

    def test_summary_defaults_to_the_current_trace(self, tracer):
        tracer.new_trace()
        with tracer.span("first"):
            pass
        tracer.new_trace()
        with tracer.span("second"):
            pass
        assert [s["name"] for s in tracer.summary()] == ["second"]

    def test_summary_accepts_an_explicit_trace(self, tracer):
        first = tracer.new_trace()
        with tracer.span("first"):
            pass
        tracer.new_trace()
        with tracer.span("second"):
            pass
        assert [s["name"] for s in tracer.summary(first)] == ["first"]

    def test_summary_of_empty_string_returns_everything(self, tracer):
        tracer.new_trace()
        with tracer.span("first"):
            pass
        tracer.new_trace()
        with tracer.span("second"):
            pass
        assert [s["name"] for s in tracer.summary("")] == ["first", "second"]


class TestFileOutput:
    def test_spans_are_written_as_jsonl(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        tracer = Tracer(ObservabilitySettings(enabled=True, trace_log_path=path))
        with tracer.span("work") as span:
            span.set(hits=2)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["name"] == "work"

    def test_an_unwritable_path_does_not_break_the_run(self, tmp_path):
        """Tracing is bookkeeping. It must never fail the work it is measuring."""
        blocked = tmp_path / "file.txt"
        blocked.write_text("not a directory", encoding="utf-8")
        tracer = Tracer(
            ObservabilitySettings(enabled=True, trace_log_path=blocked / "traces.jsonl")
        )
        with tracer.span("work") as span:
            span.set(ok=True)
        assert isinstance(span, Span)
