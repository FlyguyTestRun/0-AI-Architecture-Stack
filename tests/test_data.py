"""Tests for the data layer (layer 7)."""

from __future__ import annotations

from zerostack.data.store import RunRecord, StateStore, analytics_summary


def make_record(**overrides) -> RunRecord:
    defaults = dict(
        question="How much can an agent refund?",
        answer="Two hundred dollars.",
        orchestrator="simple",
        llm_provider="offline",
        llm_model="offline-extractive",
        latency_ms=12.5,
        sources=["support.md"],
        steps=[{"name": "plan", "detail": "route", "data": {}}],
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


class TestStateStore:
    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "state.db"
        StateStore(path)
        StateStore(path)
        assert StateStore(path).count_runs() == 0

    def test_saves_and_reads_back_a_run(self, tmp_path):
        store = StateStore(tmp_path / "state.db")
        run_id = store.save_run(make_record())

        record = store.get_run(run_id)
        assert record is not None
        assert record["question"] == "How much can an agent refund?"
        # JSON columns must come back as Python structures, not strings.
        assert record["sources"] == ["support.md"]
        assert record["steps"][0]["name"] == "plan"

    def test_missing_run_returns_none(self, tmp_path):
        assert StateStore(tmp_path / "state.db").get_run("nope") is None

    def test_recent_runs_are_newest_first_and_limited(self, tmp_path):
        store = StateStore(tmp_path / "state.db")
        for index in range(5):
            store.save_run(
                make_record(
                    question=f"question {index}",
                    created_at=f"2026-08-2{index}T10:00:00+00:00",
                )
            )

        recent = store.recent_runs(limit=3)
        assert len(recent) == 3
        assert recent[0]["question"] == "question 4"
        assert store.count_runs() == 5

    def test_in_memory_database_survives_across_calls(self):
        store = StateStore(":memory:")
        store.save_run(make_record())
        assert store.count_runs() == 1


class TestAnalytics:
    def test_reports_nothing_when_there_is_no_database(self, tmp_path):
        assert analytics_summary(tmp_path / "absent.db")["runs"] == 0

    def test_aggregates_by_backend(self, tmp_path):
        path = tmp_path / "state.db"
        store = StateStore(path)
        store.save_run(make_record(latency_ms=10.0))
        store.save_run(make_record(latency_ms=30.0))
        store.save_run(make_record(orchestrator="langgraph", latency_ms=50.0))

        summary = analytics_summary(path)
        assert summary["runs"] == 3
        # The engine may be duckdb or sqlite depending on what is installed and
        # whether the extension can be fetched. Both must produce the same shape.
        assert summary["engine"] in {"duckdb", "sqlite"}

        simple = next(row for row in summary["by_backend"] if row["orchestrator"] == "simple")
        assert simple["runs"] == 2
        assert simple["avg_latency_ms"] == 20.0
        assert simple["max_latency_ms"] == 30.0
