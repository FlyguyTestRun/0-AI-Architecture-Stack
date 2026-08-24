"""End to end tests over the composed application.

These are the tests that would catch a regression in the promise the README makes:
a clean clone answers questions correctly with no Docker, no model server and no
network access.
"""

from __future__ import annotations

import json

import pytest

from zerostack.app import ZerostackApp
from zerostack.config import Settings
from zerostack.observability.tracing import Tracer


@pytest.fixture
def app(settings, corpus_dir) -> ZerostackApp:
    settings.rag.corpus_dir = corpus_dir
    return ZerostackApp(settings=settings, include_mcp=False)


class TestOfflineFirstRun:
    def test_a_fresh_app_starts_with_no_services(self, app):
        health = app.health()
        assert health["layers"]["llm"]["ollama_reachable"] is False
        assert health["layers"]["orchestrator"]["active"] == "simple"

    def test_ingest_then_ask_produces_a_grounded_answer(self, app):
        report = app.ingest()
        assert report["files"] == 2
        assert report["chunks"] > 0

        result = app.ask("How much can an agent refund without approval?")
        assert "two hundred dollars" in result.answer
        assert result.sources == ["support.md"]

    def test_asking_before_ingesting_does_not_crash(self, app):
        result = app.ask("How much can an agent refund?")
        assert result.answer
        assert result.sources == []

    def test_empty_question_is_rejected(self, app):
        with pytest.raises(ValueError):
            app.ask("   ")

    def test_ingesting_a_missing_path_raises(self, app):
        with pytest.raises(FileNotFoundError):
            app.ingest("/does/not/exist")


class TestPersistence:
    def test_every_ask_is_recorded(self, app):
        app.ingest()
        app.ask("When does coverage begin?")
        app.ask("How much can an agent refund?")

        runs = app.recent_runs()
        assert len(runs) == 2
        assert app.analytics()["runs"] == 2

    def test_persistence_can_be_skipped(self, app):
        app.ingest()
        app.ask("When does coverage begin?", persist=False)
        assert app.recent_runs() == []

    def test_a_broken_store_does_not_break_answering(self, app, monkeypatch):
        """Recording a run is bookkeeping. It must never fail a user's answer."""

        def explode(record):
            raise RuntimeError("disk full")

        monkeypatch.setattr(app.store, "save_run", explode)
        app.ingest()
        result = app.ask("How much can an agent refund?")
        assert "two hundred dollars" in result.answer


class TestTracing:
    def test_spans_are_recorded_for_a_run(self, app):
        app.ingest()
        app.ask("How much can an agent refund?")
        names = {span["name"] for span in app.last_trace()}
        assert {"orchestrator.run", "rag.retrieve", "llm.complete"} <= names

    def test_a_span_records_an_error_and_reraises(self, tmp_path):
        tracer = Tracer()
        tracer.settings.enabled = False
        with pytest.raises(ValueError):
            with tracer.span("boom"):
                raise ValueError("bad")
        assert tracer.spans[-1].error.startswith("ValueError")

    def test_spans_are_json_serialisable(self, app):
        app.ingest()
        app.ask("How much can an agent refund?")
        json.dumps(app.last_trace(), default=str)


class TestConfiguration:
    def test_defaults_load_without_any_environment(self):
        settings = Settings()
        assert settings.llm.provider == "auto"
        assert settings.rag.backend == "auto"
        assert settings.orchestrator.kind == "auto"

    def test_environment_overrides_are_read(self, monkeypatch):
        monkeypatch.setenv("ZEROSTACK_LLM_PROVIDER", "echo")
        monkeypatch.setenv("ZEROSTACK_RAG_TOP_K", "9")
        settings = Settings()
        assert settings.llm.provider == "echo"
        assert settings.rag.top_k == 9

    def test_ensure_directories_is_safe_to_call_twice(self, settings):
        settings.ensure_directories()
        settings.ensure_directories()
        assert settings.data.sqlite_path.parent.exists()
