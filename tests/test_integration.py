"""End to end tests over the composed application.

These are the tests that would catch a regression in the promise the README makes:
a clean clone answers questions correctly with no Docker, no model server and no
network access.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from zerostack.app import ZerostackApp
from zerostack.config import Settings
from zerostack.observability.tracing import Tracer


def tmp_path_factory_dir() -> str:
    """An empty directory, so no .env file on disk can reach Settings()."""
    return tempfile.mkdtemp(prefix="zerostack-clean-")


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

    def test_ingesting_a_missing_path_inside_the_root_raises(self, app):
        with pytest.raises(FileNotFoundError):
            app.ingest(app.settings.rag.corpus_dir / "absent.md")

    def test_ingesting_outside_the_root_is_refused_before_existence(self, app):
        """The boundary answers first, so a caller cannot probe for what exists."""
        with pytest.raises(PermissionError):
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
    def test_defaults_load_without_any_environment(self, monkeypatch):
        """Defaults must hold with a genuinely clean environment.

        This previously read the ambient environment and passed only because no
        ZEROSTACK_ variable happened to be set. Running it with
        ZEROSTACK_ORCHESTRATOR_KIND exported, as the orchestrator matrix job
        does, failed it. A test that asserts "without any environment" has to
        clear the environment, or it asserts nothing.
        """
        for name in list(os.environ):
            if name.startswith("ZEROSTACK_"):
                monkeypatch.delenv(name, raising=False)
        # A .env file on the developer's machine would defeat the same intent.
        monkeypatch.chdir(tmp_path_factory_dir())

        settings = Settings()
        assert settings.llm.provider == "auto"
        assert settings.rag.backend == "auto"
        assert settings.orchestrator.kind == "auto"
        assert settings.rag.embedding_backend == "auto"

    def test_environment_is_read_even_when_defaults_exist(self, monkeypatch):
        """The inverse of the test above: an exported value must win."""
        monkeypatch.setenv("ZEROSTACK_ORCHESTRATOR_KIND", "simple")
        assert Settings().orchestrator.kind == "simple"

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


class TestDotenvLoading:
    """The documented flow is to copy .env.example to .env and edit it.

    Each settings section is built by its own default_factory, independently of
    the outer Settings object, so each has to read the dotenv file itself. When
    only the outer object read it, the documented flow appeared to do nothing: a
    deployment could name a provider explicitly and still silently run on the
    offline fallback.
    """

    @pytest.fixture
    def dotenv_dir(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "ZEROSTACK_LLM_PROVIDER=ollama\n"
            "ZEROSTACK_LLM_MODEL=custom-model\n"
            "ZEROSTACK_RAG_TOP_K=11\n"
            "ZEROSTACK_ORCHESTRATOR_KIND=simple\n"
            "ZEROSTACK_OBS_LOG_LEVEL=WARNING\n",
            encoding="utf-8",
        )
        for name in list(os.environ):
            if name.startswith("ZEROSTACK_"):
                monkeypatch.delenv(name, raising=False)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_llm_section_reads_dotenv(self, dotenv_dir):
        settings = Settings()
        assert settings.llm.provider == "ollama"
        assert settings.llm.model == "custom-model"

    def test_rag_section_reads_dotenv(self, dotenv_dir):
        assert Settings().rag.top_k == 11

    def test_orchestrator_section_reads_dotenv(self, dotenv_dir):
        assert Settings().orchestrator.kind == "simple"

    def test_observability_section_reads_dotenv(self, dotenv_dir):
        assert Settings().observability.log_level == "WARNING"

    def test_process_environment_still_wins_over_dotenv(self, dotenv_dir, monkeypatch):
        monkeypatch.setenv("ZEROSTACK_LLM_PROVIDER", "echo")
        assert Settings().llm.provider == "echo"
