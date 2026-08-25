"""End to end tests over the composed application.

These are the tests that would catch a regression in the promise the README makes:
a clean clone answers questions correctly with no Docker, no model server and no
network access.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

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


class TestCacheInvalidationIsTargeted:
    """Ingestion must drop the answers it actually invalidated, not all of them.

    Passing every source in the store to the invalidator is correct but
    worthless: it clears the whole cache on every ingestion, so any deployment
    that ingests on a schedule never serves a cached answer at all.
    """

    def test_ingesting_an_unrelated_document_keeps_the_cache(self, app, corpus_dir, tmp_path):
        """The sharp case: nothing cached cites the new file, so nothing drops."""
        app.ingest(str(corpus_dir), enforce_roots=False)
        app.ask("How long is the probationary period?", persist=False)
        app.ask("How much can a support agent refund without approval?", persist=False)
        cached = app.cache.size
        assert cached == 2

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "unrelated.md").write_text(
            "The cafeteria serves lunch from eleven until two.", encoding="utf-8"
        )
        result = app.ingest(str(elsewhere), enforce_roots=False)

        assert result["cache_entries_invalidated"] == 0
        assert app.cache.size == cached

    def test_the_report_names_what_it_wrote(self, app, corpus_dir):
        report = app.rag.ingest_path(corpus_dir)
        assert sorted(report.sources) == ["onboarding.md", "support.md"]
        # The unsupported file is skipped, so it is not reported as a source.
        assert not any(source.endswith(".png") for source in report.sources)

    def test_a_single_file_ingest_reports_only_that_file(self, app, corpus_dir):
        report = app.rag.ingest_path(corpus_dir / "support.md")
        assert report.sources == ["support.md"]

    def test_an_answer_grounded_in_a_rewritten_document_is_dropped(self, app, corpus_dir):
        """Conservative on purpose: any cited source changing invalidates."""
        app.ingest(str(corpus_dir), enforce_roots=False)
        result = app.ask("How much can a support agent refund without approval?", persist=False)
        assert "support.md" in result.sources
        assert app.cache.size == 1

        (corpus_dir / "support.md").write_text(
            "A support agent may refund up to five hundred dollars without approval.",
            encoding="utf-8",
        )
        app.ingest(str(corpus_dir / "support.md"), enforce_roots=False)
        assert app.cache.size == 0

    def test_an_ingestion_that_wrote_nothing_invalidates_nothing(self, app, corpus_dir, tmp_path):
        app.ingest(str(corpus_dir), enforce_roots=False)
        app.ask("How long is the probationary period?", persist=False)
        before = app.cache.size

        empty = tmp_path / "empty"
        empty.mkdir()
        result = app.ingest(str(empty), enforce_roots=False)

        assert result["cache_entries_invalidated"] == 0
        assert app.cache.size == before


class TestNamespaceIsolationUnderConcurrency:
    """The namespace must travel with the request, not on the shared context.

    The API serves each request on a worker thread against one application
    object. A namespace stored on the shared orchestrator context is process
    wide: one request setting it while another sat between setting and reading
    made the second retrieve inside the first one's tenant.
    """

    @pytest.fixture
    def tenants(self, settings, tmp_path):
        corpus = tmp_path / "corpus"
        (corpus / "hr").mkdir(parents=True)
        (corpus / "legal").mkdir(parents=True)
        (corpus / "hr" / "h.md").write_text(
            "TOKEN_HR marker. The probationary period is ninety days.", encoding="utf-8"
        )
        (corpus / "legal" / "l.md").write_text(
            "TOKEN_LEGAL marker. The retention period is seven years.", encoding="utf-8"
        )
        settings.rag.corpus_dir = corpus
        settings.cache.enabled = False
        instance = ZerostackApp(settings=settings, include_mcp=False)
        instance.ingest(str(corpus / "hr"), namespace="hr", enforce_roots=False)
        instance.ingest(str(corpus / "legal"), namespace="legal", enforce_roots=False)
        return instance

    def test_retrieval_uses_the_namespace_the_caller_asked_for(self, tenants, monkeypatch):
        """Instrumented at the read, because the race window is microseconds."""
        from zerostack.rag.pipeline import RAGPipeline

        wanted = threading.local()
        mismatches: list[tuple[str, str | None]] = []
        original = RAGPipeline.retrieve

        def traced(self, query, top_k=None, namespace=None):
            expected = getattr(wanted, "namespace", None)
            if expected is not None and namespace != expected:
                mismatches.append((expected, namespace))
            time.sleep(0.002)
            return original(self, query, top_k=top_k, namespace=namespace)

        monkeypatch.setattr(RAGPipeline, "retrieve", traced)

        def hammer(namespace: str) -> None:
            wanted.namespace = namespace
            for _ in range(30):
                tenants.ask(
                    "What is the period?", namespace=namespace, persist=False, use_cache=False
                )

        threads = [
            threading.Thread(target=hammer, args=("hr",)),
            threading.Thread(target=hammer, args=("legal",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert mismatches == []

    def test_a_single_request_still_reaches_its_own_tenant(self, tenants):
        result = tenants.ask("What is the period?", namespace="hr", persist=False, use_cache=False)
        assert "TOKEN_LEGAL" not in result.answer


class TestUngroundedAnswersAreInvalidated:
    """An answer built from finding nothing records no source.

    Source based invalidation can never match it, so without a separate rule it
    keeps reporting "nothing found" for its whole TTL after the very document
    that answers it was ingested.
    """

    @pytest.fixture
    def empty_corpus_app(self, settings, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        settings.rag.corpus_dir = corpus
        settings.cache.enabled = True
        return ZerostackApp(settings=settings, include_mcp=False), corpus

    def test_the_no_context_answer_is_dropped_when_a_document_arrives(self, empty_corpus_app):
        app, corpus = empty_corpus_app
        first = app.ask("How long is the probationary period?", persist=False)
        assert first.sources == []
        assert app.cache.size == 1

        (corpus / "hr.md").write_text("The probationary period is ninety days.", encoding="utf-8")
        app.ingest(str(corpus), enforce_roots=False)

        second = app.ask("How long is the probationary period?", persist=False)
        assert second.answer != first.answer
        assert second.sources == ["hr.md"]

    def test_a_grounded_entry_in_another_namespace_survives(self, empty_corpus_app):
        app, corpus = empty_corpus_app
        (corpus / "hr.md").write_text("The probationary period is ninety days.", encoding="utf-8")
        app.ingest(str(corpus), namespace="hr", enforce_roots=False)
        app.ask("How long is the probationary period?", namespace="hr", persist=False)
        before = app.cache.size

        (corpus / "other.md").write_text("Lunch is served at noon.", encoding="utf-8")
        app.ingest(str(corpus / "other.md"), namespace="default", enforce_roots=False)

        assert app.cache.size == before
