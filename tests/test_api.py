"""Tests for the API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import zerostack.api.main as api_module
from zerostack.api.main import api
from zerostack.app import ZerostackApp


@pytest.fixture
def app(settings, corpus_dir) -> ZerostackApp:
    settings.rag.corpus_dir = corpus_dir
    return ZerostackApp(settings=settings, include_mcp=False)


@pytest.fixture
def client(app, monkeypatch):
    """A test client backed by a fully offline application instance."""
    monkeypatch.setattr(api_module, "_app_instance", app)
    with TestClient(api) as test_client:
        yield test_client


class TestHealth:
    def test_reports_every_layer(self, client):
        body = client.get("/health").json()
        assert body["layers"]["orchestrator"]["active"] == "simple"
        assert body["layers"]["llm"]["active_provider"] == "offline"
        assert body["layers"]["rag"]["vector_store"] == "memory"


class TestAsk:
    def test_answers_from_the_corpus(self, client):
        client.post("/ingest", json={})
        response = client.post(
            "/ask", json={"question": "How much can an agent refund without approval?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert "two hundred dollars" in body["answer"]
        assert body["sources"] == ["support.md"]
        assert [step["name"] for step in body["steps"]] == [
            "plan",
            "retrieve",
            "generate",
        ]

    def test_empty_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_whitespace_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": "   "}).status_code == 422

    def test_missing_body_is_rejected(self, client):
        assert client.post("/ask", json={}).status_code == 422


class TestIngest:
    def test_indexes_the_corpus(self, client):
        body = client.post("/ingest", json={}).json()
        assert body["files"] == 2
        assert body["chunks"] > 0

    def test_a_missing_path_inside_the_root_returns_404(self, client, app):
        target = app.settings.rag.corpus_dir / "absent.md"
        assert client.post("/ingest", json={"path": str(target)}).status_code == 404

    def test_a_missing_path_outside_the_root_returns_403_not_404(self, client):
        """Existence is checked after the boundary, never before.

        Returning 404 for an out of bounds path would let a caller probe the
        filesystem for which paths exist, so the boundary answers first.
        """
        assert client.post("/ingest", json={"path": "/does/not/exist"}).status_code == 403


class TestRunsAndAnalytics:
    def test_runs_are_recorded_and_listed(self, client):
        client.post("/ingest", json={})
        client.post("/ask", json={"question": "When does coverage begin?"})

        runs = client.get("/runs").json()["runs"]
        assert len(runs) == 1
        assert runs[0]["question"] == "When does coverage begin?"

    def test_run_limit_is_clamped(self, client):
        assert client.get("/runs?limit=100000").status_code == 200

    def test_analytics_reports_the_run(self, client):
        client.post("/ingest", json={})
        client.post("/ask", json={"question": "When does coverage begin?"})
        assert client.get("/analytics").json()["runs"] == 1


class TestTools:
    def test_lists_registered_tools(self, client):
        body = client.get("/tools").json()
        assert body["count"] == 2
        assert {tool["name"] for tool in body["tools"]} == {"calculate", "current_time"}


class TestConcurrency:
    """The API serves sync endpoints on a threadpool, so per request state matters.

    An earlier tracer kept the current trace id on the instance rather than per
    thread, so simultaneous requests overwrote each other and runs were persisted
    with another request's trace id.
    """

    def test_concurrent_requests_get_distinct_traces(self, client):
        from concurrent.futures import ThreadPoolExecutor

        client.post("/ingest", json={})
        questions = [
            "How much can an agent refund without approval?",
            "When does coverage begin?",
            "What is the escalation path?",
            "What is the severity 1 response target?",
        ] * 3

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(lambda q: client.post("/ask", json={"question": q}).json(), questions)
            )

        trace_ids = [response["trace_id"] for response in responses]
        assert all(trace_ids), "a request completed with no trace id"
        assert len(set(trace_ids)) == len(responses), "two requests shared a trace id"

    def test_concurrent_answers_stay_bound_to_their_question(self, client):
        from concurrent.futures import ThreadPoolExecutor

        client.post("/ingest", json={})
        questions = [
            "How much can an agent refund without approval?",
            "When does coverage begin?",
        ] * 6

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(lambda q: client.post("/ask", json={"question": q}).json(), questions)
            )

        for question, response in zip(questions, responses, strict=True):
            assert response["question"] == question

    def test_every_concurrent_run_is_persisted(self, client):
        from concurrent.futures import ThreadPoolExecutor

        client.post("/ingest", json={})
        questions = ["When does coverage begin?"] * 10

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda q: client.post("/ask", json={"question": q}), questions))

        runs = client.get("/runs?limit=100").json()["runs"]
        assert len(runs) == 10
        assert len({run["trace_id"] for run in runs}) == 10
