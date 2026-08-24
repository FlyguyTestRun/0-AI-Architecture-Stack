"""Tests for the API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import zerostack.api.main as api_module
from zerostack.api.main import api
from zerostack.app import ZerostackApp


@pytest.fixture
def client(settings, monkeypatch, corpus_dir):
    """A test client backed by a fully offline application instance."""
    settings.rag.corpus_dir = corpus_dir
    instance = ZerostackApp(settings=settings, include_mcp=False)
    monkeypatch.setattr(api_module, "_app_instance", instance)
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

    def test_missing_path_returns_404(self, client):
        response = client.post("/ingest", json={"path": "/does/not/exist"})
        assert response.status_code == 404


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
