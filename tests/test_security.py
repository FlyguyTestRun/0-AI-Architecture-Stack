"""Security tests for the ingest boundary.

The ingest endpoint takes a filesystem path from the caller. Before this was
constrained, any caller who could reach the endpoint could index an arbitrary
readable directory and then read its contents back out through the ask endpoint,
turning a document tool into arbitrary file disclosure. These tests pin the
boundary shut.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import zerostack.api.main as api_module
from zerostack.api.main import api
from zerostack.app import ZerostackApp


@pytest.fixture
def secret_dir(tmp_path):
    """A directory the caller must never be able to reach."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "credentials.md").write_text(
        "The production database password is hunter2 and the signing key is sk-live-9f3a.",
        encoding="utf-8",
    )
    return secrets


@pytest.fixture
def app(settings, corpus_dir) -> ZerostackApp:
    settings.rag.corpus_dir = corpus_dir
    return ZerostackApp(settings=settings, include_mcp=False)


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setattr(api_module, "_app_instance", app)
    with TestClient(api) as test_client:
        yield test_client


class TestIngestBoundary:
    def test_arbitrary_directory_is_refused(self, client, secret_dir):
        response = client.post("/ingest", json={"path": str(secret_dir)})
        assert response.status_code == 403

    def test_traversal_path_is_refused(self, client):
        assert client.post("/ingest", json={"path": "../../../etc"}).status_code == 403

    def test_absolute_system_path_is_refused(self, client):
        assert client.post("/ingest", json={"path": "/etc"}).status_code == 403

    def test_symlink_out_of_the_root_is_refused(self, client, app, secret_dir):
        """Resolving before comparing is what stops a symlink escape."""
        link = app.settings.rag.corpus_dir / "escape"
        try:
            link.symlink_to(secret_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported here")
        assert client.post("/ingest", json={"path": str(link)}).status_code == 403

    def test_the_configured_corpus_is_allowed(self, client):
        response = client.post("/ingest", json={})
        assert response.status_code == 200
        assert response.json()["files"] == 2

    def test_a_path_inside_the_corpus_is_allowed(self, client, app):
        target = app.settings.rag.corpus_dir / "support.md"
        assert client.post("/ingest", json={"path": str(target)}).status_code == 200

    def test_secrets_cannot_be_read_back_through_ask(self, client, secret_dir):
        """The end to end exfiltration path, which is what actually mattered."""
        client.post("/ingest", json={"path": str(secret_dir)})
        answer = client.post(
            "/ask", json={"question": "What is the production database password?"}
        ).json()["answer"]
        assert "hunter2" not in answer
        assert "sk-live" not in answer


class TestAllowlistConfiguration:
    def test_an_explicit_root_widens_the_boundary(self, settings, corpus_dir, secret_dir):
        settings.rag.corpus_dir = corpus_dir
        settings.rag.allowed_ingest_roots = [secret_dir]
        app = ZerostackApp(settings=settings, include_mcp=False)
        report = app.ingest(secret_dir)
        assert report["files"] == 1

    def test_widening_one_root_does_not_widen_others(self, settings, corpus_dir, tmp_path):
        settings.rag.corpus_dir = corpus_dir
        settings.rag.allowed_ingest_roots = [tmp_path / "allowed"]
        (tmp_path / "allowed").mkdir()
        app = ZerostackApp(settings=settings, include_mcp=False)
        with pytest.raises(PermissionError):
            app.ingest(tmp_path / "not-allowed")

    def test_default_root_is_the_corpus(self, app, corpus_dir):
        assert app.allowed_ingest_roots() == [corpus_dir.resolve()]


class TestOperatorPath:
    def test_the_cli_path_can_opt_out(self, app, secret_dir):
        """An operator with shell access can already read these files."""
        report = app.ingest(secret_dir, enforce_roots=False)
        assert report["files"] == 1

    def test_enforcement_is_the_default(self, app, secret_dir):
        with pytest.raises(PermissionError):
            app.ingest(secret_dir)


class TestRefusalDoesNotDisclose:
    """A refusal must not become a way to map the filesystem.

    The API forwards the PermissionError text verbatim to an unauthenticated
    caller, so naming the requested path or the configured roots would let anyone
    probe for what exists and learn where the corpus lives.
    """

    def test_403_body_does_not_name_the_configured_roots(self, client, app, secret_dir):
        detail = client.post("/ingest", json={"path": str(secret_dir)}).json()["detail"]
        corpus = str(app.settings.rag.corpus_dir)
        assert corpus not in detail
        assert str(app.settings.rag.corpus_dir.resolve()) not in detail

    def test_403_body_does_not_echo_the_requested_path(self, client, secret_dir):
        detail = client.post("/ingest", json={"path": str(secret_dir)}).json()["detail"]
        assert str(secret_dir) not in detail
        assert "secrets" not in detail

    def test_403_still_explains_how_to_widen_the_boundary(self, client, secret_dir):
        detail = client.post("/ingest", json={"path": str(secret_dir)}).json()["detail"]
        assert "ZEROSTACK_RAG_ALLOWED_INGEST_ROOTS" in detail

    def test_the_operator_can_still_see_the_detail_in_the_log(self, app, secret_dir, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="zerostack.app"):
            with pytest.raises(PermissionError):
                app.ingest(secret_dir)
        assert str(secret_dir.resolve()) in caplog.text
