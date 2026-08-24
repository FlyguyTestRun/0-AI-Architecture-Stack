"""Coverage for the Streamlit frontend.

The app previously had no automated coverage at all, so a regression in it would
have shipped silently. AppTest runs the real script in process, which catches
import errors, API drift and exceptions raised during a render.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="streamlit extra not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# AppTest resolves a relative path against the calling file, so pass an absolute one.
APP_PATH = str(REPO_ROOT / "apps" / "streamlit_app" / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Run the app against a temporary, fully offline environment."""
    monkeypatch.setenv("ZEROSTACK_LLM_PROVIDER", "echo")
    monkeypatch.setenv("ZEROSTACK_RAG_BACKEND", "memory")
    monkeypatch.setenv("ZEROSTACK_RAG_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("ZEROSTACK_ORCHESTRATOR_KIND", "simple")
    monkeypatch.setenv("ZEROSTACK_DATA_SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("ZEROSTACK_OBS_TRACE_LOG_PATH", str(tmp_path / "traces.jsonl"))

    from zerostack.config import reset_settings_cache

    reset_settings_cache()
    return AppTest.from_file(APP_PATH, default_timeout=120)


class TestRender:
    def test_app_renders_without_exceptions(self, app):
        app.run()
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_title_is_present(self, app):
        app.run()
        assert "Zerostack" in [element.value for element in app.title]

    def test_core_controls_are_present(self, app):
        app.run()
        labels = [button.label for button in app.button]
        assert "Ask" in labels
        assert "Ingest sample corpus" in labels

    def test_sidebar_reports_layer_status(self, app):
        app.run()
        rendered = " ".join(
            str(element.value) for element in list(app.info) + list(app.warning) + list(app.success)
        )
        assert "RAG" in rendered
        assert "Orchestrator" in rendered

    def test_empty_question_does_not_raise(self, app):
        """Clicking Ask with no question must be a no op, not a traceback."""
        app.run()
        for button in app.button:
            if button.label == "Ask":
                button.click()
                break
        app.run()
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_no_deprecated_width_argument(self):
        """use_container_width was removed from Streamlit's supported surface."""
        source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "use_container_width" not in source
