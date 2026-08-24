"""The application object that wires all eight layers together.

Everything above this module (the API, the CLI, the Streamlit frontend) depends only
on :class:`ZerostackApp`. That is what allows the frontend layer to be swapped from
Streamlit to Next.js without touching any of the layers below it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zerostack.config import Settings, get_settings
from zerostack.data import RunRecord, StateStore, analytics_summary
from zerostack.llm import build_llm
from zerostack.llm.ollama import OllamaLLM
from zerostack.observability import configure_logging, get_tracer
from zerostack.orchestrator import AgentContext, AgentResult, build_orchestrator
from zerostack.orchestrator.factory import available_engines
from zerostack.rag import RAGPipeline
from zerostack.tools import build_registry

logger = logging.getLogger(__name__)


class ZerostackApp:
    """Composition root. Construct once per process and reuse."""

    def __init__(self, settings: Settings | None = None, include_mcp: bool = True) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.observability.log_level)

        self.llm = build_llm(self.settings.llm)
        self.rag = RAGPipeline(settings=self.settings.rag)
        self.tools = build_registry(include_mcp=include_mcp)
        self.store = StateStore(self.settings.data.sqlite_path)

        self.context = AgentContext(
            llm=self.llm,
            rag=self.rag,
            tools=self.tools,
            settings=self.settings.orchestrator,
        )
        self.orchestrator = build_orchestrator(self.context, self.settings.orchestrator)

    def ask(self, question: str, persist: bool = True) -> AgentResult:
        """Run one question through the agent and record the result."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        result = self.orchestrator.run(question.strip())

        if persist:
            try:
                self.store.save_run(
                    RunRecord(
                        question=result.question,
                        answer=result.answer,
                        orchestrator=result.orchestrator,
                        llm_provider=result.llm_provider,
                        llm_model=result.llm_model,
                        latency_ms=result.latency_ms,
                        trace_id=result.trace_id,
                        sources=result.sources,
                        steps=[step.to_dict() for step in result.steps],
                    )
                )
            except Exception as exc:
                # Persistence must never fail a user facing answer.
                logger.warning("could not persist run: %s", exc)

        return result

    def ingest(self, path: Path | str | None = None) -> dict[str, Any]:
        """Ingest a file or directory into the vector store."""
        target = Path(path) if path else self.settings.rag.corpus_dir
        if not target.exists():
            raise FileNotFoundError(f"nothing to ingest at {target}")
        report = self.rag.ingest_path(target)
        return {
            "path": str(target),
            "files": report.files,
            "chunks": report.chunks,
            "skipped": report.skipped,
            "indexed_chunks": self.rag.store.count(),
        }

    def health(self) -> dict[str, Any]:
        """Report which backend is live in every layer. Backs `zerostack doctor`."""
        ollama = OllamaLLM(model=self.settings.llm.model, base_url=self.settings.llm.base_url)
        ollama_up = ollama.available()

        return {
            "app": self.settings.app_name,
            "environment": self.settings.environment,
            "layers": {
                "orchestrator": {
                    "active": self.orchestrator.name,
                    "available": available_engines(),
                },
                "rag": self.rag.describe(),
                "llm": {
                    "active_provider": self.llm.name,
                    "model": self.llm.model,
                    "ollama_reachable": ollama_up,
                    "ollama_models": ollama.installed_models() if ollama_up else [],
                },
                "tools": {
                    "count": len(self.tools),
                    "names": self.tools.names(),
                },
                "data": {
                    "sqlite_path": str(self.settings.data.sqlite_path),
                    "runs": self.store.count_runs(),
                },
                "observability": {
                    "enabled": self.settings.observability.enabled,
                    "export_traces": self.settings.observability.export_traces,
                    "trace_log": str(self.settings.observability.trace_log_path),
                },
            },
        }

    def analytics(self) -> dict[str, Any]:
        """Aggregate run metrics from the data layer."""
        return analytics_summary(self.settings.data.sqlite_path)

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.recent_runs(limit=limit)

    def last_trace(self) -> list[dict[str, Any]]:
        """Return the spans recorded by the most recent run."""
        return get_tracer().summary()
