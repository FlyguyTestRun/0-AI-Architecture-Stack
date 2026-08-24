"""Shared fixtures.

Every fixture pins the offline backends so the suite is deterministic and needs no
Docker, no model server and no network.
"""

from __future__ import annotations

import pytest

from zerostack.config import (
    DataSettings,
    LLMSettings,
    ObservabilitySettings,
    OrchestratorSettings,
    RAGSettings,
    Settings,
)
from zerostack.llm.offline import OfflineLLM
from zerostack.orchestrator.base import AgentContext
from zerostack.rag.embeddings import HashingEmbeddings
from zerostack.rag.pipeline import RAGPipeline
from zerostack.rag.store import MemoryVectorStore
from zerostack.tools import ToolRegistry
from zerostack.tools.builtin import build_builtin_tools

ONBOARDING = """
# Onboarding

Every new employee is subject to a ninety day probationary period. The manager runs a
formal check in at thirty days, sixty days and ninety days.

Health, dental and vision coverage begins on the first day of the month following the
start date. Enrollment must be completed within thirty days of the start date.
"""

SUPPORT = """
# Support

Support agents may issue a refund up to two hundred dollars without approval. The
support lead may approve up to two thousand dollars.

Severity 1 means a complete outage affecting all customers. The response target is
fifteen minutes.
"""


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pinned to offline backends and a temporary directory."""
    return Settings(
        llm=LLMSettings(provider="echo"),
        rag=RAGSettings(
            backend="memory",
            embedding_backend="hashing",
            embedding_dimensions=256,
            corpus_dir=tmp_path / "corpus",
        ),
        orchestrator=OrchestratorSettings(kind="simple"),
        data=DataSettings(sqlite_path=tmp_path / "state.db", duckdb_path=tmp_path / "a.duckdb"),
        observability=ObservabilitySettings(
            enabled=False, trace_log_path=tmp_path / "traces.jsonl"
        ),
    )


@pytest.fixture
def rag(settings) -> RAGPipeline:
    """A RAG pipeline over the two sample documents."""
    pipeline = RAGPipeline(
        store=MemoryVectorStore(),
        embeddings=HashingEmbeddings(dimensions=256),
        settings=settings.rag,
    )
    pipeline.ingest_text(ONBOARDING, source="onboarding.md")
    pipeline.ingest_text(SUPPORT, source="support.md")
    return pipeline


@pytest.fixture
def tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(build_builtin_tools())
    return registry


@pytest.fixture
def context(rag, tools, settings) -> AgentContext:
    return AgentContext(llm=OfflineLLM(), rag=rag, tools=tools, settings=settings.orchestrator)


@pytest.fixture
def corpus_dir(tmp_path):
    """A directory holding the two sample documents plus one unsupported file."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "onboarding.md").write_text(ONBOARDING, encoding="utf-8")
    (directory / "support.md").write_text(SUPPORT, encoding="utf-8")
    (directory / "logo.png").write_bytes(b"\x89PNG\r\n")
    return directory
