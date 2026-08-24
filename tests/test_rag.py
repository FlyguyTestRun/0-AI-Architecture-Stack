"""Tests for the RAG pipeline (layer 3)."""

from __future__ import annotations

import pytest

from zerostack.rag.chunking import chunk_text
from zerostack.rag.embeddings import HashingEmbeddings
from zerostack.rag.store import MemoryVectorStore


class TestChunking:
    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("A short paragraph about refunds.", source="a.md")
        assert len(chunks) == 1
        assert chunks[0].source == "a.md"
        assert chunks[0].chunk_id == "a.md::0"

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("   \n\n  ", source="a.md") == []

    def test_long_text_splits_with_overlap(self):
        text = "\n\n".join(f"Paragraph number {i} about policy." for i in range(60))
        chunks = chunk_text(text, source="a.md", chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 260 for chunk in chunks)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_oversized_paragraph_is_windowed_not_dropped(self):
        text = "x" * 2000
        chunks = chunk_text(text, source="a.md", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
        assert sum(len(chunk.text) for chunk in chunks) >= 2000

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", source="a.md", chunk_size=100, chunk_overlap=100)

    def test_metadata_is_carried_onto_every_chunk(self):
        chunks = chunk_text(
            "\n\n".join(["Some policy text here."] * 20),
            source="a.md",
            chunk_size=100,
            chunk_overlap=10,
            metadata={"team": "hr"},
        )
        assert all(chunk.metadata["team"] == "hr" for chunk in chunks)


class TestHashingEmbeddings:
    def test_is_deterministic(self):
        provider = HashingEmbeddings(dimensions=128)
        assert provider.embed_one("refund policy") == provider.embed_one("refund policy")

    def test_vectors_are_normalised(self):
        provider = HashingEmbeddings(dimensions=128)
        vector = provider.embed_one("refund policy")
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-9)

    def test_related_text_scores_higher_than_unrelated(self):
        provider = HashingEmbeddings(dimensions=256)
        query = provider.embed_one("refund approval limit")
        related = provider.embed_one("agents may issue a refund without approval")
        unrelated = provider.embed_one("the kitchen is restocked on Fridays")

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        assert dot(query, related) > dot(query, unrelated)

    def test_empty_text_gives_a_zero_vector(self):
        provider = HashingEmbeddings(dimensions=32)
        assert provider.embed_one("") == [0.0] * 32

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError):
            HashingEmbeddings(dimensions=0)


class TestMemoryVectorStore:
    def test_upsert_is_idempotent_by_chunk_id(self, rag):
        store = MemoryVectorStore()
        provider = HashingEmbeddings(dimensions=64)
        chunks = chunk_text("Refund policy details.", source="a.md")
        vectors = provider.embed([c.text for c in chunks])

        store.upsert(chunks, vectors)
        store.upsert(chunks, vectors)
        assert store.count() == 1

    def test_length_mismatch_is_rejected(self):
        store = MemoryVectorStore()
        chunks = chunk_text("Refund policy details.", source="a.md")
        with pytest.raises(ValueError):
            store.upsert(chunks, [])


class TestRetrieval:
    def test_retrieves_the_relevant_document(self, rag):
        results = rag.retrieve("How much can an agent refund without approval?")
        assert results
        assert results[0].source == "support.md"
        assert "two hundred dollars" in results[0].text

    def test_retrieves_across_documents(self, rag):
        results = rag.retrieve("When does health coverage begin?")
        assert results[0].source == "onboarding.md"

    def test_relevance_cutoff_drops_weak_hits(self, rag):
        rag.settings.relevance_ratio = 0.99
        results = rag.retrieve("refund approval")
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
        assert len(results) <= rag.settings.top_k

    def test_format_context_numbers_and_cites_sources(self, rag):
        results = rag.retrieve("refund approval")
        block = rag.format_context(results)
        assert block.startswith("[1] source:")
        assert "support.md" in block

    def test_format_context_of_nothing_is_empty(self, rag):
        assert rag.format_context([]) == ""

    def test_describe_reports_the_live_backends(self, rag):
        described = rag.describe()
        assert described["vector_store"] == "memory"
        assert described["embeddings"] == "hashing"
        assert described["indexed_chunks"] > 0


class TestIngestion:
    def test_ingests_a_directory_and_ignores_unsupported_files(self, rag, corpus_dir):
        report = rag.ingest_path(corpus_dir)
        # The png is filtered by suffix before it is ever opened, so it is neither
        # ingested nor reported as skipped.
        assert report.files == 2
        assert report.chunks > 0
        assert report.skipped == []

    def test_binary_file_passed_directly_is_skipped_not_fatal(self, rag, corpus_dir):
        report = rag.ingest_path(corpus_dir / "logo.png")
        assert report.files == 0
        assert report.skipped == [str(corpus_dir / "logo.png")]

    def test_ingesting_a_single_file_works(self, rag, corpus_dir):
        report = rag.ingest_path(corpus_dir / "support.md")
        assert report.files == 1


class TestBackendSelection:
    def test_explicit_memory_backend_is_honoured(self, settings):
        from zerostack.rag.pipeline import build_vector_store

        settings.rag.backend = "memory"
        assert build_vector_store(settings.rag).name == "memory"

    def test_auto_falls_back_to_embedded_qdrant_and_persists(self, settings, tmp_path):
        """An index built in one process must still be there in the next one.

        The embedded backend is what makes `zerostack ingest` followed by a separate
        `zerostack ask` work with no Docker running.
        """
        from zerostack.rag.pipeline import RAGPipeline, build_vector_store

        settings.rag.backend = "auto"
        settings.rag.qdrant_url = "http://127.0.0.1:1"  # guaranteed unreachable
        settings.rag.qdrant_path = tmp_path / "qdrant"
        settings.rag.embedding_dimensions = 256

        first = build_vector_store(settings.rag)
        assert first.name == "qdrant"
        assert first.mode == "embedded"

        pipeline = RAGPipeline(store=first, settings=settings.rag)
        pipeline.ingest_text(
            "Support agents may issue a refund up to two hundred dollars.",
            source="support.md",
        )
        assert first.count() == 1
        # Release the directory lock the way a finishing process would.
        del pipeline, first

        second = build_vector_store(settings.rag)
        assert second.count() == 1

    def test_locked_directory_falls_through_instead_of_failing(self, settings, tmp_path):
        """A second process must degrade, not crash, when the lock is held."""
        from zerostack.rag.pipeline import build_vector_store

        settings.rag.backend = "auto"
        settings.rag.qdrant_url = "http://127.0.0.1:1"
        settings.rag.qdrant_path = tmp_path / "qdrant"

        holder = build_vector_store(settings.rag)
        holder.ensure_collection(256)

        fallback = build_vector_store(settings.rag)
        assert fallback is not None
        assert fallback.name in {"qdrant", "memory"}
