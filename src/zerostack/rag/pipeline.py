"""The RAG pipeline: build a store, ingest documents, retrieve context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from zerostack.config import RAGSettings, get_settings
from zerostack.observability import get_tracer
from zerostack.rag.chunking import Chunk, chunk_text
from zerostack.rag.embeddings import EmbeddingProvider, build_embeddings
from zerostack.rag.store import (
    ChromaVectorStore,
    MemoryVectorStore,
    QdrantVectorStore,
    SearchResult,
    VectorStore,
)

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".markdown"}


def build_vector_store(settings: RAGSettings | None = None) -> VectorStore:
    """Return a vector store according to configuration.

    ``auto`` tries the configured Qdrant server, then embedded Qdrant on disk, then
    Qdrant in process, then the dependency free memory store. The fallback chain is
    what makes a fresh clone work before ``docker compose up`` has ever been run.

    Embedded on disk sits ahead of in process on purpose. An in process index is
    discarded when the interpreter exits, so ``zerostack ingest`` followed by
    ``zerostack ask`` would find nothing. Embedded mode takes a directory lock, so if
    another process already holds it this falls through to in process rather than
    failing.
    """
    settings = settings or get_settings().rag

    if settings.backend == "memory":
        return MemoryVectorStore()

    if settings.backend == "chroma":
        return ChromaVectorStore(collection=settings.collection)

    if settings.backend == "qdrant":
        return QdrantVectorStore(url=settings.qdrant_url, collection=settings.collection)

    # auto
    try:
        server = QdrantVectorStore(url=settings.qdrant_url, collection=settings.collection)
        if server.ping():
            logger.info("rag layer: using qdrant server at %s", settings.qdrant_url)
            return server
    except Exception as exc:
        logger.debug("qdrant server unavailable: %s", exc)

    try:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        store = QdrantVectorStore(url=str(settings.qdrant_path), collection=settings.collection)
        logger.info("rag layer: using embedded qdrant at %s", settings.qdrant_path)
        return store
    except Exception as exc:
        logger.info("rag layer: embedded qdrant unavailable (%s), using in process index", exc)

    try:
        return QdrantVectorStore(url=":memory:", collection=settings.collection)
    except Exception as exc:
        logger.info("rag layer: qdrant client unavailable (%s), using memory store", exc)
        return MemoryVectorStore()


@dataclass
class IngestReport:
    """What an ingestion run did."""

    files: int = 0
    chunks: int = 0
    skipped: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = []


class RAGPipeline:
    """Owns the embedding provider and the vector store for one process."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        settings: RAGSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings().rag
        self.embeddings = embeddings or build_embeddings(
            backend=self.settings.embedding_backend,
            model_name=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )
        self.store = store or build_vector_store(self.settings)
        self.store.ensure_collection(self.embeddings.dimensions)

    def ingest_text(self, text: str, source: str, source_id: str = "", **metadata: object) -> int:
        """Chunk, embed and upsert a single document. Returns the chunk count."""
        chunks = chunk_text(
            text,
            source=source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            metadata=dict(metadata),
            source_id=source_id,
        )
        return self._upsert(chunks)

    def _display_source(self, file_path: Path, target: Path) -> str:
        """A short, readable name for citations.

        Preference order is the corpus root, then the ingest target, then the bare
        filename, so a file under the configured corpus keeps the same citation
        however the caller reached it.
        """
        candidates = [self.settings.corpus_dir, target if target.is_dir() else target.parent]
        for root in candidates:
            try:
                return file_path.resolve().relative_to(root.resolve()).as_posix()
            except (ValueError, OSError):
                continue
        return file_path.name

    def ingest_path(self, path: Path) -> IngestReport:
        """Ingest a file, or every supported text file under a directory."""
        report = IngestReport()
        paths = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
            if path.is_dir()
            else [path]
        )

        with get_tracer().span("rag.ingest", path=str(path)) as span:
            for file_path in paths:
                if file_path.suffix.lower() not in TEXT_SUFFIXES:
                    report.skipped.append(str(file_path))
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    logger.warning("could not read %s: %s", file_path, exc)
                    report.skipped.append(str(file_path))
                    continue

                # Identity is the resolved absolute path, so the same file gets
                # the same chunk id no matter which directory the caller passed.
                # Making identity relative to the ingest argument was not enough:
                # ingesting two sibling directories separately still collided, and
                # ingesting a parent then a child left a stale duplicate behind.
                count = self.ingest_text(
                    content,
                    source=self._display_source(file_path, path),
                    source_id=file_path.resolve().as_posix(),
                )
                if count:
                    report.files += 1
                    report.chunks += count
            span.set(files=report.files, chunks=report.chunks)

        return report

    def _upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        return self.store.upsert(chunks, vectors)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the chunks most relevant to ``query``."""
        top_k = top_k or self.settings.top_k
        with get_tracer().span("rag.retrieve", query=query, top_k=top_k) as span:
            vector = self.embeddings.embed_one(query)
            results = self.store.search(
                vector, top_k=top_k, score_threshold=self.settings.score_threshold
            )
            results = self._apply_relevance_cutoff(results)
            span.set(
                hits=len(results),
                top_score=round(results[0].score, 4) if results else None,
                sources=[r.source for r in results],
            )
        return results

    def _apply_relevance_cutoff(self, results: list[SearchResult]) -> list[SearchResult]:
        """Drop hits far weaker than the best one.

        Top k alone will always return k chunks, including irrelevant ones when the
        index is small. Reporting those as sources overstates what actually grounded
        the answer, so they are cut relative to the top score.
        """
        ratio = self.settings.relevance_ratio
        if not results or ratio <= 0:
            return results
        best = results[0].score
        if best <= 0:
            return results
        return [result for result in results if result.score >= best * ratio]

    @staticmethod
    def format_context(results: list[SearchResult]) -> str:
        """Render retrieved chunks into a prompt friendly block with citations."""
        if not results:
            return ""
        blocks = [
            f"[{index}] source: {result.source}\n{result.text}"
            for index, result in enumerate(results, start=1)
        ]
        return "\n\n".join(blocks)

    def describe(self) -> dict[str, object]:
        """Report which backends are actually in use. Used by `zerostack doctor`."""
        return {
            "vector_store": self.store.name,
            "vector_store_mode": getattr(self.store, "mode", "n/a"),
            "embeddings": self.embeddings.name,
            "dimensions": self.embeddings.dimensions,
            "indexed_chunks": self.store.count(),
        }
