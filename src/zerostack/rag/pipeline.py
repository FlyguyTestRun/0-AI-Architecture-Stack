"""The RAG pipeline: build a store, ingest documents, retrieve context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from zerostack.config import RAGSettings, get_settings
from zerostack.observability import get_tracer
from zerostack.rag.chunking import Chunk, chunk_text
from zerostack.rag.embeddings import EmbeddingProvider, build_embeddings
from zerostack.rag.graph import KnowledgeGraph, format_graph_context
from zerostack.rag.keyword import BM25Index, reciprocal_rank_fusion
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
    # The display sources this run actually wrote. The answer cache invalidates
    # by source, and without this a caller has nothing to invalidate against but
    # the whole store, which drops every cached answer on every ingestion.
    sources: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = []
        if self.sources is None:
            self.sources = []


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
        self.keyword = BM25Index()
        self._graphs: dict[str, KnowledgeGraph] = {}
        self.graph = self._graph_for("default")
        self._keyword_chunks: dict[str, Chunk] = {}
        # The vector store may already hold documents from a previous process
        # while the keyword index starts empty every time, so it is rebuilt from
        # the store rather than left silently behind.
        self._sync_keyword_index()

    def ingest_text(
        self,
        text: str,
        source: str,
        source_id: str = "",
        namespace: str = "default",
        **metadata: object,
    ) -> int:
        """Chunk, embed and upsert a single document. Returns the chunk count."""
        chunks = chunk_text(
            text,
            source=source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            metadata=dict(metadata),
            source_id=source_id,
            namespace=namespace,
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

    def ingest_path(self, path: Path, namespace: str = "default") -> IngestReport:
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
                source = self._display_source(file_path, path)
                count = self.ingest_text(
                    content,
                    source=source,
                    # Namespaced so the same file ingested into two namespaces
                    # produces two independent chunks rather than one shared one.
                    source_id=f"{namespace}::{file_path.resolve().as_posix()}",
                    namespace=namespace,
                )
                if count:
                    report.files += 1
                    report.chunks += count
                    if source not in report.sources:
                        report.sources.append(source)
            span.set(files=report.files, chunks=report.chunks)

        return report

    def _sync_keyword_index(self) -> None:
        """Rebuild the keyword index from whatever the vector store holds."""
        try:
            chunks = list(self.store.iter_chunks())
        except Exception as exc:
            logger.warning("could not rebuild the keyword index: %s", exc)
            return
        if not chunks:
            return
        self.keyword.clear()
        self._graphs.clear()
        self.graph = self._graph_for("default")
        self._keyword_chunks.clear()
        self._index_keywords(chunks)
        logger.info("rag layer: keyword and graph tiers rebuilt from %d chunk(s)", len(chunks))

    def _index_keywords(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.keyword.add(chunk.chunk_id, chunk.text)
            self._keyword_chunks[chunk.chunk_id] = chunk
            if self.settings.graph_enabled:
                # One graph per namespace. A shared graph would let a traversal
                # walk from one tenant's entity into another tenant's document,
                # which is exactly the boundary the namespace exists to draw.
                self._graph_for(chunk.namespace).add_document(chunk.text, chunk.source)

    def _graph_for(self, namespace: str) -> KnowledgeGraph:
        graph = self._graphs.get(namespace)
        if graph is None:
            graph = KnowledgeGraph()
            self._graphs[namespace] = graph
        return graph

    def _upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        written = self.store.upsert(chunks, vectors)
        self._index_keywords(chunks)
        return written

    def _vector_search(self, query: str, limit: int) -> list[SearchResult]:
        vector = self.embeddings.embed_one(query)
        return self.store.search(vector, top_k=limit, score_threshold=self.settings.score_threshold)

    def _keyword_search(
        self, query: str, limit: int, namespace: str | None = None
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for hit in self.keyword.search(query, top_k=limit):
            chunk = self._keyword_chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            if namespace is not None and chunk.namespace != namespace:
                continue
            results.append(
                SearchResult(
                    chunk_id=hit.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=hit.score,
                    metadata=chunk.metadata,
                    namespace=chunk.namespace,
                )
            )
        return results

    def _fuse(
        self, vector_hits: list[SearchResult], keyword_hits: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Blend two ranked lists by rank position rather than raw score.

        BM25 scores and cosine similarities are on different unnormalised scales,
        so adding them would let whichever happens to be numerically larger
        decide the order. Rank is comparable across retrievers; score is not.
        """
        by_id = {result.chunk_id: result for result in vector_hits}
        by_id.update({result.chunk_id: result for result in keyword_hits})

        fused = reciprocal_rank_fusion(
            [[r.chunk_id for r in vector_hits], [r.chunk_id for r in keyword_hits]],
            weights=[self.settings.vector_weight, self.settings.keyword_weight],
        )

        merged: list[SearchResult] = []
        for chunk_id, score in fused[:top_k]:
            result = by_id.get(chunk_id)
            if result is None:
                continue
            merged.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    score=score,
                    metadata=result.metadata,
                    namespace=result.namespace,
                )
            )
        return merged

    def retrieve(
        self, query: str, top_k: int | None = None, namespace: str | None = None
    ) -> list[SearchResult]:
        """Return the chunks most relevant to ``query``, within a namespace."""
        top_k = top_k or self.settings.top_k
        mode = self.settings.retrieval_mode
        with get_tracer().span(
            "rag.retrieve", query=query, top_k=top_k, mode=mode, namespace=namespace
        ) as span:
            candidates = max(top_k, self.settings.fusion_candidates)
            # Over fetch when scoping, because the namespace filter is applied
            # after the search and would otherwise return fewer than top_k.
            if namespace is not None:
                candidates *= 4
            use_keyword = mode in ("auto", "hybrid", "keyword") and self.keyword.size > 0
            use_vector = mode in ("auto", "hybrid", "vector")

            vector_hits = self._vector_search(query, candidates) if use_vector else []
            if namespace is not None:
                vector_hits = [h for h in vector_hits if h.namespace == namespace]
            keyword_hits = self._keyword_search(query, candidates, namespace) if use_keyword else []

            if use_vector and use_keyword:
                results = self._fuse(vector_hits, keyword_hits, top_k)
                span.set(fused=True, vector_hits=len(vector_hits), keyword_hits=len(keyword_hits))
            elif use_keyword:
                results = keyword_hits[:top_k]
            else:
                results = vector_hits[:top_k]

            # The cutoff applies to every mode. It exists so the reported sources
            # reflect what actually grounded the answer rather than whatever
            # filled out top k, and skipping it on the fused path quietly broke
            # that promise. On fused scores the ratio also does useful work: a
            # chunk both retrievers returned scores about twice one that only
            # appeared in a single list, so the default cleanly prefers agreement.
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

    def graph_context(self, query: str, namespace: str = "default") -> str:
        """Relations connecting the entities this query names, as prompt context.

        Returns an empty string when the graph knows none of the query's
        entities, so a question the graph cannot help with costs nothing.
        """
        graph = self._graphs.get(namespace)
        if not self.settings.graph_enabled or graph is None or graph.relation_count == 0:
            return ""
        with get_tracer().span("rag.graph_traverse", query=query, namespace=namespace) as span:
            neighbourhood = graph.traverse(
                query,
                hops=self.settings.graph_hops,
                max_entities=self.settings.graph_max_entities,
            )
            span.set(
                seeds=neighbourhood.seeds,
                entities=len(neighbourhood.entities),
                relations=len(neighbourhood.relations),
            )
        return format_graph_context(neighbourhood, self.settings.graph_max_relations)

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
            "retrieval_mode": self.settings.retrieval_mode,
            "keyword_index_size": self.keyword.size,
            "graph": self.graph.describe(),
            "namespaces": sorted(self._graphs),
        }
