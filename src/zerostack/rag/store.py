"""Vector store implementations for the RAG layer.

Qdrant is the default backend. The client runs in three modes with the same code
path: an in process mode (``:memory:``), an embedded on disk mode (a local path) and
a server mode (an http URL). That is why Qdrant was chosen as the baseline default:
the local development story and the production story are the same API, so growing an
SMB deployment from a laptop to a cluster is a configuration change.

A dependency free in memory store is also provided as a last resort fallback.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from zerostack.rag.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """One retrieved chunk with its similarity score."""

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """The interface the retriever depends on."""

    name: str

    def ensure_collection(self, dimensions: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    def search(
        self, vector: list[float], top_k: int = 4, score_threshold: float = 0.0
    ) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryVectorStore:
    """A pure Python cosine similarity store. No dependencies, no persistence."""

    name = "memory"

    def __init__(self) -> None:
        self._vectors: list[list[float]] = []
        self._chunks: list[Chunk] = []

    def ensure_collection(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        existing = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.chunk_id in existing:
                position = existing[chunk.chunk_id]
                self._chunks[position] = chunk
                self._vectors[position] = vector
            else:
                self._chunks.append(chunk)
                self._vectors.append(vector)
        return len(chunks)

    def search(
        self, vector: list[float], top_k: int = 4, score_threshold: float = 0.0
    ) -> list[SearchResult]:
        scored = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=_cosine(vector, candidate),
                metadata=chunk.metadata,
            )
            for chunk, candidate in zip(self._chunks, self._vectors, strict=True)
        ]
        scored = [result for result in scored if result.score >= score_threshold]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._vectors.clear()
        self._chunks.clear()


class QdrantVectorStore:
    """Qdrant backend covering in memory, embedded and server modes."""

    name = "qdrant"

    def __init__(self, url: str = ":memory:", collection: str = "zerostack") -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self.url = url
        if url in (":memory:", "memory", ""):
            self._client = QdrantClient(location=":memory:")
            self.mode = "in-memory"
        elif url.startswith(("http://", "https://")):
            # The version handshake emits a warning on every construction, including
            # when the probe fails during auto detection. The client is pinned in
            # pyproject, so the check adds noise without adding safety.
            self._client = QdrantClient(url=url, check_compatibility=False)
            self.mode = "server"
        else:
            self._client = QdrantClient(path=url)
            self.mode = "embedded"

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.debug("qdrant ping failed: %s", exc)
            return False

    def ensure_collection(self, dimensions: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return 0

        points = [
            PointStruct(
                # A deterministic uuid5 keeps re-ingesting the same file idempotent.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self, vector: list[float], top_k: int = 4, score_threshold: float = 0.0
    ) -> list[SearchResult]:
        hits = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold or None,
            with_payload=True,
        ).points
        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    score=float(hit.score),
                    metadata=payload.get("metadata", {}) or {},
                )
            )
        return results

    def count(self) -> int:
        if not self._client.collection_exists(self.collection):
            return 0
        return int(self._client.count(self.collection, exact=True).count)

    def reset(self) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)


class ChromaVectorStore:
    """Chroma backend. Requires the ``chroma`` extra."""

    name = "chroma"

    def __init__(self, path: str = "data/chroma", collection: str = "zerostack") -> None:
        import chromadb

        self.collection_name = collection
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def ensure_collection(self, dimensions: int) -> None:
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source, **chunk.metadata} for chunk in chunks],
        )
        return len(chunks)

    def search(
        self, vector: list[float], top_k: int = 4, score_threshold: float = 0.0
    ) -> list[SearchResult]:
        response = self._collection.query(query_embeddings=[vector], n_results=top_k)
        results: list[SearchResult] = []
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            # Chroma returns cosine distance, so similarity is 1 minus that distance.
            score = 1.0 - float(distance)
            if score < score_threshold:
                continue
            metadata = dict(metadata or {})
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=document,
                    source=metadata.pop("source", "unknown"),
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
