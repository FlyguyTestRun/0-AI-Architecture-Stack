"""Embedding providers for the RAG layer.

Two implementations ship. The hashing provider is deterministic, has no model
download and no native dependencies, which keeps CI fast and a clean clone runnable.
The sentence-transformers provider is what a real deployment should use.

Both produce L2 normalised vectors of the same configured dimensionality, so the
vector store and the collection schema do not change when you switch between them.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashingEmbeddings:
    """A hashed bag of words projected into a fixed dimensional space.

    Each token is hashed to a bucket and a sign, then accumulated with a sublinear
    term frequency weight. The result behaves like a lexical similarity model: it will
    not capture synonyms, but it retrieves reliably on shared vocabulary and it is
    completely deterministic, which makes retrieval assertions in tests meaningful.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return value % self.dimensions, sign

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts: dict[str, int] = {}
        for token in _TOKEN_RE.findall(text.lower()):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            bucket, sign = self._bucket(token)
            vector[bucket] += sign * (1.0 + math.log(count))
        return _l2_normalize(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class SentenceTransformerEmbeddings:
    """Wraps a sentence-transformers model. Requires the ``embeddings`` extra."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def build_embeddings(
    backend: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dimensions: int = 384,
) -> EmbeddingProvider:
    """Return an embedding provider, falling back to hashing when needed."""
    if backend == "hashing":
        return HashingEmbeddings(dimensions=dimensions)

    if backend in ("auto", "sentence-transformers"):
        try:
            provider = SentenceTransformerEmbeddings(model_name=model_name)
            logger.info("rag layer: using sentence-transformers embeddings")
            return provider
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            logger.info("rag layer: sentence-transformers unavailable (%s), hashing", exc)

    return HashingEmbeddings(dimensions=dimensions)
