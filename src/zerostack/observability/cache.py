"""A semantic answer cache.

An exact string cache almost never hits on natural language, because two people
asking the same thing rarely type the same characters. Matching on embedding
similarity instead means "what is our refund limit" can serve the answer already
computed for "how much can an agent refund", which is where the savings are.

Every entry records the sources its answer was grounded in, so an ingestion that
touches those sources can invalidate exactly the affected entries rather than
dropping the whole cache. A cache that keeps serving answers from documents that
have since changed is worse than no cache, because it is confidently stale.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CacheEntry:
    """One cached answer and everything needed to decide if it is still valid."""

    question: str
    answer: str
    embedding: list[float]
    sources: list[str] = field(default_factory=list)
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheLookup:
    """The outcome of a lookup, including why it missed."""

    entry: CacheEntry | None = None
    similarity: float = 0.0
    reason: str = "miss"

    @property
    def hit(self) -> bool:
        return self.entry is not None


class SemanticCache:
    """Bounded, similarity matched answer cache with source invalidation."""

    def __init__(
        self,
        threshold: float = 0.95,
        max_entries: int = 500,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def lookup(self, embedding: list[float], namespace: str = "default") -> CacheLookup:
        """Find an entry close enough to serve.

        The namespace is matched exactly rather than by similarity. Tenancy is a
        boundary, not a resemblance, and serving one tenant an answer computed
        for another because the questions looked alike would be a data leak.
        """
        best: CacheEntry | None = None
        best_similarity = 0.0
        expired: list[str] = []

        with self._lock:
            for key, entry in self._entries.items():
                if entry.namespace != namespace:
                    continue
                if self.ttl_seconds and entry.age_seconds() > self.ttl_seconds:
                    expired.append(key)
                    continue
                similarity = _cosine(embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best = entry
            for key in expired:
                self._entries.pop(key, None)

            if best is not None and best_similarity >= self.threshold:
                best.hits += 1
                self._entries.move_to_end(_key(best.question, best.namespace))
                return CacheLookup(entry=best, similarity=best_similarity, reason="hit")

        if best is None:
            return CacheLookup(reason="empty")
        return CacheLookup(similarity=best_similarity, reason="below_threshold")

    def store(
        self,
        question: str,
        answer: str,
        embedding: list[float],
        sources: list[str] | None = None,
        namespace: str = "default",
    ) -> None:
        entry = CacheEntry(
            question=question,
            answer=answer,
            embedding=embedding,
            sources=list(sources or []),
            namespace=namespace,
        )
        with self._lock:
            self._entries[_key(question, namespace)] = entry
            self._entries.move_to_end(_key(question, namespace))
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def invalidate_sources(self, sources: list[str]) -> int:
        """Drop every entry grounded in any of these sources.

        Called after ingestion. Without it the cache keeps serving answers built
        from a version of a document that no longer exists.
        """
        touched = set(sources)
        with self._lock:
            doomed = [
                key for key, entry in self._entries.items() if touched.intersection(entry.sources)
            ]
            for key in doomed:
                del self._entries[key]
        return len(doomed)

    def invalidate_ungrounded(self, namespace: str) -> int:
        """Drop entries in ``namespace`` that were answered from no sources.

        An answer with empty provenance was produced because retrieval found
        nothing. It records no source, so source based invalidation can never
        match it, and it would keep reporting "nothing found" for the rest of its
        TTL after the very document that answers it was ingested. The absence
        that justified the answer is exactly what ingestion changes.
        """
        with self._lock:
            doomed = [
                key
                for key, entry in self._entries.items()
                if entry.namespace == namespace and not entry.sources
            ]
            for key in doomed:
                del self._entries[key]
        return len(doomed)

    def invalidate_namespace(self, namespace: str) -> int:
        with self._lock:
            doomed = [key for key, entry in self._entries.items() if entry.namespace == namespace]
            for key in doomed:
                del self._entries[key]
        return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            total_hits = sum(entry.hits for entry in self._entries.values())
            namespaces = sorted({entry.namespace for entry in self._entries.values()})
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "served_from_cache": total_hits,
            "namespaces": namespaces,
        }


def _key(question: str, namespace: str) -> str:
    return f"{namespace}::{question}"
