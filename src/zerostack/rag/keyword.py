"""BM25 keyword retrieval, the lexical half of hybrid search.

Vector search finds passages that mean something similar. Keyword search finds
passages that contain the exact terms. Each fails where the other succeeds: a
vector index misses a part number, an error code or a policy name it never saw
in training, and a keyword index misses a question phrased entirely in synonyms.

Running both and fusing the results is the single largest retrieval quality
improvement available without a reranking model, which is why the enterprise
tier of this architecture keeps a keyword tier alongside its vector tier.

This is a pure Python implementation with no dependencies, so it works on the
offline path like everything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from zerostack.rag.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Terms that appear in nearly every document carry no discriminating signal and
# would otherwise dominate short queries.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have how
    i if in into is it its of on or our that the their them then there these they
    this to was were what when where which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, drop stopwords."""
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


@dataclass
class KeywordHit:
    """One BM25 match."""

    chunk_id: str
    score: float


@dataclass
class BM25Index:
    """An in memory BM25 Okapi index.

    ``k1`` controls term frequency saturation: how quickly repeating a term stops
    adding to the score. ``b`` controls length normalisation: how much a long
    document is penalised for having more chances to contain the term. The
    defaults are the values BM25 is normally published with and are a reasonable
    starting point for prose.
    """

    k1: float = 1.5
    b: float = 0.75

    _documents: dict[str, list[str]] = field(default_factory=dict)
    _frequencies: dict[str, Counter[str]] = field(default_factory=dict)
    _document_frequency: Counter[str] = field(default_factory=Counter)
    _lengths: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def average_length(self) -> float:
        if not self._lengths:
            return 0.0
        return sum(self._lengths.values()) / len(self._lengths)

    def add(self, chunk_id: str, text: str) -> None:
        """Index one chunk, replacing it if the id is already present."""
        if chunk_id in self._documents:
            self.remove(chunk_id)
        tokens = tokenize(text)
        self._documents[chunk_id] = tokens
        counts = Counter(tokens)
        self._frequencies[chunk_id] = counts
        self._lengths[chunk_id] = len(tokens)
        for term in counts:
            self._document_frequency[term] += 1

    def add_chunks(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self.add(chunk.chunk_id, chunk.text)
        return len(chunks)

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk and decrement the document frequencies it contributed."""
        if chunk_id not in self._documents:
            return
        for term in self._frequencies[chunk_id]:
            self._document_frequency[term] -= 1
            if self._document_frequency[term] <= 0:
                del self._document_frequency[term]
        del self._documents[chunk_id]
        del self._frequencies[chunk_id]
        del self._lengths[chunk_id]

    def clear(self) -> None:
        self._documents.clear()
        self._frequencies.clear()
        self._document_frequency.clear()
        self._lengths.clear()

    def _idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero.

        The standard BM25 idf goes negative for a term present in more than half
        the corpus. On a small index that is common and lets a ubiquitous term
        actively subtract from a score, so it is clamped.
        """
        document_frequency = self._document_frequency.get(term, 0)
        if document_frequency == 0:
            return 0.0
        total = len(self._documents)
        value = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
        return max(0.0, value)

    def search(self, query: str, top_k: int = 10) -> list[KeywordHit]:
        """Return the best matching chunks, highest score first."""
        terms = tokenize(query)
        if not terms or not self._documents:
            return []

        average = self.average_length or 1.0
        scores: dict[str, float] = {}

        for term in set(terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for chunk_id, counts in self._frequencies.items():
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                length = self._lengths[chunk_id]
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / average)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    frequency * (self.k1 + 1) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [KeywordHit(chunk_id=chunk_id, score=score) for chunk_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Combine several ranked lists into one.

    Reciprocal rank fusion scores each item by 1/(k + rank) summed across the
    lists it appears in. It is used here rather than combining the raw scores
    because BM25 scores and cosine similarities are on different, unnormalised
    scales, so adding them directly would let whichever happens to be larger
    dominate. Rank position is comparable across systems; raw score is not.

    ``k`` damps the influence of the very top positions. The value 60 is the one
    the technique was published with and behaves well without tuning.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + position)

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
