"""Retrieval and answer metrics.

Three retrieval metrics, because each hides a different failure:

Recall asks whether the right document came back at all. A system can score well
on it while burying the answer at rank twenty, where no prompt window will
include it.

Precision asks how much of what came back was useful. A system can score well by
returning one correct chunk and nothing else, while missing three other
documents that also mattered.

Mean reciprocal rank asks how near the top the first correct result was, which is
what actually decides whether a fact reaches the model's context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalScores:
    """Scores for one question."""

    recall: float = 0.0
    precision: float = 0.0
    reciprocal_rank: float = 0.0
    retrieved: int = 0
    expected: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "retrieved": self.retrieved,
            "expected": self.expected,
        }


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    """Fraction of the expected sources that were retrieved."""
    if not expected:
        return 1.0
    found = {source for source in retrieved if source in set(expected)}
    return len(found) / len(set(expected))


def precision_at_k(retrieved: list[str], expected: list[str]) -> float:
    """Fraction of the retrieved sources that were expected."""
    if not retrieved:
        return 0.0
    expected_set = set(expected)
    return sum(1 for source in retrieved if source in expected_set) / len(retrieved)


def mean_reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """One over the rank of the first correct source, or zero if none is present."""
    expected_set = set(expected)
    for position, source in enumerate(retrieved, start=1):
        if source in expected_set:
            return 1.0 / position
    return 0.0


def score_retrieval(retrieved: list[str], expected: list[str]) -> RetrievalScores:
    # Deduplicated while preserving order, because the same document appearing
    # as three chunks is one document for the purpose of these metrics and would
    # otherwise distort precision.
    unique = list(dict.fromkeys(retrieved))
    return RetrievalScores(
        recall=recall_at_k(unique, expected),
        precision=precision_at_k(unique, expected),
        reciprocal_rank=mean_reciprocal_rank(unique, expected),
        retrieved=len(unique),
        expected=len(set(expected)),
    )


def answer_contains(answer: str, phrases: list[str]) -> tuple[bool, list[str]]:
    """Check an answer for required phrases, case insensitively.

    Substring matching rather than a model graded score, deliberately. A graded
    score needs a model, which puts the harness behind a model server and makes
    the result nondeterministic, so it could not gate a merge. Substring matching
    is blunt but reproducible, and a phrase list is easy to review.
    """
    if not phrases:
        return True, []
    lowered = answer.lower()
    missing = [phrase for phrase in phrases if phrase.lower() not in lowered]
    return not missing, missing
