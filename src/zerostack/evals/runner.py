"""Running an evaluation and reporting the result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zerostack.evals.dataset import EvalCase, EvalDataset
from zerostack.evals.metrics import RetrievalScores, answer_contains, score_retrieval

if TYPE_CHECKING:
    from zerostack.app import ZerostackApp


@dataclass
class EvalResult:
    """The outcome for one case."""

    case: EvalCase
    answer: str
    retrieved_sources: list[str]
    scores: RetrievalScores
    answer_ok: bool
    missing_phrases: list[str] = field(default_factory=list)
    leaked_phrases: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.answer_ok and not self.leaked_phrases

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.case.question,
            "passed": self.passed,
            "answer_ok": self.answer_ok,
            "missing_phrases": self.missing_phrases,
            "leaked_phrases": self.leaked_phrases,
            "retrieved": self.retrieved_sources,
            "expected": self.case.expected_sources,
            "scores": self.scores.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregate results across a dataset."""

    dataset: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def _mean(self, attribute: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r.scores, attribute) for r in self.results) / len(self.results)

    @property
    def mean_recall(self) -> float:
        return self._mean("recall")

    @property
    def mean_precision(self) -> float:
        return self._mean("precision")

    @property
    def mean_reciprocal_rank(self) -> float:
        return self._mean("reciprocal_rank")

    @property
    def failures(self) -> list[EvalResult]:
        return [result for result in self.results if not result.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            "results": [result.to_dict() for result in self.results],
        }

    def summary_line(self) -> str:
        return (
            f"{self.dataset}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%}) | recall {self.mean_recall:.2f} "
            f"| precision {self.mean_precision:.2f} | mrr {self.mean_reciprocal_rank:.2f}"
        )

    def meets(self, min_pass_rate: float = 0.0, min_recall: float = 0.0) -> bool:
        """Whether this run clears its thresholds. Used to gate a merge."""
        return self.pass_rate >= min_pass_rate and self.mean_recall >= min_recall


def run_evaluation(app: ZerostackApp, dataset: EvalDataset, top_k: int | None = None) -> EvalReport:
    """Run every case and collect the results.

    Caching is disabled for the run. A cache hit would return an earlier answer
    and measure the cache rather than the retrieval configuration under test,
    which is the one thing an evaluation must not do.
    """
    report = EvalReport(dataset=dataset.name)

    for case in dataset.cases:
        started = time.perf_counter()
        try:
            retrieved = app.rag.retrieve(case.question, top_k=top_k, namespace=case.namespace)
            result = app.ask(
                case.question,
                persist=False,
                use_cache=False,
                namespace=case.namespace,
            )
            answer = result.answer
            error = None
        except Exception as exc:
            retrieved = []
            answer = ""
            error = f"{type(exc).__name__}: {exc}"

        sources = [hit.source for hit in retrieved]
        answer_ok, missing = answer_contains(answer, case.expected_phrases)
        leaked = [phrase for phrase in case.forbidden_phrases if phrase.lower() in answer.lower()]

        report.results.append(
            EvalResult(
                case=case,
                answer=answer,
                retrieved_sources=sources,
                scores=score_retrieval(sources, case.expected_sources),
                answer_ok=answer_ok and error is None,
                missing_phrases=missing,
                leaked_phrases=leaked,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
        )

    return report
