"""Evaluation: measuring whether a change helped or hurt.

Every other layer can be tested for correctness. Retrieval quality cannot, in the
same way: there is no exception when the wrong passage is returned, only a worse
answer that still looks plausible. Without measurement, tuning a chunk size or a
fusion weight is guesswork, and a regression ships silently because the tests
still pass.

This harness is deterministic and offline like the rest of the stack, so it runs
in CI and can gate a merge.
"""

from zerostack.evals.dataset import EvalCase, EvalDataset, load_dataset
from zerostack.evals.metrics import (
    RetrievalScores,
    answer_contains,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    score_retrieval,
)
from zerostack.evals.runner import EvalReport, EvalResult, run_evaluation

__all__ = [
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "EvalResult",
    "RetrievalScores",
    "answer_contains",
    "load_dataset",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "run_evaluation",
    "score_retrieval",
]
