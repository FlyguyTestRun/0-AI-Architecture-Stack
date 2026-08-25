"""Tests for the evaluation harness.

The harness is what makes retrieval quality a measured property rather than an
assumed one. Every other layer fails loudly when it breaks; retrieval returns a
worse answer that still looks plausible, so it needs measuring.
"""

from __future__ import annotations

import json

import pytest

from zerostack.app import ZerostackApp
from zerostack.evals import (
    EvalCase,
    EvalDataset,
    load_dataset,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    run_evaluation,
    score_retrieval,
)
from zerostack.evals.metrics import answer_contains


class TestMetrics:
    def test_perfect_recall(self):
        assert recall_at_k(["a.md", "b.md"], ["a.md"]) == 1.0

    def test_missed_recall(self):
        assert recall_at_k(["b.md"], ["a.md"]) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a.md"], ["a.md", "b.md"]) == 0.5

    def test_no_expectation_counts_as_satisfied(self):
        """A question with no expected source cannot fail on recall."""
        assert recall_at_k([], []) == 1.0

    def test_precision_penalises_noise(self):
        assert precision_at_k(["a.md", "x.md"], ["a.md"]) == 0.5

    def test_precision_of_nothing_is_zero(self):
        assert precision_at_k([], ["a.md"]) == 0.0

    def test_reciprocal_rank_rewards_the_top_position(self):
        assert mean_reciprocal_rank(["a.md", "b.md"], ["a.md"]) == 1.0
        assert mean_reciprocal_rank(["b.md", "a.md"], ["a.md"]) == 0.5

    def test_reciprocal_rank_is_zero_when_absent(self):
        assert mean_reciprocal_rank(["x.md"], ["a.md"]) == 0.0

    def test_repeated_chunks_of_one_document_count_once(self):
        """Three chunks of one file is one document for these metrics."""
        scores = score_retrieval(["a.md", "a.md", "a.md"], ["a.md"])
        assert scores.precision == 1.0
        assert scores.retrieved == 1

    def test_answer_contains_is_case_insensitive(self):
        ok, missing = answer_contains("Two Hundred Dollars", ["two hundred dollars"])
        assert ok and missing == []

    def test_answer_contains_reports_what_is_missing(self):
        ok, missing = answer_contains("nothing useful", ["timeline", "root cause"])
        assert not ok
        assert missing == ["timeline", "root cause"]

    def test_no_required_phrases_passes(self):
        assert answer_contains("anything", [])[0]


class TestDataset:
    def test_loads_from_json(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(
            json.dumps(
                {
                    "name": "sample",
                    "cases": [
                        {
                            "question": "q",
                            "expected_sources": ["a.md"],
                            "expected_phrases": ["x"],
                            "tags": ["policy"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        dataset = load_dataset(path)
        assert dataset.name == "sample"
        assert len(dataset) == 1
        assert dataset.cases[0].expected_sources == ["a.md"]

    def test_filters_by_tag(self):
        dataset = EvalDataset(
            name="d",
            cases=[
                EvalCase(question="a", tags=["policy"]),
                EvalCase(question="b", tags=["tools"]),
            ],
        )
        assert len(dataset.filter_by_tag("policy")) == 1

    def test_the_shipped_baseline_loads(self):
        from zerostack.config import REPO_ROOT

        dataset = load_dataset(REPO_ROOT / "evals" / "baseline.json")
        assert len(dataset) >= 5


class TestRunner:
    @pytest.fixture
    def app(self, settings, corpus_dir) -> ZerostackApp:
        settings.rag.corpus_dir = corpus_dir
        instance = ZerostackApp(settings=settings, include_mcp=False)
        instance.ingest()
        return instance

    def test_a_satisfied_case_passes(self, app):
        dataset = EvalDataset(
            name="t",
            cases=[
                EvalCase(
                    question="How much can an agent refund without approval?",
                    expected_sources=["support.md"],
                    expected_phrases=["two hundred dollars"],
                )
            ],
        )
        report = run_evaluation(app, dataset)
        assert report.pass_rate == 1.0
        assert report.mean_recall == 1.0

    def test_a_missing_phrase_fails_the_case(self, app):
        dataset = EvalDataset(
            name="t",
            cases=[
                EvalCase(
                    question="How much can an agent refund?",
                    expected_phrases=["a phrase that is definitely absent"],
                )
            ],
        )
        report = run_evaluation(app, dataset)
        assert report.pass_rate == 0.0
        assert report.failures[0].missing_phrases

    def test_a_forbidden_phrase_fails_the_case(self, app):
        """Pins a closed leak so it cannot come back unnoticed."""
        dataset = EvalDataset(
            name="t",
            cases=[
                EvalCase(
                    question="How much can an agent refund without approval?",
                    forbidden_phrases=["two hundred dollars"],
                )
            ],
        )
        assert run_evaluation(app, dataset).failures[0].leaked_phrases

    def test_the_cache_is_bypassed(self, app):
        """A cache hit would measure the cache, not the configuration under test."""
        question = "How much can an agent refund without approval?"
        app.ask(question)
        dataset = EvalDataset(
            name="t",
            cases=[EvalCase(question=question, expected_sources=["support.md"])],
        )
        report = run_evaluation(app, dataset)
        assert report.results[0].retrieved_sources

    def test_evaluation_does_not_pollute_the_run_log(self, app):
        before = len(app.recent_runs(limit=100))
        run_evaluation(app, EvalDataset(name="t", cases=[EvalCase(question="anything")]))
        assert len(app.recent_runs(limit=100)) == before

    def test_thresholds_gate_the_result(self, app):
        dataset = EvalDataset(
            name="t",
            cases=[
                EvalCase(
                    question="How much can an agent refund without approval?",
                    expected_sources=["support.md"],
                    expected_phrases=["two hundred dollars"],
                )
            ],
        )
        report = run_evaluation(app, dataset)
        assert report.meets(min_pass_rate=1.0, min_recall=1.0)
        assert not report.meets(min_pass_rate=1.1)

    def test_the_report_is_json_serialisable(self, app):
        report = run_evaluation(app, EvalDataset(name="t", cases=[EvalCase(question="anything")]))
        json.dumps(report.to_dict())

    def test_the_summary_line_reports_every_headline_metric(self, app):
        report = run_evaluation(app, EvalDataset(name="t", cases=[EvalCase(question="anything")]))
        for token in ("passed", "recall", "precision", "mrr"):
            assert token in report.summary_line()
