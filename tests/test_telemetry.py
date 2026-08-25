"""Tests for metrics, cost accounting and the semantic cache.

Traces answer "what happened in this request". Metrics answer "what is happening
across all of them", which is the question an operator actually has. Cost
accounting exists so that moving from a local model to a hosted one reveals a
number that was already being tracked rather than a surprise.
"""

from __future__ import annotations

import threading

import pytest

from zerostack.observability.cache import SemanticCache
from zerostack.observability.cost import (
    BudgetExceeded,
    CostTracker,
    estimate_tokens,
)
from zerostack.observability.metrics import MetricsRegistry
from zerostack.rag.embeddings import HashingEmbeddings


class TestMetrics:
    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        return MetricsRegistry()

    def test_counters_accumulate(self, registry):
        registry.increment("requests")
        registry.increment("requests", 2)
        assert registry.counter_value("requests") == 3

    def test_labels_separate_series(self, registry):
        registry.increment("requests", labels={"namespace": "hr"})
        registry.increment("requests", labels={"namespace": "legal"})
        assert registry.counter_value("requests", {"namespace": "hr"}) == 1
        assert registry.counter_value("requests", {"namespace": "legal"}) == 1

    def test_label_order_does_not_create_a_new_series(self, registry):
        registry.increment("r", labels={"a": "1", "b": "2"})
        registry.increment("r", labels={"b": "2", "a": "1"})
        assert registry.counter_value("r", {"a": "1", "b": "2"}) == 2

    def test_gauges_replace_rather_than_accumulate(self, registry):
        registry.set_gauge("indexed", 5)
        registry.set_gauge("indexed", 9)
        assert registry.gauge_value("indexed") == 9

    def test_histogram_records_count_and_sum(self, registry):
        for value in (0.1, 0.2, 0.3):
            registry.observe("latency", value)
        histogram = registry.histogram("latency")
        assert histogram.total == 3
        assert histogram.average == pytest.approx(0.2)

    def test_quantiles_track_the_tail(self, registry):
        for _ in range(99):
            registry.observe("latency", 0.01)
        registry.observe("latency", 10.0)
        assert registry.histogram("latency").quantile(0.99) <= 10.0

    def test_quantile_of_an_empty_histogram_is_zero(self, registry):
        registry.observe("latency", 1.0)
        assert MetricsRegistry().histogram("nothing") is None

    def test_the_timer_records_even_when_the_block_raises(self, registry):
        """A request that fails slowly is the interesting one."""
        with pytest.raises(ValueError):
            with registry.timer("latency"):
                raise ValueError("boom")
        assert registry.histogram("latency").total == 1

    def test_concurrent_increments_are_not_lost(self, registry):
        def bump() -> None:
            for _ in range(200):
                registry.increment("requests")

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert registry.counter_value("requests") == 800

    def test_snapshot_is_plain_data(self, registry):
        registry.increment("requests")
        registry.observe("latency", 0.5)
        snapshot = registry.snapshot()
        assert "requests" in snapshot["counters"]
        assert snapshot["histograms"]["latency"]["count"] == 1

    def test_reset_clears_everything(self, registry):
        registry.increment("requests")
        registry.reset()
        assert registry.counter_value("requests") == 0


class TestPrometheusFormat:
    @pytest.fixture
    def registry(self) -> MetricsRegistry:
        registry = MetricsRegistry()
        registry.describe("requests_total", "Requests served.", "counter")
        registry.increment("requests_total", labels={"namespace": "hr"})
        registry.observe("latency_seconds", 0.02)
        registry.set_gauge("indexed", 7)
        return registry

    def test_emits_help_and_type_lines(self, registry):
        rendered = registry.render_prometheus()
        assert "# HELP requests_total Requests served." in rendered
        assert "# TYPE requests_total counter" in rendered

    def test_labels_are_rendered(self, registry):
        assert 'requests_total{namespace="hr"} 1.0' in registry.render_prometheus()

    def test_histogram_buckets_are_cumulative(self, registry):
        lines = [
            line
            for line in registry.render_prometheus().splitlines()
            if line.startswith("latency_seconds_bucket")
        ]
        counts = [int(line.rsplit(" ", 1)[1]) for line in lines]
        assert counts == sorted(counts), "bucket counts must not decrease"

    def test_histogram_emits_sum_and_count(self, registry):
        rendered = registry.render_prometheus()
        assert "latency_seconds_sum" in rendered
        assert "latency_seconds_count" in rendered

    def test_an_infinite_bucket_is_emitted(self, registry):
        assert 'latency_seconds_bucket{le="+Inf"}' in registry.render_prometheus()

    def test_label_values_are_escaped(self):
        registry = MetricsRegistry()
        registry.increment("m", labels={"path": 'a"b'})
        assert '\\"' in registry.render_prometheus()


class TestCostTracking:
    def test_estimates_scale_with_length(self):
        assert estimate_tokens("one two three") > estimate_tokens("one")

    def test_empty_text_is_zero_tokens(self):
        assert estimate_tokens("") == 0

    def test_prices_are_applied_per_million(self):
        tracker = CostTracker(prices={"m": (3.0, 15.0)})
        usage = tracker.price(1_000_000, 1_000_000, "m")
        assert usage.cost_usd == pytest.approx(18.0)

    def test_a_versioned_model_matches_its_prefix(self):
        """Version suffixes change more often than pricing."""
        tracker = CostTracker(prices={"hosted": (1.0, 2.0)})
        assert tracker.price_for("hosted-v3-20260801") == (1.0, 2.0)

    def test_the_longest_matching_prefix_wins(self):
        tracker = CostTracker(prices={"a": (1.0, 1.0), "a-large": (9.0, 9.0)})
        assert tracker.price_for("a-large-v2") == (9.0, 9.0)

    def test_an_unknown_model_is_free_rather_than_an_error(self):
        assert CostTracker().price_for("something-unheard-of") == (0.0, 0.0)

    def test_a_token_ceiling_blocks_before_the_call(self):
        tracker = CostTracker(daily_token_budget=100)
        tracker.record(tracker.price(90, 0, "local"))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_tokens=20)

    def test_a_cost_ceiling_blocks_before_the_call(self):
        tracker = CostTracker(prices={"m": (1000.0, 1000.0)}, daily_cost_budget_usd=0.001)
        tracker.record(tracker.price(1000, 0, "m"))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_cost_usd=1.0)

    def test_no_ceiling_means_no_limit(self):
        tracker = CostTracker()
        tracker.record(tracker.price(10_000_000, 0, "local"))
        tracker.check(projected_tokens=10_000_000)

    def test_snapshot_reports_what_remains(self):
        tracker = CostTracker(daily_token_budget=100)
        tracker.record(tracker.price(40, 0, "local"))
        assert tracker.snapshot()["tokens_remaining"] == 60


class TestSemanticCache:
    @pytest.fixture
    def embeddings(self) -> HashingEmbeddings:
        return HashingEmbeddings(dimensions=256)

    @pytest.fixture
    def cache(self, embeddings) -> SemanticCache:
        cache = SemanticCache(threshold=0.75)
        question = "How much can a support agent refund without approval?"
        cache.store(
            question, "Two hundred dollars.", embeddings.embed_one(question), sources=["support.md"]
        )
        return cache

    def test_an_identical_question_hits(self, cache, embeddings):
        question = "How much can a support agent refund without approval?"
        assert cache.lookup(embeddings.embed_one(question)).hit

    def test_an_unrelated_question_misses(self, cache, embeddings):
        lookup = cache.lookup(embeddings.embed_one("What is the vacation policy?"))
        assert not lookup.hit
        assert lookup.reason == "below_threshold"

    def test_a_namespace_is_a_boundary_not_a_resemblance(self, cache, embeddings):
        """Serving one tenant an answer computed for another is a data leak."""
        question = "How much can a support agent refund without approval?"
        assert not cache.lookup(embeddings.embed_one(question), namespace="other").hit

    def test_ingestion_invalidates_affected_entries(self, cache):
        assert cache.invalidate_sources(["support.md"]) == 1
        assert cache.size == 0

    def test_unrelated_sources_do_not_invalidate(self, cache):
        assert cache.invalidate_sources(["unrelated.md"]) == 0
        assert cache.size == 1

    def test_the_cache_is_bounded(self, embeddings):
        cache = SemanticCache(max_entries=3)
        for index in range(10):
            cache.store(f"q{index}", "a", embeddings.embed_one(f"q{index}"))
        assert cache.size == 3

    def test_expired_entries_are_not_served(self, embeddings):
        cache = SemanticCache(threshold=0.5, ttl_seconds=-1.0)
        cache.store("q", "a", embeddings.embed_one("q"))
        assert not cache.lookup(embeddings.embed_one("q")).hit

    def test_an_empty_cache_reports_why(self, embeddings):
        assert SemanticCache().lookup(embeddings.embed_one("q")).reason == "empty"

    def test_an_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=0.0)

    def test_snapshot_counts_what_was_served(self, cache, embeddings):
        question = "How much can a support agent refund without approval?"
        cache.lookup(embeddings.embed_one(question))
        assert cache.snapshot()["served_from_cache"] == 1
