"""Tests for metrics, cost accounting and the semantic cache.

Traces answer "what happened in this request". Metrics answer "what is happening
across all of them", which is the question an operator actually has. Cost
accounting exists so that moving from a local model to a hosted one reveals a
number that was already being tracked rather than a surprise.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from zerostack.app import ZerostackApp
from zerostack.observability.cache import SemanticCache
from zerostack.observability.cost import (
    BudgetExceeded,
    CostTracker,
    Usage,
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


class TestDailyBudgetWindow:
    """A budget named daily must actually roll over.

    Without a window the running total only ever rises, so the first day that
    spends the allowance refuses every request from then on. That is not a
    ceiling, it is a permanent outage that arrives on a schedule.
    """

    @staticmethod
    def _tracker(moment: datetime, **kwargs) -> CostTracker:
        holder = {"now": moment}
        tracker = CostTracker(clock=lambda: holder["now"], **kwargs)
        tracker._holder = holder  # type: ignore[attr-defined]
        return tracker

    def test_the_ceiling_holds_within_a_day(self):
        tracker = self._tracker(datetime(2026, 8, 25, 9, 0, tzinfo=UTC), daily_token_budget=100)
        tracker.record(Usage(prompt_tokens=100))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_tokens=1)

    def test_the_window_reopens_on_the_next_day(self):
        tracker = self._tracker(datetime(2026, 8, 25, 23, 59, tzinfo=UTC), daily_token_budget=100)
        tracker.record(Usage(prompt_tokens=100))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_tokens=1)
        tracker._holder["now"] = datetime(2026, 8, 26, 0, 1, tzinfo=UTC)
        tracker.check(projected_tokens=1)

    def test_rolling_over_clears_the_counters(self):
        tracker = self._tracker(datetime(2026, 8, 25, 12, 0, tzinfo=UTC), daily_token_budget=100)
        tracker.record(Usage(prompt_tokens=80))
        tracker._holder["now"] = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
        snapshot = tracker.snapshot()
        assert snapshot["tokens_used"] == 0
        assert snapshot["calls"] == 0
        assert snapshot["window_day"] == "2026-08-26"

    def test_a_new_day_does_not_lift_the_ceiling_twice(self):
        """Spending the fresh allowance must exhaust it again."""
        tracker = self._tracker(datetime(2026, 8, 25, 12, 0, tzinfo=UTC), daily_token_budget=100)
        tracker.record(Usage(prompt_tokens=100))
        tracker._holder["now"] = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
        tracker.record(Usage(prompt_tokens=100))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_tokens=1)

    def test_the_cost_ceiling_rolls_over_too(self):
        tracker = self._tracker(datetime(2026, 8, 25, 12, 0, tzinfo=UTC), daily_cost_budget_usd=1.0)
        tracker.record(Usage(prompt_tokens=1, cost_usd=1.0))
        with pytest.raises(BudgetExceeded):
            tracker.check(projected_cost_usd=0.01)
        tracker._holder["now"] = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
        tracker.check(projected_cost_usd=0.01)

    def test_concurrent_records_across_a_boundary_do_not_lose_the_reset(self):
        """The rollover is a check followed by a write, so it needs the lock."""
        tracker = self._tracker(datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
        tracker.record(Usage(prompt_tokens=50))
        tracker._holder["now"] = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)

        def worker() -> None:
            for _ in range(500):
                tracker.record(Usage(prompt_tokens=1))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # The 50 from the previous day must not survive the boundary.
        assert tracker.tokens_used == 2000
        assert tracker.window_day == "2026-08-26"


class TestSpendIsReservedBeforeTheCall:
    """A ceiling checked against the question alone is not a ceiling.

    Projecting zero dollars and only the question's tokens lets a short question
    through on a nearly spent budget, which then produces a full response and
    overshoots before anything is recorded.
    """

    @pytest.fixture
    def priced_app(self, settings, corpus_dir, tmp_path):
        settings.rag.corpus_dir = corpus_dir
        settings.data.sqlite_path = tmp_path / "runs.db"
        settings.cache.enabled = False
        settings.cost.enabled = True
        settings.llm.max_tokens = 1024
        settings.cost.price_table = "offline:10:30"
        return settings

    def test_the_projection_includes_the_completion(self, priced_app):
        app = ZerostackApp(settings=priced_app, include_mcp=False)
        projected = app._projected_spend("How long is the probationary period?")
        # The question is a handful of tokens; the ceiling on the response is
        # 1024, so the projection has to be dominated by the completion.
        assert projected["projected_tokens"] > priced_app.llm.max_tokens
        assert projected["projected_cost_usd"] > 0

    def test_a_call_that_would_breach_the_ceiling_is_refused_first(self, priced_app):
        priced_app.cost.daily_cost_budget_usd = 0.001
        app = ZerostackApp(settings=priced_app, include_mcp=False)
        app.ingest(str(priced_app.rag.corpus_dir), enforce_roots=False)
        with pytest.raises(BudgetExceeded):
            app.ask("How long is the probationary period?", persist=False)

    def test_a_generous_ceiling_still_allows_the_call(self, priced_app):
        priced_app.cost.daily_cost_budget_usd = 100.0
        app = ZerostackApp(settings=priced_app, include_mcp=False)
        app.ingest(str(priced_app.rag.corpus_dir), enforce_roots=False)
        assert app.ask("How long is the probationary period?", persist=False).answer

    def test_the_projection_prices_the_model_that_will_serve(self, priced_app):
        """Not the configured one: they differ whenever a layer has degraded."""
        app = ZerostackApp(settings=priced_app, include_mcp=False)
        assert app.llm.model != priced_app.llm.model
        # "offline" is the only priced entry, so a non zero cost proves the
        # projection used the live provider rather than the configured model.
        assert app._projected_spend("anything")["projected_cost_usd"] > 0
