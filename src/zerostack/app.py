"""The application object that wires all eight layers together.

Everything above this module (the API, the CLI, the Streamlit frontend) depends only
on :class:`ZerostackApp`. That is what allows the frontend layer to be swapped from
Streamlit to Next.js without touching any of the layers below it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zerostack.config import SecuritySettings, Settings, get_settings
from zerostack.data import RunRecord, StateStore, analytics_summary
from zerostack.llm import build_llm
from zerostack.llm.ollama import OllamaLLM
from zerostack.observability import (
    CostTracker,
    SemanticCache,
    configure_logging,
    get_metrics,
    get_tracer,
)
from zerostack.observability.cost import DEFAULT_PRICES
from zerostack.orchestrator import (
    AgentContext,
    AgentResult,
    AgentStep,
    build_orchestrator,
)
from zerostack.orchestrator.factory import available_engines
from zerostack.rag import RAGPipeline
from zerostack.security import PrincipalStore, RateLimiter, normalise_namespace
from zerostack.tools import build_registry

logger = logging.getLogger(__name__)


class ZerostackApp:
    """Composition root. Construct once per process and reuse."""

    def __init__(self, settings: Settings | None = None, include_mcp: bool = True) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.observability.log_level)

        self.llm = build_llm(self.settings.llm)
        self.rag = RAGPipeline(settings=self.settings.rag)
        self.tools = build_registry(include_mcp=include_mcp)
        self.store = StateStore(self.settings.data.sqlite_path)
        self.metrics = get_metrics()
        self.costs = CostTracker(
            prices=_parse_prices(self.settings.cost.price_table),
            daily_token_budget=self.settings.cost.daily_token_budget,
            daily_cost_budget_usd=self.settings.cost.daily_cost_budget_usd,
        )
        self.principals = _build_principals(self.settings.security)
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.settings.security.requests_per_minute,
            burst=self.settings.security.rate_limit_burst,
        )
        self.cache = SemanticCache(
            threshold=self.settings.cache.threshold,
            max_entries=self.settings.cache.max_entries,
            ttl_seconds=self.settings.cache.ttl_seconds,
        )

        self.context = AgentContext(
            llm=self.llm,
            rag=self.rag,
            tools=self.tools,
            settings=self.settings.orchestrator,
        )
        self.orchestrator = build_orchestrator(self.context, self.settings.orchestrator)

    def ask(
        self,
        question: str,
        persist: bool = True,
        use_cache: bool = True,
        namespace: str | None = None,
    ) -> AgentResult:
        """Run one question through the agent and record the result."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        question = question.strip()

        # Normalised before it reaches a metric label, a cache partition or a
        # retrieval filter. The API validates too, but the CLI and the frontend
        # call straight into here, so the guarantee has to live at this level.
        namespace = normalise_namespace(namespace or self.settings.security.default_namespace)
        self.metrics.increment("zerostack_requests_total", labels={"namespace": namespace})

        cached = self._cache_lookup(question, namespace) if use_cache else None
        if cached is not None:
            # A cache hit is still a question somebody asked. Returning early
            # without recording it would leave the run log, and therefore the
            # audit trail, silently incomplete the moment caching is enabled.
            if persist:
                self._persist(cached)
            return cached

        self.context.namespace = namespace

        if self.settings.cost.enabled:
            # Checked before the call so the ceiling is a limit rather than a
            # report of the overspend after the fact.
            self.costs.check(projected_tokens=len(question.split()) * 2)

        try:
            with self.metrics.timer("zerostack_request_seconds"):
                result = self.orchestrator.run(question)
        except Exception:
            self.metrics.increment("zerostack_request_errors_total")
            raise

        self._account(question, result)
        self._cache_store(question, result, namespace)

        if persist:
            self._persist(result)

        return result

    def _persist(self, result: AgentResult) -> None:
        """Record a run. Never allowed to fail the answer it is recording."""
        try:
            self.store.save_run(
                RunRecord(
                    question=result.question,
                    answer=result.answer,
                    orchestrator=result.orchestrator,
                    llm_provider=result.llm_provider,
                    llm_model=result.llm_model,
                    latency_ms=result.latency_ms,
                    trace_id=result.trace_id,
                    sources=result.sources,
                    steps=[step.to_dict() for step in result.steps],
                )
            )
        except Exception as exc:
            logger.warning("could not persist run: %s", exc)

    def allowed_ingest_roots(self) -> list[Path]:
        """Directories an untrusted caller may ingest from."""
        configured = self.settings.rag.allowed_ingest_roots
        roots = list(configured) if configured else [self.settings.rag.corpus_dir]
        return [root.expanduser().resolve() for root in roots]

    def _check_ingest_allowed(self, target: Path) -> None:
        """Reject a path outside the allowed roots.

        Without this, any caller who can reach the ingest endpoint can index an
        arbitrary readable directory and then read its contents back out through
        the ask endpoint, which turns a document tool into arbitrary file
        disclosure. Resolving first collapses traversal segments and symlinks, so
        the comparison cannot be walked around with "..".
        """
        resolved = target.expanduser().resolve()
        roots = self.allowed_ingest_roots()
        if not any(resolved == root or root in resolved.parents for root in roots):
            # This message names neither the requested path nor the configured
            # roots. The API forwards it verbatim to an unauthenticated caller,
            # so echoing either would turn a refusal into a way to probe the
            # filesystem and learn where the corpus lives. The detail goes to the
            # server log, where an operator can see it and a caller cannot.
            logger.warning(
                "refused ingest of %s, outside allowed roots %s",
                resolved,
                [str(root) for root in roots],
            )
            raise PermissionError(
                "path is outside the allowed ingest roots. See the server log for "
                "the requested path, and set ZEROSTACK_RAG_ALLOWED_INGEST_ROOTS "
                "to widen them."
            )

    def _cache_lookup(self, question: str, namespace: str = "default") -> AgentResult | None:
        """Serve a close enough previous answer, if one exists."""
        if not self.settings.cache.enabled:
            return None
        try:
            embedding = self.rag.embeddings.embed_one(question)
        except Exception as exc:
            logger.debug("cache lookup skipped: %s", exc)
            return None

        # A fresh trace id per request, so two callers served the same cached
        # answer are still distinguishable in the trace and the run log.
        trace_id = get_tracer().new_trace()
        lookup = self.cache.lookup(embedding, namespace=namespace)
        self.metrics.increment(
            "zerostack_cache_events_total",
            labels={"outcome": "hit" if lookup.hit else "miss"},
        )
        if not lookup.hit or lookup.entry is None:
            return None

        entry = lookup.entry
        return AgentResult(
            answer=entry.answer,
            question=question,
            orchestrator=f"{self.orchestrator.name}+cache",
            llm_provider="cache",
            llm_model="cache",
            latency_ms=0.0,
            sources=list(entry.sources),
            steps=[
                AgentStep(
                    name="cache",
                    detail=f"served a previous answer, similarity {lookup.similarity:.3f}",
                    data={"similarity": round(lookup.similarity, 4)},
                )
            ],
            trace_id=trace_id,
        )

    def _cache_store(self, question: str, result: AgentResult, namespace: str = "default") -> None:
        if not self.settings.cache.enabled or not result.answer:
            return
        try:
            self.cache.store(
                question=question,
                answer=result.answer,
                embedding=self.rag.embeddings.embed_one(question),
                sources=result.sources,
                namespace=namespace,
            )
        except Exception as exc:
            logger.debug("cache store skipped: %s", exc)

    def _account(self, question: str, result: AgentResult) -> None:
        """Record token volume and estimated spend for this run."""
        if not self.settings.cost.enabled:
            return
        usage = self.costs.estimate(question, result.answer, result.llm_model)
        self.costs.record(usage)
        self.metrics.increment(
            "zerostack_llm_tokens_total", usage.prompt_tokens, labels={"direction": "prompt"}
        )
        self.metrics.increment(
            "zerostack_llm_tokens_total",
            usage.completion_tokens,
            labels={"direction": "completion"},
        )
        if usage.cost_usd:
            self.metrics.increment("zerostack_llm_cost_usd_total", usage.cost_usd)
        self.metrics.observe("zerostack_llm_seconds", result.latency_ms / 1000.0)

    def ingest(
        self,
        path: Path | str | None = None,
        enforce_roots: bool = True,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a file or directory into the vector store.

        ``enforce_roots`` defaults to True so that every caller is restricted
        unless it deliberately opts out. The CLI opts out; the API does not.
        """
        target = Path(path) if path else self.settings.rag.corpus_dir
        if enforce_roots:
            self._check_ingest_allowed(target)
        if not target.exists():
            raise FileNotFoundError(f"nothing to ingest at {target}")
        namespace = normalise_namespace(namespace or self.settings.security.default_namespace)
        report = self.rag.ingest_path(target, namespace=namespace)

        # An answer built from a document that has since changed is worse than no
        # cache at all, so ingestion drops the entries grounded in what this run
        # rewrote. Passing every source in the store instead would clear the
        # whole cache on each ingestion, which makes caching worthless for any
        # deployment that ingests on a schedule.
        invalidated = self.cache.invalidate_sources(report.sources) if report.sources else 0
        self.metrics.set_gauge("zerostack_documents_indexed", self.rag.store.count())

        return {
            "cache_entries_invalidated": invalidated,
            "namespace": namespace,
            "path": str(target),
            "files": report.files,
            "chunks": report.chunks,
            "skipped": report.skipped,
            "indexed_chunks": self.rag.store.count(),
        }

    def health(self) -> dict[str, Any]:
        """Report which backend is live in every layer. Backs `zerostack doctor`."""
        ollama = OllamaLLM(model=self.settings.llm.model, base_url=self.settings.llm.base_url)
        ollama_up = ollama.available()

        return {
            "app": self.settings.app_name,
            "environment": self.settings.environment,
            "layers": {
                "orchestrator": {
                    "active": self.orchestrator.name,
                    "available": available_engines(),
                },
                "rag": self.rag.describe(),
                "llm": {
                    "active_provider": self.llm.name,
                    "model": self.llm.model,
                    "ollama_reachable": ollama_up,
                    "ollama_models": ollama.installed_models() if ollama_up else [],
                },
                "tools": {
                    "count": len(self.tools),
                    "names": self.tools.names(),
                },
                "data": {
                    "sqlite_path": str(self.settings.data.sqlite_path),
                    "runs": self.store.count_runs(),
                },
                "security": {
                    **self.principals.describe(),
                    "rate_limit": self.rate_limiter.snapshot(),
                    "default_namespace": self.settings.security.default_namespace,
                },
                "cost": self.costs.snapshot(),
                "cache": self.cache.snapshot(),
                "observability": {
                    "enabled": self.settings.observability.enabled,
                    "export_traces": self.settings.observability.export_traces,
                    "trace_log": str(self.settings.observability.trace_log_path),
                },
            },
        }

    def analytics(self) -> dict[str, Any]:
        """Aggregate run metrics from the data layer."""
        return analytics_summary(self.settings.data.sqlite_path)

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.recent_runs(limit=limit)

    def metrics_snapshot(self) -> dict[str, Any]:
        """Metric values as plain data, for the health endpoint and the UI."""
        return self.metrics.snapshot()

    def prometheus_metrics(self) -> str:
        """The registry in Prometheus text exposition format."""
        return self.metrics.render_prometheus()

    def last_trace(self) -> list[dict[str, Any]]:
        """Return the spans recorded by the most recent run."""
        return get_tracer().summary()


def _parse_prices(table: str) -> dict[str, tuple[float, float]]:
    """Parse "model:prompt:completion" entries into a price table.

    A malformed entry is skipped with a warning rather than raising, so one typo
    in an environment variable cannot stop the application from starting.
    """
    prices = dict(DEFAULT_PRICES)
    for item in (part.strip() for part in table.split(",") if part.strip()):
        pieces = item.split(":")
        if len(pieces) != 3:
            logger.warning("ignoring malformed price entry %r", item)
            continue
        model, prompt_price, completion_price = pieces
        try:
            prices[model.strip()] = (float(prompt_price), float(completion_price))
        except ValueError:
            logger.warning("ignoring price entry with non numeric values %r", item)
    return prices


def _build_principals(settings: SecuritySettings) -> PrincipalStore:
    """Load principals from a file if given, otherwise from the inline table."""
    if settings.principals_file:
        return PrincipalStore.from_file(settings.principals_file)
    return PrincipalStore.from_json(settings.principals)
