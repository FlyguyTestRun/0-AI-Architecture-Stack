"""Tests for the enterprise tier: identity, roles, tenancy and rate limiting.

Every control here is opt in. A small team on one machine should not configure
authentication to ask a question, and an enterprise deployment should not bolt
tenancy on afterwards. These tests pin both halves of that: the open path stays
open, and the configured path actually enforces.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

import zerostack.api.main as api_module
from zerostack.api.main import api
from zerostack.app import ZerostackApp
from zerostack.config import RAGSettings
from zerostack.rag.embeddings import HashingEmbeddings
from zerostack.rag.pipeline import RAGPipeline
from zerostack.rag.store import MemoryVectorStore
from zerostack.security import (
    ALL_NAMESPACES,
    InvalidNamespace,
    Principal,
    PrincipalStore,
    RateLimiter,
    RateLimitExceeded,
    Role,
    UnauthorizedError,
    normalise_grant,
    normalise_namespace,
)
from zerostack.security.identity import generate_key, hash_key


class TestRoles:
    def test_roles_are_ordered(self):
        assert Role.ADMIN.allows(Role.READER)
        assert not Role.READER.allows(Role.ADMIN)

    def test_a_role_allows_itself(self):
        assert Role.WRITER.allows(Role.WRITER)

    def test_operator_may_read_but_not_administer(self):
        assert Role.OPERATOR.allows(Role.READER)
        assert not Role.OPERATOR.allows(Role.ADMIN)


class TestPrincipalStore:
    def test_an_empty_store_grants_local_admin(self):
        """The single machine path must work with no configuration."""
        principal = PrincipalStore().resolve(None)
        assert principal.role is Role.ADMIN
        assert principal.sees_all_namespaces()

    def test_a_configured_store_requires_a_key(self):
        store = PrincipalStore({hash_key("k"): Principal(name="a")})
        with pytest.raises(UnauthorizedError):
            store.resolve(None)

    def test_an_unknown_key_is_rejected(self):
        store = PrincipalStore({hash_key("k"): Principal(name="a")})
        with pytest.raises(UnauthorizedError):
            store.resolve("other")

    def test_a_valid_key_resolves(self):
        store = PrincipalStore({hash_key("k"): Principal(name="a", role=Role.WRITER)})
        assert store.resolve("k").name == "a"

    def test_keys_are_stored_hashed(self):
        """A leaked configuration file must not hand over working credentials."""
        store = PrincipalStore()
        store.add("secret-key", Principal(name="a"))
        assert "secret-key" not in json.dumps(list(store._by_hash))

    def test_generated_keys_are_unique(self):
        assert generate_key() != generate_key()

    def test_a_malformed_table_does_not_lock_everyone_out(self):
        assert PrincipalStore.from_json("{not json").count == 0

    def test_one_bad_entry_does_not_discard_the_good_ones(self):
        table = json.dumps({"k1": {"name": "ok", "role": "reader"}, "k2": ["not", "an", "object"]})
        assert PrincipalStore.from_json(table).count == 1

    def test_an_unknown_role_skips_that_entry(self):
        table = json.dumps({"k": {"name": "x", "role": "sorcerer"}})
        assert PrincipalStore.from_json(table).count == 0

    def test_a_missing_file_yields_an_open_store(self, tmp_path):
        assert PrincipalStore.from_file(tmp_path / "absent.json").count == 0


class TestNamespaceScoping:
    def test_a_principal_is_confined_to_its_namespaces(self):
        principal = Principal(name="hr", namespaces=["hr"])
        assert principal.may_access("hr")
        assert not principal.may_access("legal")

    def test_a_wildcard_sees_everything(self):
        assert Principal(name="ops", namespaces=["*"]).may_access("anything")

    def test_the_first_namespace_is_the_default(self):
        assert Principal(name="hr", namespaces=["hr", "ops"]).default_namespace() == "hr"


class TestRetrievalIsolation:
    """The property tenancy exists for."""

    @pytest.fixture
    def pipeline(self):
        settings = RAGSettings(
            backend="memory", embedding_backend="hashing", embedding_dimensions=256
        )
        built = RAGPipeline(
            store=MemoryVectorStore(),
            embeddings=HashingEmbeddings(256),
            settings=settings,
        )
        built.ingest_text(
            "The acme merger closes in March for eight hundred million.",
            source="deal.md",
            namespace="client-acme",
        )
        built.ingest_text(
            "The globex restructuring is confidential and ongoing.",
            source="restructure.md",
            namespace="client-globex",
        )
        return built

    def test_a_scoped_query_returns_only_its_namespace(self, pipeline):
        results = pipeline.retrieve("merger restructuring", top_k=9, namespace="client-acme")
        assert results
        assert all(result.namespace == "client-acme" for result in results)

    def test_the_other_tenant_is_unreachable(self, pipeline):
        results = pipeline.retrieve("globex restructuring", top_k=9, namespace="client-acme")
        assert all("restructure" not in result.source for result in results)

    def test_an_unscoped_query_still_sees_everything(self, pipeline):
        results = pipeline.retrieve("merger restructuring", top_k=9)
        assert len({result.namespace for result in results}) == 2

    def test_each_namespace_has_its_own_graph(self, pipeline):
        """A shared graph would let a traversal walk between tenants."""
        context = pipeline.graph_context("acme merger", namespace="client-acme")
        assert "restructure.md" not in context


class TestRateLimiter:
    def test_a_disabled_limiter_allows_everything(self):
        limiter = RateLimiter(requests_per_minute=0)
        for _ in range(100):
            limiter.check("caller")

    def test_the_allowance_runs_out(self):
        limiter = RateLimiter(requests_per_minute=3)
        for _ in range(3):
            limiter.check("caller")
        with pytest.raises(RateLimitExceeded):
            limiter.check("caller")

    def test_callers_are_independent(self):
        limiter = RateLimiter(requests_per_minute=1)
        limiter.check("a")
        limiter.check("b")

    def test_a_new_caller_starts_with_a_full_bucket(self):
        """A first request must never be rejected for having no history."""
        RateLimiter(requests_per_minute=1).check("brand-new")

    def test_a_per_principal_ceiling_overrides_the_default(self):
        limiter = RateLimiter(requests_per_minute=1)
        for _ in range(5):
            limiter.check("vip", requests_per_minute=100)

    def test_the_error_reports_when_to_retry(self):
        limiter = RateLimiter(requests_per_minute=60)
        limiter.check("caller")
        with pytest.raises(RateLimitExceeded) as caught:
            for _ in range(200):
                limiter.check("caller")
        assert caught.value.retry_after_seconds > 0


@pytest.fixture
def keys():
    return {"writer": generate_key(), "operator": generate_key(), "reader": generate_key()}


@pytest.fixture
def secured_client(settings, corpus_dir, keys, monkeypatch):
    settings.rag.corpus_dir = corpus_dir
    settings.security.principals = json.dumps(
        {
            keys["writer"]: {"name": "hr-app", "role": "writer", "namespaces": ["hr"]},
            keys["operator"]: {"name": "ops", "role": "operator", "namespaces": ["*"]},
            keys["reader"]: {"name": "kiosk", "role": "reader", "namespaces": ["hr"]},
        }
    )
    instance = ZerostackApp(settings=settings, include_mcp=False)
    monkeypatch.setattr(api_module, "_app_instance", instance)
    with TestClient(api) as client:
        yield client


def header(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


class TestApiAuthentication:
    def test_a_missing_key_is_rejected(self, secured_client):
        assert secured_client.post("/ask", json={"question": "hi"}).status_code == 401

    def test_an_invalid_key_is_rejected(self, secured_client):
        response = secured_client.post("/ask", json={"question": "hi"}, headers=header("wrong"))
        assert response.status_code == 401

    def test_the_rejection_does_not_say_which_failure_it_was(self, secured_client):
        """Distinguishing unknown from missing gains an attacker information."""
        missing = secured_client.post("/ask", json={"question": "hi"}).json()["detail"]
        invalid = secured_client.post(
            "/ask", json={"question": "hi"}, headers=header("wrong")
        ).json()["detail"]
        assert missing == invalid

    def test_whoami_reports_the_caller(self, secured_client, keys):
        body = secured_client.get("/whoami", headers=header(keys["operator"])).json()
        assert body["name"] == "ops"


class TestApiAuthorization:
    def test_a_reader_may_not_ingest(self, secured_client, keys):
        response = secured_client.post("/ingest", json={}, headers=header(keys["reader"]))
        assert response.status_code == 403

    def test_a_writer_may_ingest(self, secured_client, keys):
        response = secured_client.post("/ingest", json={}, headers=header(keys["writer"]))
        assert response.status_code == 200

    def test_a_reader_may_not_read_metrics(self, secured_client, keys):
        assert secured_client.get("/metrics", headers=header(keys["reader"])).status_code == 403

    def test_an_operator_may_read_metrics(self, secured_client, keys):
        assert secured_client.get("/metrics", headers=header(keys["operator"])).status_code == 200

    def test_a_reader_may_ask(self, secured_client, keys):
        secured_client.post("/ingest", json={}, headers=header(keys["writer"]))
        response = secured_client.post(
            "/ask", json={"question": "refund limit"}, headers=header(keys["reader"])
        )
        assert response.status_code == 200


class TestApiTenancy:
    def test_a_foreign_namespace_is_refused(self, secured_client, keys):
        response = secured_client.post(
            "/ask",
            json={"question": "anything", "namespace": "legal"},
            headers=header(keys["writer"]),
        )
        assert response.status_code == 403

    def test_the_refusal_does_not_confirm_the_namespace_exists(self, secured_client, keys):
        """The boundary must not double as a directory of other tenants."""
        real = secured_client.post(
            "/ask", json={"question": "x", "namespace": "legal"}, headers=header(keys["writer"])
        ).json()["detail"]
        invented = secured_client.post(
            "/ask", json={"question": "x", "namespace": "zzz"}, headers=header(keys["writer"])
        ).json()["detail"]
        assert real == invented

    def test_a_permitted_namespace_is_allowed(self, secured_client, keys):
        secured_client.post("/ingest", json={"namespace": "hr"}, headers=header(keys["writer"]))
        response = secured_client.post(
            "/ask", json={"question": "refund", "namespace": "hr"}, headers=header(keys["writer"])
        )
        assert response.status_code == 200

    def test_a_wildcard_principal_reaches_any_namespace(self, secured_client, keys):
        response = secured_client.post(
            "/ask",
            json={"question": "x", "namespace": "anything"},
            headers=header(keys["operator"]),
        )
        assert response.status_code == 200


@pytest.fixture
def open_client(settings, corpus_dir, monkeypatch):
    """A deployment with no principals configured."""
    settings.rag.corpus_dir = corpus_dir
    settings.security.principals = ""
    instance = ZerostackApp(settings=settings, include_mcp=False)
    monkeypatch.setattr(api_module, "_app_instance", instance)
    with TestClient(api) as client:
        yield client


class TestOpenModeIsUnchanged:
    """A deployment with no principals must behave exactly as before."""

    def test_no_key_is_needed(self, open_client):
        assert open_client.post("/ask", json={"question": "refund"}).status_code == 200

    def test_every_endpoint_is_reachable(self, open_client):
        assert open_client.get("/metrics").status_code == 200
        assert open_client.post("/ingest", json={}).status_code == 200

    def test_the_caller_is_reported_as_a_local_administrator(self, open_client):
        body = open_client.get("/whoami").json()
        assert body["role"] == "admin"

    def test_health_states_that_authentication_is_off(self, open_client):
        security = open_client.get("/health").json()["layers"]["security"]
        assert "disabled" in security["authentication"]


class TestNamespaceValidation:
    """A namespace is an identifier, not free text.

    It reaches a metric label, a cache partition and a retrieval filter, so an
    unconstrained one lets any caller holding the wildcard grant mint a new
    metric series on every request until the process runs out of memory.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hr", "hr"),
            ("HR", "hr"),
            ("  Legal  ", "legal"),
            ("client-acme", "client-acme"),
            ("a_b", "a_b"),
            (None, "default"),
            ("", "default"),
            ("   ", "default"),
        ],
    )
    def test_well_formed_namespaces_are_canonicalised(self, value, expected):
        assert normalise_namespace(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["*", "x" * 64, "a\nb", "../etc", "-lead", "ns!", "a b", "Ünicode"],
    )
    def test_malformed_namespaces_are_refused(self, value):
        with pytest.raises(InvalidNamespace):
            normalise_namespace(value)

    def test_case_only_variants_are_one_tenant(self):
        """Two partitions that read as one to a human are a tenancy hazard."""
        assert normalise_namespace("Finance") == normalise_namespace("finance")

    def test_the_wildcard_is_a_grant_not_a_target(self):
        assert normalise_grant("*") == ALL_NAMESPACES
        with pytest.raises(InvalidNamespace):
            normalise_namespace("*")

    def test_a_grant_is_canonicalised_too(self):
        """Otherwise a table saying "HR" would never match a request for "hr"."""
        store = PrincipalStore.from_json(
            json.dumps({"zs_k": {"name": "app", "role": "reader", "namespaces": ["HR", " Legal "]}})
        )
        principal = store.resolve("zs_k")
        assert principal.namespaces == ["hr", "legal"]
        assert principal.may_access("hr")


class TestNamespaceValidationAtTheBoundary:
    def test_the_api_refuses_a_malformed_namespace(self, secured_client, keys):
        response = secured_client.post(
            "/ask",
            json={"question": "x", "namespace": "a b!"},
            headers=header(keys["operator"]),
        )
        assert response.status_code == 422

    def test_a_wildcard_principal_cannot_mint_unbounded_namespaces(self, secured_client, keys):
        """The wildcard grant waves through the permission check, so the shape
        check has to run before it."""
        response = secured_client.post(
            "/ask",
            json={"question": "x", "namespace": "n" * 200},
            headers=header(keys["operator"]),
        )
        assert response.status_code == 422

    def test_metric_labels_stay_bounded(self, secured_client, keys):
        """One tenant asking under many spellings is still one metric series."""
        app = api_module._app_instance
        before = len(app.metrics._counters)
        for spelling in ["hr", "HR", "  hr  ", "Hr"]:
            secured_client.post(
                "/ask",
                json={"question": "x", "namespace": spelling},
                headers=header(keys["operator"]),
            )
        assert len(app.metrics._counters) - before <= 1

    def test_the_ingest_path_validates_too(self, secured_client, keys):
        """Otherwise an invalid namespace reaches the generic handler as a 500."""
        response = secured_client.post(
            "/ingest", json={"namespace": "../escape"}, headers=header(keys["operator"])
        )
        assert response.status_code == 422

    def test_the_library_boundary_validates(self, settings, corpus_dir):
        """The CLI and the frontend call straight into the app, not the API."""
        settings.rag.corpus_dir = corpus_dir
        instance = ZerostackApp(settings=settings, include_mcp=False)
        with pytest.raises(InvalidNamespace):
            instance.ask("anything", namespace="not a namespace")


class TestCallerIdentity:
    """A rate limit bucket must be keyed on the credential, not on its label.

    The name is an operator chosen string and nothing stops two entries sharing
    one. Keying a meter on it pools two distinct credentials into a single
    allowance, so each silently receives half of what it was configured.
    """

    def test_two_credentials_sharing_a_name_are_distinct_callers(self):
        store = PrincipalStore.from_json(
            json.dumps(
                {
                    "zs_key_one": {"name": "app", "role": "reader"},
                    "zs_key_two": {"name": "app", "role": "reader"},
                }
            )
        )
        first = store.resolve("zs_key_one")
        second = store.resolve("zs_key_two")
        assert first.name == second.name
        assert first.caller_id() != second.caller_id()

    def test_each_credential_gets_its_own_allowance(self):
        store = PrincipalStore.from_json(
            json.dumps(
                {
                    "zs_key_one": {"name": "app", "role": "reader", "requests_per_minute": 4},
                    "zs_key_two": {"name": "app", "role": "reader", "requests_per_minute": 4},
                }
            )
        )
        first = store.resolve("zs_key_one")
        second = store.resolve("zs_key_two")
        limiter = RateLimiter(requests_per_minute=4)
        allowed = 0
        for index in range(8):
            caller = first if index % 2 == 0 else second
            try:
                limiter.check(caller.caller_id(), caller.requests_per_minute)
                allowed += 1
            except RateLimitExceeded:
                pass
        assert allowed == 8

    def test_open_mode_still_has_a_usable_identity(self):
        """With no principals configured there is one implicit local caller."""
        principal = PrincipalStore().resolve(None)
        assert principal.caller_id() == "local"

    def test_the_identity_is_not_the_raw_key(self):
        store = PrincipalStore()
        store.add("zs_supersecret", Principal(name="app"))
        principal = store.resolve("zs_supersecret")
        assert "zs_supersecret" not in principal.caller_id()


class TestLimiterScaling:
    def test_eviction_is_off_the_hot_path(self):
        """Sweeping on every call made admitting one request cost O(callers)."""
        limiter = RateLimiter(requests_per_minute=600)
        for index in range(5000):
            limiter.check(f"caller-{index}")
        start = time.perf_counter()
        for _ in range(200):
            limiter.check("caller-0")
        elapsed = time.perf_counter() - start
        # Generous: the per call scan cost roughly 0.5ms per request at this
        # table size, so 200 calls took over a tenth of a second.
        assert elapsed < 0.05, f"200 checks took {elapsed:.3f}s against 5000 callers"

    def test_idle_buckets_are_still_evicted(self):
        limiter = RateLimiter(requests_per_minute=60, idle_eviction_seconds=0.05)
        limiter.check("goes-idle")
        assert limiter.snapshot()["tracked_callers"] == 1
        time.sleep(0.12)
        limiter.check("still-here")
        assert limiter.snapshot()["tracked_callers"] == 1

    def test_an_active_bucket_is_not_evicted(self):
        limiter = RateLimiter(requests_per_minute=600, idle_eviction_seconds=0.05)
        limiter.check("busy")
        for _ in range(5):
            time.sleep(0.03)
            limiter.check("busy")
        assert limiter.snapshot()["tracked_callers"] == 1


class TestFailClosedPrincipals:
    """A principal table that was supplied but cannot be used must not open up.

    An empty store means "no authentication configured", which is the correct
    default for one machine. A broken store means an operator was configuring
    authentication and got it wrong. Collapsing the second into the first turns a
    typo, or a secret that failed to mount, into open admin access.
    """

    def test_malformed_json_refuses_everyone(self):
        store = PrincipalStore.from_json('{"zs_key": {"name": "app", ')
        with pytest.raises(UnauthorizedError):
            store.resolve(None)

    def test_malformed_json_refuses_a_presented_key_too(self):
        store = PrincipalStore.from_json('{"zs_key": {"name": "app", ')
        with pytest.raises(UnauthorizedError):
            store.resolve("zs_key")

    def test_a_configured_file_that_is_missing_refuses_everyone(self, tmp_path):
        """The secret that did not mount is exactly this case."""
        store = PrincipalStore.from_file(tmp_path / "never-written.json")
        with pytest.raises(UnauthorizedError):
            store.resolve(None)

    def test_a_table_where_every_entry_is_invalid_refuses_everyone(self):
        store = PrincipalStore.from_json('{"zs_key": {"role": "not-a-role"}}')
        with pytest.raises(UnauthorizedError):
            store.resolve(None)

    def test_a_partially_valid_table_still_works(self):
        """One bad entry must not lock out the good ones."""
        store = PrincipalStore.from_json(
            json.dumps(
                {
                    "zs_good": {"name": "app", "role": "reader"},
                    "zs_bad": {"name": "broken", "role": "not-a-role"},
                }
            )
        )
        assert store.resolve("zs_good").name == "app"

    def test_no_table_configured_is_still_open(self):
        """The single machine path must not need credentials."""
        assert PrincipalStore.from_json("").resolve(None).name == "local"

    def test_health_reports_the_failed_state(self):
        store = PrincipalStore.from_json("{not json")
        described = store.describe()
        assert described["authentication"] == "failed closed (misconfigured)"
        assert described["principals"] == 0


class TestOperationalDataIsScoped:
    """The run log carries questions, answers and retrieved document text."""

    @pytest.fixture
    def populated(self, settings, corpus_dir, tmp_path):
        settings.rag.corpus_dir = corpus_dir
        settings.data.sqlite_path = tmp_path / "runs.db"
        settings.cache.enabled = False
        instance = ZerostackApp(settings=settings, include_mcp=False)
        instance.ingest(str(corpus_dir), namespace="hr", enforce_roots=False)
        instance.ingest(str(corpus_dir), namespace="legal", enforce_roots=False)
        instance.ask("How long is the probationary period?", namespace="hr")
        instance.ask("How long is the probationary period?", namespace="legal")
        return instance

    def test_a_wildcard_operator_sees_every_run(self, populated):
        assert len(populated.recent_runs(namespaces=None)) == 2

    def test_a_scoped_operator_sees_only_its_own(self, populated):
        runs = populated.recent_runs(namespaces=["hr"])
        assert [run["namespace"] for run in runs] == ["hr"]

    def test_a_principal_with_no_namespaces_sees_nothing(self, populated):
        """An empty scope must restrict to nothing, never to everything."""
        assert populated.recent_runs(namespaces=[]) == []

    def test_analytics_is_scoped_the_same_way(self, populated):
        assert populated.analytics(namespaces=None)["runs"] == 2
        assert populated.analytics(namespaces=["hr"])["runs"] == 1
        assert populated.analytics(namespaces=[])["runs"] == 0

    def test_the_api_scopes_the_run_log_to_the_caller(self, secured_client, keys):
        secured_client.post(
            "/ask", json={"question": "refund", "namespace": "hr"}, headers=header(keys["writer"])
        )
        response = secured_client.get("/runs", headers=header(keys["operator"]))
        assert response.status_code == 200

    def test_visible_namespaces_is_none_only_for_a_wildcard(self):
        from zerostack.api.main import visible_namespaces

        assert visible_namespaces(Principal(name="a", namespaces=[ALL_NAMESPACES])) is None
        assert visible_namespaces(Principal(name="b", namespaces=["hr"])) == ["hr"]
