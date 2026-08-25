"""Tests for the enterprise tier: identity, roles, tenancy and rate limiting.

Every control here is opt in. A small team on one machine should not configure
authentication to ask a question, and an enterprise deployment should not bolt
tenancy on afterwards. These tests pin both halves of that: the open path stays
open, and the configured path actually enforces.
"""

from __future__ import annotations

import json

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
    Principal,
    PrincipalStore,
    RateLimiter,
    RateLimitExceeded,
    Role,
    UnauthorizedError,
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
