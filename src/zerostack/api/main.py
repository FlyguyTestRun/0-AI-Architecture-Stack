"""FastAPI service (the backend behind the frontend layer).

A Next.js or Streamlit frontend talks to this service. Everything below it is reached
through :class:`~zerostack.app.ZerostackApp`, so the API stays a thin translation of
HTTP into that one object.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from zerostack.app import ZerostackApp
from zerostack.observability import BudgetExceeded
from zerostack.security import (
    Principal,
    RateLimitExceeded,
    Role,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

_app_instance: ZerostackApp | None = None


def get_app() -> ZerostackApp:
    """Return the shared application instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = ZerostackApp()
    return _app_instance


def enforce_rate_limit(principal: Principal) -> None:
    try:
        get_app().rate_limiter.check(principal.name, principal.requests_per_minute)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(max(1, int(exc.retry_after_seconds)))},
        ) from exc


def authenticate(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """Resolve the caller.

    With no principals configured this returns a local administrator, so a
    single machine deployment needs no credentials. The failure is a 401 with no
    detail about why the key was rejected, since distinguishing "unknown key"
    from "missing key" to an attacker gains them information and gains a
    legitimate caller nothing.
    """
    try:
        principal = get_app().principals.resolve(x_api_key)
    except UnauthorizedError as exc:
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "ApiKey"},
        ) from exc

    # Limiting inside the dependency covers every endpoint that authenticates.
    # Applying it only to the expensive ones left the observability endpoints
    # unbounded, and a caller hammering those is still a denial of service.
    enforce_rate_limit(principal)
    return principal


def require(principal: Principal, role: Role) -> None:
    if not principal.may(role):
        raise HTTPException(
            status_code=403,
            detail=f"this operation requires the {role.value} role",
        )


def resolve_namespace(principal: Principal, requested: str | None) -> str:
    """Pick the namespace for a request and confirm the caller may use it."""
    namespace = requested or principal.default_namespace()
    if not principal.may_access(namespace):
        # Refusing without confirming whether the namespace exists keeps the
        # boundary from doubling as a directory of other tenants.
        raise HTTPException(status_code=403, detail="namespace not permitted")
    return namespace


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    persist: bool = True
    namespace: str | None = None
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    question: str
    orchestrator: str
    llm_provider: str
    llm_model: str
    latency_ms: float
    sources: list[str]
    steps: list[dict[str, Any]]
    trace_id: str


class IngestRequest(BaseModel):
    path: str | None = Field(
        default=None, description="File or directory to ingest. Defaults to the corpus."
    )
    namespace: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Building the stack at startup surfaces configuration problems immediately
    # rather than on the first user request.
    instance = get_app()
    logger.info("zerostack api ready, orchestrator=%s", instance.orchestrator.name)
    yield


api = FastAPI(
    title="Zerostack API",
    version="0.1.0",
    description="A zero cost AI architecture baseline.",
    lifespan=lifespan,
)


@api.get("/health")
def health() -> dict[str, Any]:
    """Report which backend is live in every layer."""
    return get_app().health()


@api.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, principal: Principal = Depends(authenticate)) -> AskResponse:
    """Run one question through the agent."""
    require(principal, Role.READER)
    namespace = resolve_namespace(principal, request.namespace)
    try:
        result = get_app().ask(
            request.question,
            persist=request.persist,
            use_cache=request.use_cache,
            namespace=namespace,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BudgetExceeded as exc:
        # The ceiling was reached. That is a throttle, not a server fault, and a
        # caller can retry once the window rolls over.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result.to_dict())


@api.post("/ingest")
def ingest(request: IngestRequest, principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """Index documents into the vector store."""
    require(principal, Role.WRITER)
    namespace = resolve_namespace(principal, request.namespace)
    try:
        return get_app().ingest(request.path, namespace=namespace)
    except PermissionError as exc:
        # The path is outside the configured roots. Refusing is the whole point,
        # so this is a 403 rather than a server error.
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.get("/runs")
def runs(limit: int = 20, principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """Return recent agent runs from the data layer."""
    require(principal, Role.OPERATOR)
    limit = max(1, min(limit, 200))
    return {"runs": get_app().recent_runs(limit=limit)}


@api.get("/analytics")
def analytics(principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """Aggregate run metrics."""
    require(principal, Role.OPERATOR)
    return get_app().analytics()


@api.get("/metrics", response_class=Response)
def metrics(principal: Principal = Depends(authenticate)) -> Response:
    """Metrics in the Prometheus text exposition format.

    Served as plain text with the exposition content type so a standard scraper
    can read it without any adapter.
    """
    require(principal, Role.OPERATOR)
    return Response(
        content=get_app().prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@api.get("/metrics.json")
def metrics_json(principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """The same metrics as JSON, for dashboards and the browser frontend."""
    require(principal, Role.OPERATOR)
    return get_app().metrics_snapshot()


@api.get("/costs")
def costs(principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """Token volume, estimated spend and remaining budget."""
    require(principal, Role.OPERATOR)
    return get_app().costs.snapshot()


@api.get("/whoami")
def whoami(principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """Report the calling identity, for debugging a credential."""
    return principal.to_dict()


@api.get("/tools")
def tools(principal: Principal = Depends(authenticate)) -> dict[str, Any]:
    """List the tools registered in the tool layer."""
    require(principal, Role.READER)
    registry = get_app().tools
    return {"count": len(registry), "tools": registry.schemas()}
