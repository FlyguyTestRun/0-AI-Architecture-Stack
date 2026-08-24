"""FastAPI service (the backend behind the frontend layer).

A Next.js or Streamlit frontend talks to this service. Everything below it is reached
through :class:`~zerostack.app.ZerostackApp`, so the API stays a thin translation of
HTTP into that one object.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from zerostack.app import ZerostackApp

logger = logging.getLogger(__name__)

_app_instance: ZerostackApp | None = None


def get_app() -> ZerostackApp:
    """Return the shared application instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = ZerostackApp()
    return _app_instance


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    persist: bool = True


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
def ask(request: AskRequest) -> AskResponse:
    """Run one question through the agent."""
    try:
        result = get_app().ask(request.question, persist=request.persist)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result.to_dict())


@api.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    """Index documents into the vector store."""
    try:
        return get_app().ingest(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.get("/runs")
def runs(limit: int = 20) -> dict[str, Any]:
    """Return recent agent runs from the data layer."""
    limit = max(1, min(limit, 200))
    return {"runs": get_app().recent_runs(limit=limit)}


@api.get("/analytics")
def analytics() -> dict[str, Any]:
    """Aggregate run metrics."""
    return get_app().analytics()


@api.get("/tools")
def tools() -> dict[str, Any]:
    """List the tools registered in the tool layer."""
    registry = get_app().tools
    return {"count": len(registry), "tools": registry.schemas()}
