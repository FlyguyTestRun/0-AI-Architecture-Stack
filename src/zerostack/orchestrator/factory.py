"""Orchestrator selection (the adapter seam)."""

from __future__ import annotations

import importlib.util
import logging

from zerostack.config import OrchestratorSettings, get_settings
from zerostack.orchestrator.base import AgentContext, Orchestrator
from zerostack.orchestrator.simple import SimpleOrchestrator

logger = logging.getLogger(__name__)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_engines() -> dict[str, bool]:
    """Report which engines can be constructed on this machine."""
    return {
        "simple": True,
        "langgraph": _installed("langgraph"),
        "crewai": _installed("crewai"),
    }


def build_orchestrator(
    context: AgentContext, settings: OrchestratorSettings | None = None
) -> Orchestrator:
    """Return an orchestrator according to configuration.

    ``auto`` prefers LangGraph when it is installed and falls back to the simple
    engine. CrewAI is never chosen automatically because it requires a reachable model
    server, so selecting it silently would turn an offline run into a failure.
    """
    settings = settings or get_settings().orchestrator
    kind = settings.kind

    if kind == "simple":
        return SimpleOrchestrator(context)

    if kind == "crewai":
        from zerostack.orchestrator.crewai_engine import CrewAIOrchestrator

        return CrewAIOrchestrator(context)

    if kind == "langgraph":
        from zerostack.orchestrator.langgraph_engine import LangGraphOrchestrator

        return LangGraphOrchestrator(context)

    # auto
    if _installed("langgraph"):
        try:
            from zerostack.orchestrator.langgraph_engine import LangGraphOrchestrator

            orchestrator = LangGraphOrchestrator(context)
            logger.info("orchestrator layer: using langgraph")
            return orchestrator
        except Exception as exc:
            logger.warning("langgraph present but failed to build (%s), using simple", exc)

    logger.info("orchestrator layer: using simple engine")
    return SimpleOrchestrator(context)
