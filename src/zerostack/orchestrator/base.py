"""Types shared by every orchestrator implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from zerostack.config import OrchestratorSettings, get_settings
from zerostack.llm.base import LLMClient
from zerostack.rag.pipeline import RAGPipeline
from zerostack.tools.registry import ToolRegistry


@dataclass
class AgentStep:
    """One recorded step in an agent run. Used for the UI trace panel."""

    name: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "detail": self.detail, "data": self.data}


@dataclass
class AgentResult:
    """The outcome of a full agent run."""

    answer: str
    question: str
    orchestrator: str
    llm_provider: str
    llm_model: str
    latency_ms: float = 0.0
    sources: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "question": self.question,
            "orchestrator": self.orchestrator,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "latency_ms": self.latency_ms,
            "sources": self.sources,
            "steps": [step.to_dict() for step in self.steps],
            "trace_id": self.trace_id,
        }


@dataclass
class AgentContext:
    """Everything an orchestrator needs, injected rather than constructed.

    Passing the context in is what lets the three orchestrator implementations share
    one set of node functions and lets tests substitute fakes for any layer.
    """

    llm: LLMClient
    rag: RAGPipeline
    tools: ToolRegistry
    settings: OrchestratorSettings = field(default_factory=lambda: get_settings().orchestrator)
    # Set per request by the application. Nodes read it so retrieval and graph
    # traversal stay inside the caller's tenancy boundary.
    namespace: str = "default"


@runtime_checkable
class Orchestrator(Protocol):
    """Layer 2. The only method the frontend and API depend on."""

    name: str

    def run(self, question: str) -> AgentResult: ...
