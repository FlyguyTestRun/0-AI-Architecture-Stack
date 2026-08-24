"""The dependency free orchestrator.

It executes the node graph directly. This is the engine ``auto`` selects when neither
LangGraph nor CrewAI is installed, and it is what keeps a clean clone runnable.
"""

from __future__ import annotations

import time

from zerostack.observability import get_tracer
from zerostack.orchestrator.base import AgentContext, AgentResult
from zerostack.orchestrator.nodes import (
    generate_node,
    initial_state,
    plan_node,
    retrieve_node,
    tool_node,
)


class SimpleOrchestrator:
    """Runs plan, retrieve, tools and generate in sequence."""

    name = "simple"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def run(self, question: str) -> AgentResult:
        tracer = get_tracer()
        trace_id = tracer.new_trace()
        start = time.perf_counter()

        with tracer.span("orchestrator.run", engine=self.name, question=question):
            state = initial_state(question)
            state = plan_node(state, self.context)
            if state["needs_knowledge"]:
                state = retrieve_node(state, self.context)
            state = tool_node(state, self.context)
            state = generate_node(state, self.context)

        return AgentResult(
            answer=state["answer"],
            question=question,
            orchestrator=self.name,
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model"],
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            sources=state["sources"],
            steps=state["steps"],
            trace_id=trace_id,
        )
