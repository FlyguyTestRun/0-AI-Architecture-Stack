"""LangGraph orchestrator.

The same node functions are wired as a StateGraph with an explicit conditional edge
for the "need external knowledge" decision from the architecture diagram. Using
LangGraph buys durable state, checkpointing and streaming once a deployment needs
them, without changing what the nodes do.

Requires the ``langgraph`` extra.
"""

from __future__ import annotations

import time
from typing import Any

from zerostack.observability import get_tracer
from zerostack.orchestrator.base import AgentContext, AgentResult
from zerostack.orchestrator.nodes import (
    generate_node,
    initial_state,
    plan_node,
    retrieve_node,
    tool_node,
)


class LangGraphOrchestrator:
    """Executes the agent graph on LangGraph."""

    name = "langgraph"

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(dict)

        builder.add_node("plan", lambda state: plan_node(state, self.context))
        builder.add_node("retrieve", lambda state: retrieve_node(state, self.context))
        builder.add_node("tools", lambda state: tool_node(state, self.context))
        builder.add_node("generate", lambda state: generate_node(state, self.context))

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            lambda state: "retrieve" if state.get("needs_knowledge") else "tools",
            {"retrieve": "retrieve", "tools": "tools"},
        )
        builder.add_edge("retrieve", "tools")
        builder.add_edge("tools", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    def run(self, question: str) -> AgentResult:
        tracer = get_tracer()
        trace_id = tracer.new_trace()
        start = time.perf_counter()

        with tracer.span("orchestrator.run", engine=self.name, question=question):
            state = self._graph.invoke(initial_state(question))

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
