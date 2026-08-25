"""CrewAI orchestrator.

CrewAI expresses the same work as a crew of role playing agents rather than a graph.
Retrieval still runs through this repository's RAG pipeline so that both engines share
one index and one citation format. Only the reasoning step is delegated to the crew.

Requires the ``crewai`` extra and a reachable model server. CrewAI cannot run against
the offline provider, so ``auto`` never selects this engine.
"""

from __future__ import annotations

import time
from typing import Any

from zerostack.observability import get_tracer
from zerostack.orchestrator.base import AgentContext, AgentResult, AgentStep
from zerostack.orchestrator.nodes import (
    SYSTEM_PROMPT,
    initial_state,
    plan_node,
    retrieve_node,
    tool_node,
)


class CrewAIOrchestrator:
    """Executes the reasoning step as a single agent CrewAI crew."""

    name = "crewai"

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        from crewai import LLM

        llm_client = self.context.llm
        base_url = getattr(llm_client, "base_url", "http://localhost:11434")
        return LLM(model=f"ollama/{llm_client.model}", base_url=base_url)

    def _build_crew(self, question: str, context_block: str) -> Any:
        from crewai import Agent, Crew, Process, Task

        analyst = Agent(
            role="Business Knowledge Analyst",
            goal="Answer the question using only the supplied context, with citations.",
            backstory=(
                "You support a small business team. You are precise, you never invent "
                "facts, and you always cite the bracketed source number for each claim."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        task = Task(
            description=(f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {question}"),
            expected_output="A concise, cited answer grounded in the supplied context.",
            agent=analyst,
        )

        return Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)

    def run(self, question: str, namespace: str = "default") -> AgentResult:
        tracer = get_tracer()
        trace_id = tracer.new_trace()
        start = time.perf_counter()

        with tracer.span("orchestrator.run", engine=self.name, question=question):
            state = initial_state(question, namespace)
            state = plan_node(state, self.context)
            if state["needs_knowledge"]:
                state = retrieve_node(state, self.context)
            state = tool_node(state, self.context)

            context_block = state.get("context") or "No context retrieved."
            if state.get("tool_output"):
                context_block += f"\n\nTool result: {state['tool_output']}"

            with tracer.span("crewai.kickoff"):
                crew_output = self._build_crew(question, context_block).kickoff()

            answer = str(crew_output).strip()
            state["steps"].append(
                AgentStep(
                    name="generate",
                    detail=f"crewai/{self.context.llm.model}",
                    data={"engine": "crewai"},
                )
            )

        return AgentResult(
            answer=answer,
            question=question,
            orchestrator=self.name,
            llm_provider="crewai",
            llm_model=self.context.llm.model,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            sources=state["sources"],
            steps=state["steps"],
            trace_id=trace_id,
        )
