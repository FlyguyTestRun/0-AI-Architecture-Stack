"""The agent graph, expressed as pure functions over a state dictionary.

Every orchestrator implementation runs these same nodes. LangGraph wires them as a
StateGraph, CrewAI wraps them as tasks, and the simple engine calls them in order.
Keeping the logic here rather than in the engines is the whole point of the adapter:
switching orchestrators changes the execution strategy, never the behaviour.

The node sequence follows the decision path in the architecture diagram:

    plan -> (needs external knowledge?) -> retrieve -> tools -> generate
                                       \\-> tools -> generate
"""

from __future__ import annotations

import logging
import re
from typing import Any

from zerostack.llm.base import ChatMessage
from zerostack.llm.offline import CONTEXT_PREFIX, TOOL_PREFIX
from zerostack.observability import get_tracer
from zerostack.orchestrator.base import AgentContext, AgentStep

logger = logging.getLogger(__name__)

State = dict[str, Any]

SYSTEM_PROMPT = """You are a retrieval grounded assistant for a small business team.

Rules:
1. Answer only from the context provided below. Do not invent facts.
2. Cite the bracketed source number after each claim, for example [1].
3. If the context does not contain the answer, say so plainly.
4. Be direct and concise. No preamble."""

# A question needing no document lookup is either arithmetic or asks about the
# conversation itself. Anything else is assumed to need the knowledge base, because
# a needless retrieval is far cheaper than a confidently wrong ungrounded answer.
_ARITHMETIC_RE = re.compile(r"^[\s\d+\-*/().%]+$")
_TOOL_HINT_RE = re.compile(
    r"\b(calculate|compute|what is|how much is|sum of|product of|current time|what time|today's date)\b",
    re.IGNORECASE,
)
# Matches an inline arithmetic expression including parentheses, so "(120 * 3) / 4"
# is captured whole rather than truncated at the closing bracket.
_EXPRESSION_RE = re.compile(r"[\d(][\d\s.()+\-*/%]*[\d)]")
# Words that carry no retrieval intent, used when deciding whether a question is
# arithmetic wrapped in words rather than a genuine knowledge question.
_ROUTING_STOPWORDS = {
    "a",
    "an",
    "and",
    "equals",
    "is",
    "of",
    "please",
    "result",
    "the",
    "to",
    "value",
    "what",
    "whats",
    "me",
    "give",
    "tell",
    "answer",
    "plus",
    "minus",
    "times",
    "divided",
    "by",
}


def plan_node(state: State, context: AgentContext) -> State:
    """Decide whether the question needs the knowledge base and which tools apply."""
    question = state["question"]
    stripped = question.strip()

    is_bare_expression = bool(_ARITHMETIC_RE.match(stripped)) and any(
        ch.isdigit() for ch in stripped
    )

    # A question can be arithmetic even when it is wrapped in words ("what is 15 * 4?").
    # Strip the expression and the tool phrasing: if nothing meaningful is left, the
    # question is arithmetic and retrieval would only add noise.
    expression_match = _EXPRESSION_RE.search(stripped)
    has_expression = bool(expression_match) and any(
        op in expression_match.group(0) for op in "+-*/%"
    )
    remainder = _EXPRESSION_RE.sub(" ", stripped)
    remainder = _TOOL_HINT_RE.sub(" ", remainder)
    remainder_terms = [
        token
        for token in re.findall(r"[a-z]+", remainder.lower())
        if token not in _ROUTING_STOPWORDS
    ]

    is_pure_math = is_bare_expression or (has_expression and not remainder_terms)

    # The planner is the single source of truth for the expression to evaluate. The
    # tool node must not re-derive it, or the two can disagree and evaluate a bare
    # number that carries no operator.
    if is_bare_expression:
        state["expression"] = stripped
    elif has_expression:
        state["expression"] = expression_match.group(0).strip()
    else:
        state["expression"] = ""

    wants_tool = bool(_TOOL_HINT_RE.search(question)) or is_pure_math

    needs_knowledge = not is_pure_math

    state["needs_knowledge"] = needs_knowledge
    state["is_pure_math"] = is_pure_math
    state["wants_tool"] = wants_tool and context.settings.enable_tools
    state["steps"].append(
        AgentStep(
            name="plan",
            detail=(
                "route: retrieve then generate"
                if needs_knowledge
                else "route: skip retrieval, answer directly"
            ),
            data={"needs_knowledge": needs_knowledge, "wants_tool": state["wants_tool"]},
        )
    )
    return state


def retrieve_node(state: State, context: AgentContext) -> State:
    """Fetch supporting context from the RAG pipeline."""
    if not state.get("needs_knowledge"):
        state["context"] = ""
        state["sources"] = []
        return state

    results = context.rag.retrieve(state["question"], namespace=context.namespace)
    state["results"] = results
    passage_context = context.rag.format_context(results)

    # The graph tier contributes only when it recognises an entity in the
    # question, so it adds nothing to the prompt for questions it cannot help
    # with. It is appended rather than merged so a reader can tell which part of
    # the context came from passages and which from relations between documents.
    graph_context = context.rag.graph_context(state["question"], namespace=context.namespace)
    state["graph_context"] = graph_context
    state["context"] = (
        f"{passage_context}\n\nRelationships across documents:\n{graph_context}"
        if graph_context and passage_context
        else (passage_context or graph_context)
    )
    state["sources"] = list(dict.fromkeys(result.source for result in results))
    state["steps"].append(
        AgentStep(
            name="retrieve",
            detail=(
                f"{len(results)} chunk(s) from {len(state['sources'])} source(s)"
                + (", plus graph relations" if graph_context else "")
            ),
            data={
                "sources": state["sources"],
                "top_score": round(results[0].score, 4) if results else None,
            },
        )
    )
    return state


def tool_node(state: State, context: AgentContext) -> State:
    """Run a deterministic tool when the question clearly calls for one.

    This baseline uses rule based tool selection rather than model driven function
    calling so that the tool path is testable and works with any provider, including
    the offline one. Swapping in native function calling is a change confined to this
    node.
    """
    if not state.get("wants_tool"):
        return state

    question = state["question"]
    expression = state.get("expression", "")

    if expression:
        result = context.tools.call("calculate", expression=expression)
    elif re.search(r"\b(current time|what time|today's date)\b", question, re.IGNORECASE):
        result = context.tools.call("current_time")
    else:
        return state

    state["tool_output"] = result.content if result.ok else ""
    state["steps"].append(
        AgentStep(
            name="tool",
            detail=result.content if result.ok else (result.error or "tool failed"),
            data={"ok": result.ok},
        )
    )
    return state


def generate_node(state: State, context: AgentContext) -> State:
    """Produce the final answer with the LLM layer."""
    messages: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

    # These prefixes are a contract, not cosmetics. Providers that cannot reason (the
    # offline extractive one) rely on them to tell source material apart from
    # instructions.
    if state.get("context"):
        messages.append(
            ChatMessage(role="system", content=f"{CONTEXT_PREFIX}\n\n{state['context']}")
        )
    if state.get("tool_output"):
        messages.append(ChatMessage(role="system", content=f"{TOOL_PREFIX} {state['tool_output']}"))
    messages.append(ChatMessage(role="user", content=state["question"]))

    with get_tracer().span("llm.complete", provider=context.llm.name) as span:
        response = context.llm.complete(messages)
        span.set(model=response.model, latency_ms=response.latency_ms)

    state["answer"] = response.text
    state["llm_provider"] = response.provider
    state["llm_model"] = response.model
    state["steps"].append(
        AgentStep(
            name="generate",
            detail=f"{response.provider}/{response.model} in {response.latency_ms}ms",
            data={"latency_ms": response.latency_ms},
        )
    )
    return state


def initial_state(question: str) -> State:
    """Build the starting state for a run."""
    return {
        "question": question,
        "needs_knowledge": True,
        "is_pure_math": False,
        "expression": "",
        "wants_tool": False,
        "context": "",
        "graph_context": "",
        "results": [],
        "sources": [],
        "tool_output": "",
        "answer": "",
        "llm_provider": "",
        "llm_model": "",
        "steps": [],
    }
