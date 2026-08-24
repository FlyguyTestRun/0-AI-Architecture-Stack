"""Tests for the orchestrator layer (layer 2) and its routing decisions."""

from __future__ import annotations

import pytest

from zerostack.config import OrchestratorSettings
from zerostack.llm.offline import OfflineLLM
from zerostack.orchestrator.base import AgentContext
from zerostack.orchestrator.factory import available_engines, build_orchestrator
from zerostack.orchestrator.nodes import initial_state, plan_node, tool_node
from zerostack.orchestrator.simple import SimpleOrchestrator


def plan(question: str, context: AgentContext) -> dict:
    return plan_node(initial_state(question), context)


class TestRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "How much can an agent refund without approval?",
            "When does health coverage begin?",
            "What is the escalation path?",
        ],
    )
    def test_knowledge_questions_route_to_retrieval(self, question, context):
        assert plan(question, context)["needs_knowledge"] is True

    @pytest.mark.parametrize(
        "question",
        ["(120 * 3) / 4", "2+2", "What is 15 * 4 + 2?", "what is 100/5"],
    )
    def test_arithmetic_questions_skip_retrieval(self, question, context):
        state = plan(question, context)
        assert state["needs_knowledge"] is False
        assert state["wants_tool"] is True

    def test_a_question_with_numbers_but_real_content_still_retrieves(self, context):
        """Numbers alone must not divert a genuine knowledge question."""
        state = plan("If we refund 3 customers at 200 each, what is the refund policy?", context)
        assert state["needs_knowledge"] is True

    def test_tools_can_be_disabled(self, rag, tools):
        context = AgentContext(
            llm=OfflineLLM(),
            rag=rag,
            tools=tools,
            settings=OrchestratorSettings(kind="simple", enable_tools=False),
        )
        assert plan("2 + 2", context)["wants_tool"] is False


class TestToolNode:
    def test_evaluates_a_parenthesised_expression_whole(self, context):
        state = tool_node(plan("(120 * 3) / 4", context), context)
        assert state["tool_output"] == "90.0"

    def test_evaluates_an_expression_embedded_in_words(self, context):
        state = tool_node(plan("What is 15 * 4 + 2?", context), context)
        assert state["tool_output"] == "62.0"

    def test_a_bare_number_is_not_treated_as_an_expression(self, context):
        """A lone number carries no operator, so no tool call should fire."""
        state = plan("What does section 200 say about refunds?", context)
        assert state["expression"] == ""

    def test_no_tool_runs_when_not_wanted(self, context):
        state = plan("What is the refund policy?", context)
        assert tool_node(state, context).get("tool_output", "") == ""


class TestSimpleOrchestrator:
    def test_answers_from_the_knowledge_base(self, context):
        result = SimpleOrchestrator(context).run("How much can an agent refund without approval?")
        assert "two hundred dollars" in result.answer
        assert "support.md" in result.sources
        assert [step.name for step in result.steps] == ["plan", "retrieve", "generate"]

    def test_answers_arithmetic_without_retrieval(self, context):
        result = SimpleOrchestrator(context).run("(120 * 3) / 4")
        assert result.answer == "90.0"
        assert result.sources == []
        assert [step.name for step in result.steps] == ["plan", "tool", "generate"]

    def test_result_is_fully_populated(self, context):
        result = SimpleOrchestrator(context).run("When does coverage begin?")
        assert result.orchestrator == "simple"
        assert result.llm_provider == "offline"
        assert result.latency_ms >= 0
        assert result.trace_id
        assert result.to_dict()["question"] == "When does coverage begin?"


class TestFactory:
    def test_simple_is_always_available(self):
        assert available_engines()["simple"] is True

    def test_explicit_simple_is_honoured(self, context):
        orchestrator = build_orchestrator(context, OrchestratorSettings(kind="simple"))
        assert orchestrator.name == "simple"

    def test_auto_never_selects_crewai(self, context):
        """CrewAI needs a live model server, so auto must not pick it."""
        orchestrator = build_orchestrator(context, OrchestratorSettings(kind="auto"))
        assert orchestrator.name in {"simple", "langgraph"}

    @pytest.mark.skipif(
        not available_engines()["langgraph"], reason="langgraph extra not installed"
    )
    def test_langgraph_produces_the_same_answer_as_simple(self, context):
        """The adapter must change the engine, never the behaviour."""
        question = "How much can an agent refund without approval?"
        simple = build_orchestrator(context, OrchestratorSettings(kind="simple")).run(question)
        graph = build_orchestrator(context, OrchestratorSettings(kind="langgraph")).run(question)
        assert graph.answer == simple.answer
        assert graph.sources == simple.sources
        assert [s.name for s in graph.steps] == [s.name for s in simple.steps]


class TestEngineDetection:
    """Engine detection must reflect what can actually be imported.

    "langgraph" is a namespace package. langgraph-checkpoint, langgraph-sdk and
    langgraph-prebuilt are separate distributions that arrive as transitive
    dependencies and make the namespace resolve on their own, so probing
    "langgraph" reported the engine as available while importing langgraph.graph
    still failed. That made `zerostack doctor` report a backend the process could
    not build.
    """

    def test_simple_is_always_reported_available(self):
        assert available_engines()["simple"] is True

    def test_detection_probes_the_imported_module(self):
        from zerostack.orchestrator.factory import _ENGINE_MODULES

        # Probing the bare namespace is what caused the false positive.
        assert _ENGINE_MODULES["langgraph"] == "langgraph.graph"

    def test_a_namespace_without_the_engine_reports_unavailable(self, monkeypatch):
        import zerostack.orchestrator.factory as factory

        def only_namespace(module: str):
            if module == "langgraph.graph":
                return None
            return object()

        monkeypatch.setattr(factory.importlib.util, "find_spec", only_namespace)
        assert factory.available_engines()["langgraph"] is False

    def test_detection_survives_a_raising_find_spec(self, monkeypatch):
        """find_spec raises when a parent package is missing or is not a package."""
        import zerostack.orchestrator.factory as factory

        def boom(module: str):
            raise ModuleNotFoundError(module)

        monkeypatch.setattr(factory.importlib.util, "find_spec", boom)
        engines = factory.available_engines()
        assert engines["langgraph"] is False
        assert engines["crewai"] is False
        assert engines["simple"] is True

    def test_auto_falls_back_when_the_engine_cannot_be_built(self, context, monkeypatch):
        """A false positive must degrade to simple, never crash a request."""
        import zerostack.orchestrator.factory as factory

        monkeypatch.setattr(factory, "_installed", lambda module: True)
        orchestrator = build_orchestrator(context, OrchestratorSettings(kind="auto"))
        assert orchestrator.name in {"simple", "langgraph"}
        assert orchestrator.run("When does coverage begin?").answer
