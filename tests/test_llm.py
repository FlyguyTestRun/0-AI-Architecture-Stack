"""Tests for the LLM layer (layer 4)."""

from __future__ import annotations

from zerostack.config import LLMSettings
from zerostack.llm.base import ChatMessage
from zerostack.llm.offline import CONTEXT_PREFIX, TOOL_PREFIX, OfflineLLM
from zerostack.llm.ollama import OllamaLLM
from zerostack.llm.registry import build_llm

SYSTEM_RULES = "You are an assistant. If the context does not contain the answer, say so."
CONTEXT = (
    f"{CONTEXT_PREFIX}\n\n[1] source: support.md\n"
    "Support agents may issue a refund up to two hundred dollars without approval. "
    "The support lead may approve up to two thousand dollars."
)


class TestOfflineLLM:
    def test_is_always_available(self):
        assert OfflineLLM().available() is True

    def test_extracts_the_answering_sentence(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=SYSTEM_RULES),
                ChatMessage(role="system", content=CONTEXT),
                ChatMessage(role="user", content="How much can an agent refund?"),
            ]
        )
        assert "two hundred dollars" in response.text
        assert response.provider == "offline"

    def test_does_not_leak_the_instruction_prompt(self):
        """The system rules are instructions, not source material."""
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=SYSTEM_RULES),
                ChatMessage(role="system", content=CONTEXT),
                ChatMessage(role="user", content="How much can an agent refund?"),
            ]
        )
        assert "You are an assistant" not in response.text
        assert "say so" not in response.text

    def test_does_not_return_citation_headers_as_content(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=CONTEXT),
                ChatMessage(role="user", content="refund approval limit"),
            ]
        )
        assert "source: support.md" not in response.text

    def test_tool_output_is_returned_when_there_is_no_context(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=f"{TOOL_PREFIX} 90.0"),
                ChatMessage(role="user", content="(120 * 3) / 4"),
            ]
        )
        assert response.text == "90.0"
        assert response.usage["used_tool_output"] is True

    def test_says_so_when_there_is_no_context_at_all(self):
        response = OfflineLLM().complete(
            [ChatMessage(role="user", content="What is our refund policy?")]
        )
        assert "No context was retrieved" in response.text

    def test_reports_when_context_does_not_answer(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=CONTEXT),
                ChatMessage(role="user", content="zzzz qqqq unrelated"),
            ]
        )
        assert "does not appear to answer" in response.text

    def test_is_deterministic(self):
        messages = [
            ChatMessage(role="system", content=CONTEXT),
            ChatMessage(role="user", content="refund approval"),
        ]
        assert OfflineLLM().complete(messages).text == OfflineLLM().complete(messages).text


class TestOllamaAvailability:
    def test_unreachable_server_reports_unavailable(self):
        client = OllamaLLM(base_url="http://127.0.0.1:1")
        assert client.available() is False
        assert client.installed_models() == []


class TestRegistry:
    def test_echo_provider_is_selected_explicitly(self):
        client = build_llm(LLMSettings(provider="echo"))
        assert client.name == "offline"

    def test_explicit_ollama_is_honoured_even_when_unreachable(self):
        """An explicit choice must fail loudly rather than degrade silently."""
        client = build_llm(LLMSettings(provider="ollama", base_url="http://127.0.0.1:1"))
        assert client.name == "ollama"

    def test_auto_falls_back_when_ollama_is_missing(self):
        client = build_llm(LLMSettings(provider="auto", base_url="http://127.0.0.1:1"))
        assert client.name == "offline"


class TestScaffoldingAndDuplicates:
    """Context now carries graph scaffolding as well as citation headers.

    Structural text labels the context; it is not part of it. Scoring it let a
    label outrank the sentence it introduced, which is the same defect citation
    headers caused before they were stripped.
    """

    GRAPH_CONTEXT = (
        f"{CONTEXT_PREFIX}\n\n[1] source: escalation.md\n"
        "The Escalation Policy is owned by the Engineering Director.\n\n"
        "Relationships across documents:\n"
        "Related to: support lead, engineering director\n"
        "- support lead and refund policy [refunds.md]: "
        "The Refund Policy is owned by the Support Lead."
    )

    def test_graph_labels_do_not_reach_the_answer(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=self.GRAPH_CONTEXT),
                ChatMessage(role="user", content="Who owns the Refund Policy?"),
            ]
        )
        assert "Relationships across documents" not in response.text
        assert "Related to:" not in response.text

    def test_graph_evidence_still_reaches_the_answer(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=self.GRAPH_CONTEXT),
                ChatMessage(role="user", content="Who owns the Refund Policy?"),
            ]
        )
        assert "Support Lead" in response.text

    def test_the_relation_triple_is_dropped_but_not_its_evidence(self):
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=self.GRAPH_CONTEXT),
                ChatMessage(role="user", content="Who owns the Refund Policy?"),
            ]
        )
        assert "[refunds.md]:" not in response.text

    def test_a_sentence_present_twice_appears_once(self):
        """The same sentence can arrive as a passage and as graph evidence."""
        duplicated = (
            f"{CONTEXT_PREFIX}\n\nThe Support Lead owns the Refund Policy.\n\n"
            "The Support Lead owns the Refund Policy."
        )
        response = OfflineLLM().complete(
            [
                ChatMessage(role="system", content=duplicated),
                ChatMessage(role="user", content="Who owns the Refund Policy?"),
            ]
        )
        assert response.text.count("The Support Lead owns the Refund Policy") == 1
