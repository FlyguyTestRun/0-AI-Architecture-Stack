"""Tests for the Graph RAG tier.

The graph exists to answer questions whose answer is spread across documents
that never mention each other. Passage retrieval cannot reach those, because no
single passage contains the answer.
"""

from __future__ import annotations

import pytest

from zerostack.rag.graph import (
    KnowledgeGraph,
    PatternEntityExtractor,
    format_graph_context,
)

DOCUMENTS = {
    "refunds.md": (
        "The Refund Policy is owned by the Support Lead. "
        "Refunds above two thousand dollars require Finance Team approval."
    ),
    "escalation.md": (
        "The Escalation Policy is owned by the Engineering Director. "
        "The Escalation Policy governs Severity 1 handling."
    ),
    "oncall.md": (
        "Severity 1 incidents page the Engineering Director immediately. "
        "The Support Lead may raise a ticket to Severity 1."
    ),
}


@pytest.fixture
def graph() -> KnowledgeGraph:
    knowledge = KnowledgeGraph()
    for source, text in DOCUMENTS.items():
        knowledge.add_document(text, source)
    return knowledge


class TestExtraction:
    def test_finds_multi_word_names(self):
        names = {e.name for e in PatternEntityExtractor().extract("The Support Lead approved it.")}
        assert "Support Lead" in names

    def test_strips_a_leading_article(self):
        """ "The Support Lead" and "Support Lead" must be one node, not two."""
        names = {e.name for e in PatternEntityExtractor().extract("We told the Support Lead.")}
        assert "Support Lead" in names
        assert "The Support Lead" not in names

    def test_finds_identifiers(self):
        names = {e.name for e in PatternEntityExtractor().extract("See GDPR and ISO 27001.")}
        assert "GDPR" in names

    def test_ignores_a_sentence_initial_common_word(self):
        """Every sentence starts with a capital, so position is not evidence."""
        names = {e.name for e in PatternEntityExtractor().extract("Only the policy applies.")}
        assert "Only" not in names

    def test_keeps_an_all_caps_token_in_first_position(self):
        names = {e.name for e in PatternEntityExtractor().extract("GDPR applies to this data.")}
        assert "GDPR" in names

    def test_empty_text_yields_nothing(self):
        assert PatternEntityExtractor().extract("") == []


class TestGraphConstruction:
    def test_entities_and_relations_are_built(self, graph):
        assert graph.entity_count > 0
        assert graph.relation_count > 0

    def test_only_real_entities_become_nodes(self, graph):
        names = {graph.entity(key).name for key in ["support lead", "engineering director"]}
        assert names == {"Support Lead", "Engineering Director"}

    def test_repeated_pairs_reinforce_rather_than_duplicate(self):
        knowledge = KnowledgeGraph()
        text = "The Support Lead owns the Refund Policy."
        knowledge.add_document(text, "a.md")
        first = knowledge.relation_count
        knowledge.add_document(text, "b.md")
        assert knowledge.relation_count == first

    def test_relations_record_every_source(self):
        knowledge = KnowledgeGraph()
        text = "The Support Lead owns the Refund Policy."
        knowledge.add_document(text, "a.md")
        knowledge.add_document(text, "b.md")
        relation = next(iter(knowledge._relations.values()))
        assert relation.sources == {"a.md", "b.md"}

    def test_entities_are_linked_within_a_sentence_not_a_document(self):
        """Document wide linking connects everything to everything."""
        knowledge = KnowledgeGraph()
        knowledge.add_document(
            "The Support Lead is here.\n\nThe Finance Team is elsewhere.", "a.md"
        )
        assert "finance team" not in knowledge.neighbours("support lead")

    def test_clear_empties_the_graph(self, graph):
        graph.clear()
        assert graph.entity_count == 0
        assert graph.relation_count == 0


class TestTraversal:
    def test_finds_the_entities_a_query_names(self, graph):
        assert "support lead" in graph.traverse("What does the Support Lead own?").seeds

    def test_a_lowercase_mention_still_matches(self, graph):
        assert graph.traverse("who is the support lead").seeds

    def test_an_unknown_query_returns_an_empty_neighbourhood(self, graph):
        assert graph.traverse("quantum chromodynamics").is_empty

    def test_two_hops_connect_documents_that_never_reference_each_other(self, graph):
        """The property the whole tier exists for."""
        neighbourhood = graph.traverse("Support Lead", hops=2)
        reached = {name.lower() for name in neighbourhood.entities}
        assert "engineering director" in reached
        assert {"refunds.md", "escalation.md", "oncall.md"} <= set(neighbourhood.sources)

    def test_one_hop_stays_local(self, graph):
        one = graph.traverse("Support Lead", hops=1)
        two = graph.traverse("Support Lead", hops=2)
        assert len(one.entities) <= len(two.entities)

    def test_zero_hops_returns_only_the_seeds(self, graph):
        neighbourhood = graph.traverse("Support Lead", hops=0)
        assert [name.lower() for name in neighbourhood.entities] == ["support lead"]

    def test_entity_budget_is_respected(self, graph):
        assert len(graph.traverse("Support Lead", hops=3, max_entities=2).entities) <= 2


class TestContextFormatting:
    def test_empty_neighbourhood_renders_nothing(self, graph):
        assert format_graph_context(graph.traverse("nothing here")) == ""

    def test_context_carries_evidence_and_a_source(self, graph):
        rendered = format_graph_context(graph.traverse("Support Lead"))
        assert "Support Lead" in rendered or "support lead" in rendered
        assert ".md]" in rendered

    def test_relation_budget_is_respected(self, graph):
        rendered = format_graph_context(graph.traverse("Support Lead"), max_relations=1)
        assert rendered.count("\n- ") <= 1


class TestGraphIsRebuiltOnReingest:
    """A corrected document must not leave its old relations behind.

    Vector and keyword entries are addressed by chunk id, so re-ingesting
    replaces them. A graph edge carries no chunk identity, so appending the new
    text leaves the previous version's relations in place and a traversal then
    presents the old and the new fact side by side, both cited to the same live
    source, with nothing to say which is current.
    """

    @pytest.fixture
    def pipeline(self):
        from zerostack.config import get_settings
        from zerostack.rag.pipeline import RAGPipeline

        settings = get_settings().rag
        settings.backend = "memory"
        return RAGPipeline(settings=settings)

    def test_the_superseded_relation_is_gone(self, pipeline):
        pipeline.ingest_text(
            "The Support Lead reports to the Engineering Director. "
            "The Engineering Director owns the escalation path.",
            source="org.md",
            source_id="default::org.md",
        )
        pipeline.ingest_text(
            "The Support Lead reports to the Operations Director. "
            "The Operations Director owns the escalation path.",
            source="org.md",
            source_id="default::org.md",
        )
        context = pipeline.graph_context("Who does the Support Lead report to?")
        assert "Operations Director" in context
        assert "Engineering Director" not in context

    def test_rebuilding_one_namespace_leaves_another_alone(self, pipeline):
        pipeline.ingest_text(
            "The Legal Counsel reports to the General Counsel.",
            source="legal.md",
            source_id="legal::legal.md",
            namespace="legal",
        )
        pipeline.ingest_text(
            "The Support Lead reports to the Field Director.",
            source="org.md",
            source_id="default::org.md",
        )
        legal = pipeline.graph_context("Who does the Legal Counsel report to?", namespace="legal")
        assert "General Counsel" in legal
        assert "Support Lead" not in legal

    def test_an_unrelated_document_still_contributes(self, pipeline):
        """Rebuilding must not drop the documents it is not replacing."""
        pipeline.ingest_text(
            "The Support Lead reports to the Engineering Director.",
            source="org.md",
            source_id="default::org.md",
        )
        pipeline.ingest_text(
            "The Billing Lead reports to the Finance Director.",
            source="finance.md",
            source_id="default::finance.md",
        )
        context = pipeline.graph_context("Who does the Support Lead report to?")
        assert "Engineering Director" in context
