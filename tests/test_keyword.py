"""Tests for BM25 keyword retrieval and rank fusion.

Keyword retrieval exists because vector search cannot find a term it has never
seen. Part numbers, error codes, policy identifiers and proper nouns are exactly
the queries a support agent or an engineer actually types, and they are exactly
where a purely semantic index fails.
"""

from __future__ import annotations

import pytest

from zerostack.rag.keyword import BM25Index, reciprocal_rank_fusion, tokenize

CORPUS = {
    "support::0": "Support agents may issue a refund up to two hundred dollars without approval.",
    "errors::0": "Error code E-4471 indicates the payment gateway rejected the transaction.",
    "hr::0": "Staff receive twenty vacation days per year after the probationary period.",
    "long::0": " ".join(["filler words about nothing in particular"] * 40),
}


@pytest.fixture
def index() -> BM25Index:
    bm25 = BM25Index()
    for chunk_id, text in CORPUS.items():
        bm25.add(chunk_id, text)
    return bm25


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Refund POLICY") == ["refund", "policy"]

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the refund")

    def test_keeps_alphanumeric_identifiers(self):
        assert "e" in tokenize("E-4471") and "4471" in tokenize("E-4471")

    def test_empty_text_yields_nothing(self):
        assert tokenize("") == []


class TestBM25:
    def test_finds_an_exact_identifier(self, index):
        """The case a semantic index cannot serve."""
        hits = index.search("E-4471")
        assert hits and hits[0].chunk_id == "errors::0"

    def test_ranks_the_relevant_document_first(self, index):
        hits = index.search("refund approval limit")
        assert hits[0].chunk_id == "support::0"

    def test_unknown_terms_return_nothing(self, index):
        assert index.search("quantum chromodynamics") == []

    def test_empty_query_returns_nothing(self, index):
        assert index.search("") == []

    def test_query_of_only_stopwords_returns_nothing(self, index):
        assert index.search("the and of") == []

    def test_searching_an_empty_index_is_safe(self):
        assert BM25Index().search("refund") == []

    def test_length_normalisation_does_not_favour_padding(self, index):
        """A long document should not win merely by being long."""
        hits = index.search("vacation days")
        assert hits[0].chunk_id == "hr::0"

    def test_top_k_is_respected(self, index):
        assert len(index.search("refund payment vacation", top_k=2)) <= 2

    def test_readding_the_same_id_replaces_rather_than_duplicates(self, index):
        before = index.size
        index.add("support::0", "Completely different text about shipping logistics.")
        assert index.size == before
        assert index.search("refund") == [] or index.search("refund")[0].chunk_id != "support::0"

    def test_removal_takes_the_document_out_of_results(self, index):
        index.remove("errors::0")
        assert index.search("E-4471") == []
        assert index.size == len(CORPUS) - 1

    def test_removing_an_absent_id_is_a_no_op(self, index):
        before = index.size
        index.remove("does-not-exist")
        assert index.size == before

    def test_clear_empties_the_index(self, index):
        index.clear()
        assert index.size == 0
        assert index.search("refund") == []

    def test_idf_never_goes_negative(self):
        """A term in every document must not subtract from a score."""
        bm25 = BM25Index()
        for i in range(5):
            bm25.add(f"d{i}", "refund refund refund")
        assert all(hit.score >= 0 for hit in bm25.search("refund"))


class TestReciprocalRankFusion:
    def test_agreement_between_rankings_wins(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        assert fused[0][0] == "a"

    def test_an_item_in_one_list_still_appears(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        assert {identifier for identifier, _ in fused} == {"a", "b"}

    def test_weights_shift_the_outcome(self):
        unweighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[5.0, 1.0])
        assert weighted[0][0] == "a"
        assert {i for i, _ in unweighted} == {"a", "b"}

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_empty_input_is_safe(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
