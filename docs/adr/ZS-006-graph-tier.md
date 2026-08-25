# ZS-006: A graph tier for questions spread across documents

Status: Accepted
Date: 2026-08-25

## Context

Vector and keyword retrieval answer the same shape of question: which passage
looks like this query. Neither can answer a question whose answer is spread
across documents that never mention each other, because no single passage
contains it.

"What connects the support lead to the engineering director" requires joining up
each document's mention of them. That join exists nowhere in the text, so no
amount of better passage retrieval will find it.

## Decision

A knowledge graph is built during ingestion. Entities become nodes, sentences
mentioning two entities become weighted edges carrying that sentence as evidence,
and a query walks outward from the entities it names.

Extraction is deterministic rather than model driven. A model extracts richer
relations, but it costs money per document, is nondeterministic, and cannot run
offline, which would put the entire tier behind a model server. The extractor
sits behind a protocol so a model driven one can replace it where available.

Three extraction rules decide whether the graph carries information at all:

- Entities are linked within a sentence, not within a document. Document wide
  co-occurrence connects everything to everything on any reasonably sized page,
  producing a dense graph that says nothing.
- A leading article is stripped, so "the support lead" and "support lead" are one
  node rather than two halves of one neighbourhood.
- A single capitalised word in first position is not a name, because every
  sentence starts with a capital. Without this, "Only", "Refunds" and "Staff"
  became nodes and connected unrelated documents through grammar.

Two hops by default: one returns only what a single sentence already said, and
three pulls in most of a small graph and stops being selective.

Each namespace gets its own graph, because a shared one would let a traversal
walk from one tenant's entity into another tenant's document.

## Consequences

Good:

- Cross document questions become answerable, and the answer carries the sentence
  that justifies each connection rather than a bare triple.
- Costs nothing when the graph does not recognise an entity in the question.
- No dependency, and it runs offline.

Costs:

- Deterministic extraction is weaker than a model's. It finds co-occurrence, not
  typed relations, so it can say two things are related but not how.
- The graph is held in memory and rebuilt from the vector store, so a very large
  corpus would need a real graph store.
- Feeding relations into the prompt required teaching the extractive provider to
  strip the scaffolding, which is a coupling between the tier and the provider.
