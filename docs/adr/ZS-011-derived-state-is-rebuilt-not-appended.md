# ZS-011: Derived state is rebuilt, not appended

Status: Accepted
Date: 2026-08-25

## Context

Three structures are derived from the documents in the vector store: the keyword
index, the knowledge graph, and the answer cache. Ingestion kept them current by
adding to them. That is correct for a document that is new and wrong for one that
changed, and a corpus that is maintained rather than loaded once is mostly the
second case.

The keyword index was fine by accident: its entries are addressed by chunk id, so
re-ingesting a document replaces them. The other two are not addressable that way
and both went stale.

**The graph accumulated superseded relations.** A graph edge carries no chunk
identity, so appending the corrected text left the previous version's relations
in place. After a document changed a reporting line, a traversal returned both
the old and the new relation, each cited to the same live source, with nothing to
indicate which was current. A model given that context has no way to choose
correctly.

**The cache could not invalidate what it had no provenance for.** An answer
produced when retrieval found nothing records no sources. Source based
invalidation matches on the intersection of an entry's sources with what was
ingested, and the intersection with an empty list is always empty, so the entry
survived. The question asked just before its answering document was ingested kept
returning "nothing found" for the rest of its hour long TTL. Making invalidation
correctly targeted, in the change immediately before this one, is what exposed
this: the previous blanket invalidation had been hiding it.

## Decision

Derived state is rebuilt from its source when its source changes, unless every
element of it is individually addressable by the identity that changed.

- Upserting chunks rebuilds the knowledge graph for each affected namespace from
  what the store now holds, rather than adding the new text to the existing
  graph. Only the affected namespaces are rebuilt, so one tenant's ingestion does
  not cost a rebuild of another's.
- Ingestion invalidates cache entries with empty provenance in the target
  namespace, alongside the entries whose sources it rewrote. An answer built from
  an absence of knowledge is invalidated by that absence ending.
- The keyword index keeps its incremental path, because chunk id addressing makes
  it a true replacement. Building the graph was moved out of the keyword indexing
  function so that the difference is visible rather than incidental.

## Consequences

Good:

- A corrected document produces a corrected graph. The superseded relation is
  gone rather than presented alongside its replacement.
- A question asked before its document existed is answered from the document as
  soon as the document arrives.
- The invalidation the module documents is the invalidation it performs.

Bad, and accepted:

- Ingesting one document rebuilds the whole graph for its namespace, which is
  linear in that namespace's chunk count. Ingestion already pays for embedding,
  so this is not the dominant cost, but it will want incremental edge removal
  before a namespace grows very large.
- Dropping ungrounded entries on ingest discards some answers that were correctly
  ungrounded and would have stayed so. Recomputing a cheap "nothing found" answer
  is a far smaller cost than serving one that has become wrong.

## Notes

The general rule: if a derived structure cannot delete by the identity that
changed, it must be rebuilt from the source of truth. Appending is only safe when
replacement is addressable.
