# ZS-005: Retrieval is hybrid by default

Status: Accepted
Date: 2026-08-25

## Context

Vector search finds passages that mean something similar to the query. Keyword
search finds passages that contain the query's exact terms. The queries people
actually type sit squarely in the gap between them.

A part number, an error code, a policy identifier or a proper noun is invisible
to an embedding model that never saw it during training. A question phrased
entirely in synonyms is invisible to a keyword index. Choosing one retriever
means choosing which half of your users to fail.

## Decision

Both run, and their results are fused. The default mode is `auto`, which uses
both whenever the keyword index is populated.

Fusion is by rank position rather than by raw score. BM25 scores and cosine
similarities are on different, unnormalised scales, so adding them lets whichever
happens to be numerically larger decide the order regardless of which retriever
was actually more confident. Rank is comparable across systems; score is not.

BM25 is implemented in pure Python with no dependencies, so the keyword tier
works on the offline path like every other layer.

The keyword index is rebuilt from the vector store at construction. It lives in
memory while the vector store persists, so without the rebuild a second run
would start with an empty keyword tier and silently degrade to vector only, with
nothing in the logs to say so.

## Consequences

Good:

- Exact identifiers become retrievable, which purely semantic retrieval cannot do.
- No new dependency, and the tier works offline.
- The relevance cutoff on fused scores prefers agreement, because a chunk both
  retrievers returned scores roughly twice one that appeared in a single list.

Costs:

- Two searches per query instead of one. At this corpus size the keyword search
  is microseconds, but a very large index would want a real inverted index rather
  than this one.
- The keyword index duplicates chunk text in memory.
- Fusion weights are configuration nobody has tuned per corpus. The defaults are
  equal weighting, which is the right prior without evidence.
