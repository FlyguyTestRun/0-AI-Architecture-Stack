# ZS-002: Qdrant is the default vector store

Status: Accepted
Date: 2026-08-24

## Context

The reference architecture named both Chroma and Qdrant. Picking one as the default
matters more than the feature comparison, because the default is what every project
built on this baseline inherits.

The deciding constraint is growth. These projects start as an SMB pilot on a laptop
and some of them need to serve a real workload a year later. A default that forces a
migration at that moment is a bad default.

## Decision

Qdrant is the default. Chroma remains supported behind the same `VectorStore`
protocol.

Qdrant runs the same client API in three modes:

- in process (`:memory:`) for tests
- embedded on disk (a path) for local development with no Docker
- server (an http URL) for a deployment

Moving from a laptop to a cluster is a change to `ZEROSTACK_RAG_QDRANT_URL`. No code
change, no reindexing strategy, no second integration to write and test.

The `auto` chain is: server, then embedded on disk, then in process, then the pure
Python store. Embedded sits ahead of in process deliberately: an in process index
dies with the interpreter, so `zerostack ingest` followed by a separate
`zerostack ask` would find an empty collection. Embedded mode takes a directory lock,
so when another process holds it the chain falls through rather than failing.

## Consequences

Good:

- One API across test, laptop and production.
- No migration at the moment of growth, which is the moment least able to absorb one.
- Filtering and payload search are available when a project needs them.

Costs:

- Qdrant embedded mode locks its directory to one process. Running the Streamlit app
  and the CLI at once means the second one degrades to an in process index. This is
  documented in the runbook, and starting the Qdrant container removes the limit.
- Chroma is simpler for a purely single node use case, and this decision gives that
  up as the default. It stays available via `ZEROSTACK_RAG_BACKEND=chroma`.
