# Task queue

**Last updated:** 2026-08-24
**Current phase:** Foundation complete, hardening next
**Active mode:** BUILD

---

## Status legend

- PLANNED, defined but not started
- IN PROGRESS, being worked on now
- BLOCKED, waiting on a dependency or a decision
- IMPLEMENTED, code complete
- VALIDATED, tested and reviewed
- SHIPPED, merged to main

---

## Shipped

### Foundation, 2026-08-24

| Task | Status | Notes |
|------|--------|-------|
| Configuration for all eight layers | VALIDATED | One settings section per layer, every selector accepts `auto` |
| Observability layer | VALIDATED | JSONL spans, optional OTLP export to Phoenix |
| LLM layer | VALIDATED | Ollama plus offline extractive provider |
| RAG pipeline | VALIDATED | Paragraph chunking, two embedding backends, four store backends |
| Orchestrator adapter | VALIDATED | Shared nodes, three engines, parity asserted by test |
| Tool layer and MCP adapter | VALIDATED | Registry, AST guarded calculator, `mcp.json` discovery |
| Data layer | VALIDATED | SQLite state, DuckDB analytics with SQLite fallback |
| FastAPI service | VALIDATED | Six endpoints, validation on the boundary |
| Typer CLI | VALIDATED | doctor, ingest, ask, demo, runs, analytics, serve |
| Streamlit frontend | IMPLEMENTED | Not covered by automated tests, see below |
| Docker Compose, Makefile, CI | VALIDATED | Qdrant, Ollama, Phoenix; CI runs with no services |
| Governance docs and ADRs | VALIDATED | CLAUDE.md, six modes, five ADRs |
| Test suite | VALIDATED | 148 tests, no network, no Docker, no model server |

---

## Next up

### 1. Model driven tool selection, PLANNED

Tool selection is rule based today. It is deterministic and testable, but it does not
generalise past arithmetic and clock reads.

- Add native function calling to `tool_node` when the provider supports it.
- Keep the rule based path as the fallback so the offline provider still works.
- Assert both paths reach the same result for the arithmetic cases.

Acceptance: a question needing an MCP tool selects it without a hand written rule.

### 2. Reranking, PLANNED

The relevance ratio cutoff is crude. A cross encoder reranker over the top k would
materially improve precision.

- Add a `Reranker` protocol with a no op default so the offline path is unaffected.
- Ship a cross encoder implementation behind the `embeddings` extra.

Acceptance: retrieval precision improves on a fixed question set, offline path
unchanged.

### 3. Streaming responses, PLANNED

`LLMClient.complete` is synchronous and returns a whole response. A chat UI needs
tokens as they arrive.

- Add an optional `stream()` method to the protocol, defaulting to yielding the whole
  response once.
- Add an SSE endpoint to the API.

Acceptance: the Streamlit app renders tokens incrementally against Ollama.

### 4. Authentication, BLOCKED

Blocked on a decision: is the first deployment single tenant or multi tenant? The
answer changes the data model, not just the middleware.

- Single tenant: an API key check on the FastAPI boundary is enough.
- Multi tenant: every table needs a tenant column, every query needs a tenant filter,
  and the vector collection needs per tenant partitioning.

Do not start until that is settled. Retrofitting tenancy is far more expensive than
building it in.

### 5. Next.js frontend, PLANNED

The API exists for this. Deferred until a project needs a client facing UI.

Acceptance: a Next.js app on the Vercel free tier consuming `/ask` with citations.

---

## Known gaps

Recorded rather than hidden.

| Gap | Impact | Mitigation |
|-----|--------|------------|
| CrewAI engine untested in CI | Needs a live model server | Structurally correct, unverified end to end. Verify before using it on a project |
| Embedded Qdrant locks its directory | Two local processes conflict | Second process degrades to in process, documented in RUNBOOK |
| Hashing embeddings are lexical | Offline retrieval misses synonyms | Install the `embeddings` extra |
| No rate limiting on the API | An open deployment can be exhausted | Add before any public exposure |
| No multi tenancy | One deployment serves one customer | Task 4 above |

---

## Change log

### 2026-08-24, later

Bug hunt against the surfaces the first pass did not cover. Six defects found and
fixed, each with a regression test:

- The tracer held the current trace id and open span stack on the instance, so
  concurrent API requests overwrote each other. Spans were misattributed and the
  wrong `trace_id` was written to the run record. Now thread local.
- The tracer retained every span for the life of the process, a memory leak in a
  long running server. Now bounded by `max_retained_spans`, default 1000.
- `last_trace()` returned every span ever recorded rather than the last run's.
  `summary()` now scopes to the current trace.
- `QdrantVectorStore.reset()` deleted the collection without recreating it, so the
  next upsert raised "collection not found" while the memory and chroma backends
  stayed usable. Backends must behave alike or they are not swappable.
- A negative `chunk_overlap` was accepted and silently dropped most of the
  document, roughly 80 percent in the observed case. Now rejected.
- The DuckDB `ATTACH` interpolated the database path into SQL, breaking on any
  path containing a quote. Now escaped. DuckDB's `ATTACH` accepts no prepared
  statement parameters, so escaping is the available fix.

Two further findings from checking that the new tests actually ran in CI rather
than skipping:

- Engine detection probed the `langgraph` namespace package. `langgraph-checkpoint`,
  `langgraph-sdk` and `langgraph-prebuilt` are separate distributions that arrive as
  transitive dependencies and make that namespace resolve on their own, so
  `zerostack doctor` reported LangGraph as available while importing
  `langgraph.graph` still failed. Detection now probes the module actually imported.
- The orchestrator matrix job ran only `zerostack demo`, never `pytest`. The
  LangGraph parity test skips when LangGraph is absent, so it ran nowhere and the
  central claim of ADR ZS-003 went unverified in CI. The matrix job now runs the
  suite.

Running the suite in the matrix job then exposed a third: a test named
`test_defaults_load_without_any_environment` never cleared the environment, so it
asserted nothing and passed only because no `ZEROSTACK_` variable happened to be
set. It now clears them, and a paired test asserts an exported value still wins.
The whole suite is verified against every environment combination CI uses.

Also closed two coverage gaps: the Streamlit app is now driven by `AppTest`, and
Chroma has its own suite. Both extras are installed in CI so neither skips. The
Streamlit app no longer uses the deprecated `use_container_width`.

### 2026-08-24

- Initial foundation. All eight layers implemented, tested and documented.
- Decisions recorded as ADRs ZS-001 through ZS-004.
- Governance ported from the organisation ADRs GLOBAL-001, GLOBAL-002 and GLOBAL-003.
