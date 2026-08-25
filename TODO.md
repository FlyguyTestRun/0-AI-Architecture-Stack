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
| Governance docs and ADRs | VALIDATED | AGENTS.md, six modes, five ADRs |
| Test suite | VALIDATED | 359 tests, no network, no Docker, no model server |

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
| Rate limiting is per process | Replicas each permit the full rate | Needs a shared limiter before horizontal scaling |
| Namespace filter is not pushed into the backend | A scoped query over fetches | Correct but not efficient; push down before hundreds of tenants |
| Graph extraction is deterministic | Finds relation, not relation type | Swap the extractor protocol where a model is available |

---

## Change log

### 2026-08-25, expansion

Built out the platform for both audiences in one codebase.

| Capability | What it closes |
|------------|----------------|
| Hybrid BM25 plus vector retrieval | Exact identifiers were unfindable by a purely semantic index |
| Graph retrieval tier | Questions whose answer spans documents that never reference each other |
| Metrics and Prometheus endpoint | Traces described one request; nothing described the fleet |
| Token accounting and spend ceilings | A local model has no invoice, so cost was invisible until it was not |
| Semantic answer cache | Repeated questions re-ran the whole pipeline |
| Identity, roles and namespaces | No way to serve more than one tenant or restrict who sees what |
| Rate limiting | An endpoint that runs a model on every call is a denial of wallet |
| Evaluation harness and CI gate | Quality regressions merged silently, since retrieval failure raises nothing |

Defects found and fixed while building, each now covered:

- Fusion bypassed the relevance cutoff, so fused results reported sources that
  had not grounded the answer. The cutoff now applies to every mode.
- Graph scaffolding was scored as content by the extractive provider, exactly as
  citation headers had been, so labels outranked the sentences they introduced.
- A sentence reaching the context twice, as a passage and as graph evidence, was
  answered twice verbatim. Sentences are now deduplicated.
- Cache hits returned before persistence, silently emptying the audit trail the
  moment caching was enabled, and left concurrent callers indistinguishable.
- Rate limiting was applied only to ask and ingest, leaving the observability
  endpoints unbounded. It now sits in the authentication dependency.
- The evaluation harness found two questions retrieving the right document and
  omitting the answering sentence. The extractive window moved from four to six.

Known limits recorded rather than hidden: namespace filtering is applied after
the search rather than pushed into the backend, rate limiting is per process, and
graph extraction finds that entities are related but not how.

### 2026-08-24, fifth pass

Three findings from a second automated review, all three verified and all three
real. Two of them are follow ups to fixes made earlier the same day, which is a
useful reminder that a fix is not done until the property it claims is tested.

- **The unique source fix was incomplete.** Deriving identity from the path
  relative to the ingest argument fixed the case it was tested against and left
  two others open: ingesting two sibling directories separately still collided,
  and ingesting a parent then a child produced a stale duplicate. Identity is now
  the resolved absolute path, separated from the human readable display name that
  appears in citations. The two answer different questions and should not have
  been the same value.
- **The refusal disclosed what it was protecting.** The ingest allowlist added
  earlier returned the requested path and every configured root in its 403 body,
  which the API forwards to an unauthenticated caller. A refusal became a way to
  probe the filesystem and locate the corpus. The detail now goes to the server
  log and the caller gets a generic message that still explains how to widen the
  boundary.
- **The attribution rule was only half enforced.** The pre-commit hook scans
  staged file diffs, which never contain the commit message, yet the rule names
  commit message trailers explicitly. A `commit-msg` hook now covers that half,
  matching by shape rather than by brand and reusing the shared dash checker
  rather than reimplementing the match in shell.

### 2026-08-24, fourth pass

Four findings from an automated review on the merged pull request, each verified
independently before acting. Two were correct and serious, one was correct, and one
did not reproduce as described but pointed at a real defect next to it.

- **Configuration silently ignored.** Each settings section is built by its own
  factory, independently of the outer object, so only the outer object read the
  dotenv file. The documented "copy .env.example to .env" flow therefore did
  nothing: a deployment could name a provider explicitly and still run on the
  offline fallback with no warning. Every section now reads it.
- **Documents silently lost on ingestion.** Chunk ids derive from the source, and
  the source was the bare filename, so `hr/policy.md` and `legal/policy.md`
  collided and the second replaced the first. Verified: two distinct files went in,
  one came out. Sources are now paths relative to the ingest root. A company
  document tree almost always repeats filenames, so this was the ordinary case
  rather than an edge case.
- **Dependency floor too low.** `query_points` arrived in qdrant-client 1.10, but
  the floor allowed 1.9, which exposes only the older `search` API. A resolve to
  the floor would ingest happily and then fail every retrieval. Floor raised.
- **Unbounded expression evaluation.** The reported integer exponentiation attack
  does not reproduce, because operands are cast to float and overflow immediately.
  Probing it did surface a real one: a long expression raised an uncaught
  `RecursionError` during parsing. Expression length is now capped before parsing,
  and stack and memory exhaustion are reported as bad input rather than escaping.

### 2026-08-24, third pass

Security finding, the most serious of the session. The ingest endpoint accepted any
filesystem path from the caller, so anyone who could reach the API could index an
arbitrary readable directory and then read its contents back out through the ask
endpoint. Verified end to end by planting a credentials file outside the corpus and
retrieving its contents through a question. That is path traversal plus information
disclosure in one step.

Ingest is now restricted to a configured allowlist of roots, defaulting to the corpus
directory. Paths are resolved before comparison so traversal segments and symlinks
cannot walk around it, and the boundary is checked before existence so a caller
cannot probe the filesystem for what exists. The CLI opts out deliberately, since an
operator with shell access can already read those files. Twelve tests cover the
boundary, including the end to end exfiltration path.

Also removed every vendor product name from the repository, renamed the operating
rules file to the vendor neutral `AGENTS.md`, generalized the pre-commit attribution
check to match by shape rather than by brand, and rewrote the README as a dual
audience document with a purpose statement, an honest fit critique, and a growth path.

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
