# Architecture

How the eight layers fit together, what each one guarantees, and where to change
things.

Last updated: 2026-08-24

---

## Principles

**One protocol per layer, at least two implementations.** Every layer defines a
`Protocol` and ships a preferred backend plus an offline fallback. Callers depend on
the protocol, never on a concrete class.

**Dependency direction is strict.** A layer may import from the layers below it and
never from the layers above.

```
frontend / api / cli
    -> app.ZerostackApp
        -> orchestrator -> {llm, rag, tools}
            -> config, observability
```

If a change requires the orchestrator to know the API exists, the design is wrong.

**Automatic degradation, explicit escalation.** `auto` picks the best backend that is
actually reachable. An explicit backend name is honoured even when the service is
absent, so production fails loudly rather than serving degraded answers quietly.

**Composition happens in exactly one place.** `ZerostackApp` builds every layer.
Nothing else constructs a backend. That is what makes the frontend swappable.

---

## Request flow

```
question
   │
   ▼
ZerostackApp.ask()
   │
   ▼
Orchestrator.run()                        engine: langgraph | crewai | simple
   │
   ├─▶ plan_node       decide: does this need the knowledge base? which tool?
   │                   also derives the arithmetic expression, if any
   │
   ├─▶ retrieve_node   embed query, search vector store, apply relevance cutoff,
   │                   format numbered context with source citations
   │
   ├─▶ tool_node       run the deterministic tool the planner selected
   │
   └─▶ generate_node   build messages, call the LLM layer, return the answer
   │
   ▼
StateStore.save_run()                     SQLite, never fails the answer
   │
   ▼
AgentResult { answer, sources, steps, latency, trace_id }
```

Every step above emits a span through the observability layer.

---

## Layer 1: Frontend

Three frontends, one dependency: `ZerostackApp`.

- **Streamlit** (`apps/streamlit_app/app.py`) exposes the internals on purpose:
  which backend served each layer, which sources grounded the answer, the agent steps
  and the raw spans. For an SMB pilot that visibility is what earns trust.
- **FastAPI** (`src/zerostack/api/main.py`) is the backend for a Next.js frontend.
  Endpoints: `/ask`, `/ingest`, `/health`, `/runs`, `/analytics`, `/tools`.
- **CLI** (`src/zerostack/cli.py`) is the operator interface. `zerostack doctor` is
  the command that matters on a new machine.

**To swap in Next.js:** deploy the FastAPI service, point the frontend at `/ask`, and
nothing below layer 1 changes.

## Layer 2: Orchestrator

The agent graph lives once, as pure functions over a state dictionary, in
`orchestrator/nodes.py`. Three engines execute those same functions:

| Engine | Mechanism | Requires |
|--------|-----------|----------|
| `simple` | sequential calls | nothing |
| `langgraph` | `StateGraph` with a conditional edge | `langgraph` extra |
| `crewai` | shared nodes, then a crew for reasoning | `crewai` extra and a live model server |

A test asserts LangGraph and `simple` produce identical answers, sources and step
sequences for the same question. Drift becomes a build failure.

`auto` prefers LangGraph and falls back to `simple`. It never selects CrewAI, because
CrewAI needs a reachable model server and choosing it silently would break an offline
run. See [ADR ZS-003](docs/adr/ZS-003-orchestrator-adapter.md).

**To change agent behaviour:** edit `nodes.py`. All three engines inherit the change.

**To add an engine:** implement `run(question) -> AgentResult` over the shared nodes
and register it in `factory.py`.

## Layer 3: RAG

```
document ─▶ chunk_text ─▶ EmbeddingProvider ─▶ VectorStore.upsert
query    ─▶ embed_one  ─▶ VectorStore.search ─▶ relevance cutoff ─▶ format_context
```

**Chunking** is paragraph aware with character overlap. Paragraphs stay intact when
they fit, which preserves more meaning than a fixed window at no extra cost. An
oversized paragraph is windowed rather than dropped.

**Embeddings** have two backends. `sentence-transformers` for real work, and a
deterministic hashed bag of words for tests and offline use. Both emit L2 normalised
vectors of the configured dimensionality, so the collection schema does not change
when you switch.

**Vector stores** implement one `VectorStore` protocol: Qdrant (in process, embedded,
or server), Chroma, and a pure Python cosine store.

**Relevance cutoff.** Top k alone always returns k chunks, including irrelevant ones
when the index is small, which overstates what actually grounded the answer. Hits
scoring below `relevance_ratio` of the best hit are dropped. A relative cutoff is used
because an absolute threshold is not portable across embedding backends.

Qdrant is the default because one client API covers laptop and cluster, so growth is a
URL change rather than a migration. See [ADR ZS-002](docs/adr/ZS-002-qdrant-default.md).

## Layer 4: LLM

`LLMClient` is two methods: `available()` and `complete(messages)`. Anything richer
belongs behind a provider specific adapter, so swapping providers never breaks the
orchestrator.

- **Ollama** for local models. Availability is probed with a short timeout so `auto`
  never blocks startup.
- **Offline extractive provider** scores the sentences in retrieved context against
  the question and returns the best ones in document order.

The offline provider relies on a contract: `generate_node` tags source material with
`Context:` and `Tool result:` prefixes. Without those tags the provider would score
the instruction prompt as if it were content and leak the rules into the answer. The
constants live in `llm/offline.py` and are imported by `nodes.py` so the contract is
explicit rather than a pair of matching string literals.

## Layer 5: Tools via MCP

One `ToolRegistry` holds built in tools and MCP discovered tools in the same shape.
The orchestrator resolves by name and cannot tell them apart.

Servers are declared in `mcp.json`, using the format Claude Code and Claude Desktop
already use. Discovery failures are contained per server.

Tool selection is rule based rather than model driven, which keeps the path
deterministic, testable, and functional with any provider including the offline one.
Native function calling is a change confined to `tool_node`.

The calculator parses to an AST and walks an explicit operator allowlist. It never
calls `eval`, because tool arguments can originate in retrieved documents and must be
treated as hostile. See [ADR ZS-004](docs/adr/ZS-004-mcp-for-tools.md).

## Layer 6: Code agent

Not runtime code. The conventions that let Claude Code or Aider work on this
repository productively: `CLAUDE.md` for operating rules, `modes/` for how much to ask
in each phase, `docs/adr/` for decisions that should not be relitigated.

## Layer 7: Data

**SQLite** holds application state: every run with its question, answer, engine,
provider, latency, sources and steps. It is the default because it needs no server,
it is transactional, and every deployment target can carry the file.

**DuckDB** reads the same SQLite file through its sqlite scanner for analytics, so
reporting never competes for the write lock. When the scanner is unavailable the same
aggregate is computed in SQLite, and the result shape is identical either way.

The DDL avoids SQLite specific types, so moving to Supabase Postgres is a connection
change plus the same schema.

Persistence never fails an answer. A failing `save_run` is logged and swallowed,
because bookkeeping must not cost the user their result.

## Layer 8: Deployment

`docker-compose.yml` provides Qdrant, Ollama and Phoenix. Partial stacks are supported
deliberately: start only Qdrant and the other layers keep using their fallbacks.

CI runs with no services at all, which is what keeps the offline path honest. A second
CI job runs the demo on each orchestrator engine.

## Observability

A span is recorded for every retrieval, LLM call, tool invocation and agent run.
Spans append to a JSONL file so a developer can inspect a run with no collector
running, and export to Phoenix over OTLP when `ZEROSTACK_OBS_EXPORT_TRACES` is on and
the OpenTelemetry packages are installed.

The tracer is small on purpose. The guarantee is that everything is traced by default,
not that the tracer is feature complete.

---

## Extending it

| Goal | Where to change | Ripple |
|------|-----------------|--------|
| Different agent behaviour | `orchestrator/nodes.py` | all three engines |
| New orchestration engine | implement `Orchestrator`, register in `factory.py` | none |
| Hosted model API | implement `LLMClient`, register in `llm/registry.py` | none |
| Different vector store | implement `VectorStore` | none |
| New integration | add a server to `mcp.json` | none |
| Postgres instead of SQLite | `data/store.py` connection, schema is portable | none |
| Next.js frontend | consume the FastAPI service | none |

---

## Known limits

- **Embedded Qdrant locks its directory.** Running the Streamlit app and the CLI at
  once means the second process degrades to an in process index. Starting the Qdrant
  container removes the limit.
- **The offline provider is extractive.** It cannot synthesise across sources or
  reason. It answers when retrieval worked and says so plainly when it did not.
- **Hashing embeddings are lexical.** They match shared vocabulary, not meaning.
  Install the `embeddings` extra for semantic retrieval.
- **Rule based tool selection does not generalise.** Open ended tool use needs model
  driven selection in `tool_node`.
- **Single tenant.** Add authentication and tenant isolation before serving more than
  one customer from one deployment.
