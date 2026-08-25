<div align="center">

# Zerostack

**A production shaped AI architecture that costs nothing to run.**

Eight layers. Self hosted. No API keys, no cloud bill, no vendor lock in.
Clone it, run one command, and you have a working AI assistant that answers
questions from your own documents and shows its sources.

[![CI](https://github.com/FlyguyTestRun/0-AI-Architecture-Stack/actions/workflows/ci.yml/badge.svg)](https://github.com/FlyguyTestRun/0-AI-Architecture-Stack/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-359%20passing-brightgreen.svg)](tests/)

[Quick start](#quick-start-60-seconds) ·
[What it does](#what-it-actually-does) ·
[Who it is for](#who-this-is-for) ·
[Where it does not fit](#where-this-is-the-wrong-tool) ·
[Technical reference](#technical-reference) ·
[Architecture](ARCHITECTURE.md) ·
[Roadmap](TODO.md)

</div>

---

## The short version

Most small and midsize companies want the same thing from AI: **ask a question in
plain English, get an answer drawn from our own documents, and be able to check
where that answer came from.**

Getting there normally means signing an enterprise contract, sending your
documents to somebody else's servers, and paying per question forever. For a
fifteen person company evaluating whether any of this is worth it, that is a
large commitment to make before you know the answer.

Zerostack removes that commitment. It is a complete, working system you run on
your own hardware, for free, that does the thing above. If it turns out to be
valuable, you scale it up. If it does not, you have spent an afternoon.

---

## Why this repository exists

Three specific problems, each of which costs real money.

**1. Evaluating AI usually requires committing to it first.**
Vendor trials are time limited, need your data uploaded, and rarely survive a
security review. There is no low risk way to find out whether retrieval over
your own documents is actually useful for your team. This repository is that
low risk way: nothing leaves the building, nothing is billed, and nothing needs
approval beyond permission to install Python.

**2. Every new AI project rebuilds the same plumbing.**
Document chunking, embeddings, a vector store, an agent loop, tool calling,
persistence, tracing. That is four to six weeks of undifferentiated work before
anyone sees a useful answer, repeated on every engagement. Here it is already
built, already tested, and already documented, so a project starts at week five
instead of week one.

**3. Prototypes that work are usually thrown away.**
The typical demo is a notebook with hardcoded paths that nobody can deploy. This
one is structured the way a production system is structured, with defined layer
boundaries, swappable backends, a test suite, and continuous integration. The
thing you show a customer is the thing you ship, with the backends changed.

---

## Quick start, 60 seconds

You need [Python 3.11 or newer](https://www.python.org/downloads/). Nothing else.
No Docker, no account, no API key, no internet connection after cloning.

```bash
git clone https://github.com/FlyguyTestRun/0-AI-Architecture-Stack.git
cd 0-AI-Architecture-Stack
make demo
```

You will see two sample company policy documents indexed, then four questions
answered with the source named for each one:

```
╭─ How much can a support agent refund without approval? ─────────────────────╮
│ Support agents may issue a refund up to two hundred dollars without         │
│ approval. The support lead may approve up to two thousand dollars.          │
╰─────────────────────────────────────────────────────────────────────────────╯
plan -> retrieve -> generate | sources: support-runbook.md | 3.26ms
```

That last line is the point. It tells you what the system did, which document
the answer came from, and how long it took. Nothing is a black box.

### Point it at your own documents

```bash
zerostack ingest ./my-company-docs      # index a folder of .md or .txt files
zerostack ask "What is our refund policy?"
```

### See what is running

```bash
make doctor
```

This prints a table of every layer and which backend is actually serving it.
When something is running in a reduced mode, it says so plainly rather than
failing quietly.

### Use it as a web app

```bash
make ui        # a browser interface at localhost:8501
make serve     # a REST API at localhost:8000
```

---

## What it actually does

In plain terms, the system reads your documents, finds the passages relevant to
a question, and writes an answer using only those passages, citing them.

```
   Your question
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │  1. Decide: does this need the document library? │
   └─────────────────────────────────────────────────┘
        │                              │
       yes                             no
        ▼                              │
   ┌─────────────────────────┐         │
   │  2. Find the relevant    │         │
   │     passages             │         │
   └─────────────────────────┘         │
        │                              │
        └──────────────┬───────────────┘
                       ▼
   ┌─────────────────────────────────────────────────┐
   │  3. Use a tool if the question needs one         │
   │     (arithmetic, a lookup, an outside system)    │
   └─────────────────────────────────────────────────┘
                       ▼
   ┌─────────────────────────────────────────────────┐
   │  4. Write the answer from those passages only    │
   └─────────────────────────────────────────────────┘
                       ▼
   Answer + the sources it came from + a record of every step
```

Every step is recorded. If an answer looks wrong, you can see exactly which
passages produced it, which is usually enough to tell whether the problem is the
question, the documents, or the system.

### The eight layers

| # | Layer | What it is, in plain terms | What ships |
|---|-------|---------------------------|------------|
| 1 | **Frontend** | How people talk to it | Browser app, REST API, command line |
| 2 | **Orchestrator** | The decision maker that runs the steps above | Three interchangeable engines |
| 3 | **Retrieval** | The searchable library of your documents | Keyword, vector and graph tiers, four store options |
| 4 | **Language model** | The part that writes the answer | Local model server, plus an offline fallback |
| 5 | **Tools** | How it reaches other systems | Open protocol adapter and a tool registry |
| 6 | **Code agent** | Conventions that let a coding assistant work here safely | Operating rules, six work modes, decision records |
| 7 | **Data** | What it remembers | Every question, answer, source and timing |
| 8 | **Deployment** | How it gets to a server | Containers, continuous integration |
|   | **Observability** | The flight recorder across all layers | Traces, metrics, cost accounting, caching |
|   | **Security** | Who may ask what, about which documents | API keys, roles, namespaces, rate limits |
|   | **Evaluation** | Proof that a change helped rather than hurt | Golden dataset and a merge gate |

Full detail in [ARCHITECTURE.md](ARCHITECTURE.md).

### Three ways to find an answer

Most systems offer one. Each fails somewhere the others do not, so this runs all
three and combines them.

| Tier | Finds | Fails at |
|------|-------|----------|
| **Keyword** | Exact terms: part numbers, error codes, policy names | Questions phrased in synonyms |
| **Vector** | Meaning, regardless of wording | Terms the model never saw, like `E-4471` |
| **Graph** | Facts spread across documents that never mention each other | Anything with no named entity |

The first two are fused by rank position. The third contributes only when it
recognises something in your question, so it costs nothing otherwise.

> **Why this matters in practice.** Ask an assistant about error code `E-4471`
> and a purely semantic system will confidently return something about payments
> in general. Ask "what connects our support lead to our engineering director"
> and no single passage anywhere contains the answer. Both are ordinary questions
> and both need a tier most systems do not have.

### What it records about itself

| Signal | Answers | Where |
|--------|---------|-------|
| Traces | What happened inside one request | `data/traces.jsonl` |
| Metrics | What is happening across all requests | `/metrics`, `/metrics.json` |
| Cost | Tokens and estimated spend, against a ceiling | `/costs` |
| Runs | Every question asked and what grounded the answer | `/runs` |
| Quality | Whether a change improved or broke answers | `zerostack evaluate` |

---

## Who this is for

### Small and midsize companies not on an enterprise plan

This is the group the project is built for. Under roughly two hundred employees,
without a dedicated machine learning team, and without the budget or the legal
appetite for an enterprise AI contract.

**What you get, concretely:**

| Need | How this addresses it | What it would otherwise cost |
|------|----------------------|------------------------------|
| Answer questions from internal policy, contracts, runbooks | Retrieval with citations, out of the box | $20 to $60 per seat per month |
| Keep documents in house | Everything runs on your hardware. No document ever leaves | Usually not available below enterprise tier |
| Know why the system said something | Every answer names its sources; every step is traced | Often an enterprise only feature |
| Try before committing | Runs free on a laptop, indefinitely | A paid pilot |
| Avoid lock in | Every layer is swappable behind an interface | Migration projects |

**Realistic first uses.** These are chosen because they are narrow, the documents
already exist, and success is easy to judge:

- **Internal policy desk.** HR and operations questions answered from the handbook,
  so the same six questions stop reaching a person every week.
- **Support runbook lookup.** Agents ask in plain English instead of searching a wiki
  during a live call.
- **Contract and SOW recall.** "What did we agree on payment terms with this client?"
  answered from the signed documents.
- **Onboarding.** New hires self serve on process questions during their first month.
- **Bid and proposal reuse.** Find what was written for a similar past proposal.

### Consultancies and agencies

Every client engagement needs the same foundation. Fork this, replace the sample
documents, change the backends, keep the tests. The architectural decisions are
already recorded in [docs/adr/](docs/adr/), so the same questions are not
relitigated on each project.

### Engineers learning the shape of these systems

The codebase is deliberately readable. Every layer defines an interface and ships
at least two implementations, so the seams are visible rather than theoretical.
Comments explain why a decision was made, not what a line does.

---

## Where this is the wrong tool

A baseline that oversells itself is worse than no baseline, so here is the honest
assessment. **Do not use this, as it stands, for:**

**Anything safety critical or regulated without additional work.** There is no
authentication and no rate limiting, and it is single tenant by construction.
Medical, legal, or financial advice reaching a customer needs review controls this
does not have. The ingest endpoint is restricted to a configured set of directories,
so it cannot be pointed at arbitrary files, but that is a boundary rather than a
substitute for authentication.

**Multi customer SaaS at scale.** Namespaces isolate tenants across every
retrieval tier, and the boundary is covered by tests. But the namespace filter is
applied after the search rather than pushed into the vector backend, so a scoped
query over fetches, and rate limiting is per process, so several replicas each
permit the configured rate. Both are fine for tens of tenants on one deployment
and want real work before hundreds across many.

**Very large document collections.** The defaults are tuned for thousands of
documents, not millions. Past roughly a hundred thousand chunks you want a managed
vector database, a reranking stage, and a hardware conversation.

**Questions needing reasoning across many documents at once.** Retrieval finds
passages that resemble the question. "Summarize every contract signed last quarter
and identify the outliers" is an analytical workload, not a retrieval one.

**Real time or high concurrency serving.** It has been verified correct under
concurrent load, but it has not been load tested for throughput. Treat published
latency figures as a local development signal only.

**A replacement for search.** If people know which document they need, a good
search box is cheaper, faster, and more predictable.

### Honest limitations of the current build

- The offline language model is **extractive**, not generative. It selects the
  sentences that answer the question rather than composing prose. It exists so the
  system runs and can be tested with no model server. Install a local model server
  for fluent answers.
- The offline embeddings are **lexical**. They match shared vocabulary, not meaning,
  so a question phrased entirely in synonyms will miss. The semantic option is one
  extra install.
- The multi agent engine is structurally complete but **unverified end to end**,
  because it needs a live model server. Treat it as unproven until exercised.
- Graph extraction is **deterministic, not model driven**. It finds that two
  things are related, not how they are related. That is the price of a tier that
  costs nothing and runs offline.
- Tool selection is **rule based**, which is deterministic and testable but does not
  generalize to open ended tool use.

Every one of these is tracked in [TODO.md](TODO.md) with the work needed to close it.

---

## For larger and regulated deployments

Every control below is **off by default**. A small team never configures them. An
organisation that needs them turns them on with configuration rather than a
migration, which is why there is one codebase rather than two editions
([ADR ZS-007](docs/adr/ZS-007-enterprise-controls-are-opt-in.md)).

**Authentication and roles.** API keys map to a principal with one of four
ordered roles. Keys are stored hashed, so a leaked configuration file does not
hand over working credentials, and compared in constant time.

| Role | May |
|------|-----|
| `reader` | Ask questions, list tools |
| `writer` | Everything above, plus ingest documents |
| `operator` | Everything above, plus metrics, costs and the run log |
| `admin` | Everything |

**Tenancy through namespaces.** Documents are ingested into a namespace and
queries are scoped to one. A principal is confined to the namespaces it is
granted. The boundary holds across all three retrieval tiers, including a
separate graph per namespace, so a traversal cannot walk from one tenant's
entity into another tenant's document.

**Rate limiting.** A token bucket per caller, enforced on every authenticated
endpoint rather than only the expensive ones.

**Spend ceilings.** Token and cost budgets checked before a call rather than
after, so the limit is a limit rather than a report of the overspend.

**Audit trail.** Every question is recorded with its answer, sources, timing and
trace id, including questions served from cache.

```bash
# One principal turns every control on at once.
ZEROSTACK_SECURITY_PRINCIPALS='{"zs_...":{"name":"hr-app","role":"writer","namespaces":["hr"],"requests_per_minute":60}}'
ZEROSTACK_SECURITY_REQUESTS_PER_MINUTE=120
ZEROSTACK_COST_DAILY_TOKEN_BUDGET=2000000
```

```bash
curl -X POST localhost:8000/ask \
  -H 'X-API-Key: zs_...' \
  -H 'content-type: application/json' \
  -d '{"question": "What is our refund policy?", "namespace": "hr"}'
```

---

## Knowing whether it actually works

Retrieval quality has no exception to catch when it breaks. It returns a
different passage, the answer gets worse, and every test still passes. So it is
measured rather than assumed.

```bash
zerostack evaluate                    # score against the golden dataset
make evals                            # the same, as a threshold gate
```

The harness scores retrieval and answer quality **separately**, because a system
can retrieve perfectly and still answer badly, and knowing which half broke is
the point. It runs in CI and fails the build on a regression
([ADR ZS-008](docs/adr/ZS-008-quality-is-measured.md)).

This is not decoration. On its first run it found two questions where retrieval
was perfect and the answer omitted the fact, which is how the extractive window
came to be six sentences rather than four: measured, not guessed.

---

## Growth path

The architecture is designed so that outgrowing the free tier is a configuration
change rather than a rewrite.

| Stage | Situation | What changes | Cost |
|-------|-----------|--------------|------|
| **Evaluate** | Does this help at all? | Nothing. Clone and run | $0 |
| **Pilot** | One team, real documents | Start the local model server for fluent answers | $0, one machine |
| **Deploy** | The company depends on it | Move the vector store to a server, add authentication | Hosting only |
| **Scale** | Multiple teams, large corpus | Namespaces per team, managed vector database, reranking | Usage based |
| **Govern** | Regulated data, audit requirements, role based access | Move to a governed enterprise platform | Enterprise |

The last row is where a purpose built platform earns its price. Compliance
classification, role based access control, per tenant isolation, and audit trails
are substantial engineering, and rebuilding them is not a good use of a small
team's time. This repository is deliberately the tier below that: it shares the
architectural vocabulary, so the concepts and the document corpus carry forward
rather than being thrown away.

> **Note for review:** if this repository is going to be referenced in customer
> facing launch material, the positioning above should be checked by whoever owns
> that messaging before it is used. It is written to be accurate about this
> codebase, not to make claims about any commercial product.

---

## Technical reference

Everything below is for engineers evaluating or extending the system.

### Design principles

**One protocol per layer, at least two implementations.** Every layer defines a
`Protocol` and ships a preferred backend plus an offline fallback. Callers depend on
the protocol, never a concrete class. This is what makes backends swappable in fact
rather than in aspiration.

**Strict dependency direction.** A layer may import from layers below it and never
from layers above:

```
frontend / api / cli
    -> app.ZerostackApp
        -> orchestrator -> {llm, rag, tools}
            -> config, observability
```

If a change requires the orchestrator to know the API exists, the design is wrong.

**Automatic degradation, explicit escalation.** `auto` selects the best backend that
is actually reachable and falls back when it is not. An explicitly named backend is
honored even when the service is missing, so a misconfigured production deployment
fails loudly instead of quietly serving worse answers. **Set backends explicitly in
production.**

**Composition in exactly one place.** `ZerostackApp` builds every layer. Nothing else
constructs a backend, which is what allows the frontend to be replaced without
touching anything beneath it.

### Backend matrix

| Layer | Preferred | Offline fallback | Selector |
|-------|-----------|------------------|----------|
| Language model | Local model server | Extractive provider over retrieved context | `ZEROSTACK_LLM_PROVIDER` |
| Vector store | Server mode | Embedded on disk, then in process, then pure Python | `ZEROSTACK_RAG_BACKEND` |
| Embeddings | Sentence transformers | Deterministic hashed bag of words | `ZEROSTACK_RAG_EMBEDDING_BACKEND` |
| Orchestrator | Graph engine | Sequential engine over the same nodes | `ZEROSTACK_ORCHESTRATOR_KIND` |
| Data | Managed Postgres | SQLite, with an analytical engine alongside | `ZEROSTACK_DATA_*` |

Each selector accepts `auto` plus explicit names. See [.env.example](.env.example)
for the full set.

### The orchestrator adapter

The agent graph is defined **once**, as pure functions over a state dictionary in
[`src/zerostack/orchestrator/nodes.py`](src/zerostack/orchestrator/nodes.py). Three
engines execute those same functions:

| Engine | Mechanism | Requires |
|--------|-----------|----------|
| `simple` | Sequential calls | Nothing |
| Graph engine | A state graph with a conditional edge | One extra install |
| Crew engine | Shared nodes, then a role based crew for reasoning | An extra install and a live model server |

A test asserts the graph engine and `simple` return **the same answer, the same
sources, and the same step sequence** for the same question, so engine drift becomes
a build failure rather than a surprise. That assertion runs in a dedicated CI job on
each engine.

To change agent behaviour, edit `nodes.py`. All engines inherit the change.

### Retrieval details

- **Chunking** is paragraph aware with character overlap. Paragraphs stay intact when
  they fit, which preserves more meaning than a fixed window at no extra cost. An
  oversized paragraph is windowed rather than dropped, and invalid parameters are
  rejected rather than silently losing text.
- **Relevance cutoff.** Top k alone always returns k chunks, including irrelevant
  ones when the index is small, which overstates what actually grounded the answer.
  Hits below a configurable fraction of the best score are dropped. The cutoff is
  relative because an absolute threshold is not portable across embedding backends.
- **Citations** are numbered and carried through to the answer, so every claim can be
  traced to a source document.

### Observability

A span is recorded for every retrieval, model call, tool invocation and agent run.
Spans append to a newline delimited JSON file so a developer can inspect a run with
no collector running, and export over OTLP to a tracing UI when enabled.

Trace state is **thread local**, because the API serves synchronous endpoints on a
worker thread pool. Holding it per process let simultaneous requests overwrite each
other's trace, which misattributed spans and stored the wrong trace identifier on
the run record. That is now covered by concurrency tests.

### Testing

**359 tests, no network, no containers, no model server.** That constraint is
deliberate: it keeps the suite fast and deterministic, and it means the offline path
is exercised on every commit and cannot rot.

```bash
make check     # lint, formatting, dash policy and tests, exactly what CI runs
make test      # tests only
make fmt       # apply formatting
```

CI runs the suite on Python 3.11 and 3.12, then again on each orchestrator engine,
and finishes by running the demo end to end.

### Extending it

| Goal | Where to change | Ripple |
|------|-----------------|--------|
| Different agent behaviour | `orchestrator/nodes.py` | All engines inherit it |
| New orchestration engine | Implement `Orchestrator`, register in `factory.py` | None |
| Hosted model API | Implement `LLMClient`, register in `llm/registry.py` | None |
| Different vector store | Implement `VectorStore` | None |
| New external integration | Add a server to `mcp.json` | None |
| Postgres instead of SQLite | `data/store.py` connection; the schema is portable | None |
| A different frontend | Consume the REST API | None |

### Project layout

```
src/zerostack/
├── app.py              Composition root, wires every layer
├── config.py           One settings section per layer
├── cli.py              Command line interface
├── api/                REST service
├── llm/                Layer 4: providers and selection
├── rag/                Layer 3: chunking, embeddings, stores, pipeline
├── orchestrator/       Layer 2: shared nodes, three engines, factory
├── tools/              Layer 5: registry, built in tools, protocol adapter
├── data/               Layer 7: state and analytics
├── observability/      Tracing, metrics, cost accounting, caching
├── security/           Identity, roles, namespaces, rate limiting
└── evals/              Quality measurement and the merge gate

apps/streamlit_app/     Layer 1 browser frontend
evals/                  Golden datasets
docs/adr/               Architecture decision records
modes/                  Operational mode definitions
tests/                  359 tests, no services required
```

---

## Configuration

Copy [.env.example](.env.example) to `.env`. Every value is optional; with no `.env`
at all the stack runs offline.

```bash
ZEROSTACK_LLM_PROVIDER=auto             # auto | local server | offline
ZEROSTACK_RAG_BACKEND=auto              # auto | server | embedded | memory
ZEROSTACK_ORCHESTRATOR_KIND=auto        # auto | graph | crew | simple
ZEROSTACK_RAG_RETRIEVAL_MODE=auto       # auto | hybrid | vector | keyword
ZEROSTACK_RAG_GRAPH_ENABLED=true        # the cross document tier
ZEROSTACK_CACHE_ENABLED=true            # semantic answer cache
```

### Optional capabilities

Each is additive and none is required for the demo:

```bash
pip install -e ".[embeddings]"      # semantic embeddings
pip install -e ".[langgraph]"       # graph orchestration engine
pip install -e ".[crewai]"          # role based crew engine
pip install -e ".[chroma]"          # alternative vector store
pip install -e ".[ui]"              # browser frontend
pip install -e ".[observability]"   # trace export
```

### External integrations

Copy `mcp.json.example` to `mcp.json` and declare servers using the conventional
[Model Context Protocol](https://modelcontextprotocol.io/) client configuration
shape. Discovered tools join the registry alongside the built in ones, and the
orchestrator cannot tell them apart. See
[ADR ZS-004](docs/adr/ZS-004-mcp-for-tools.md).

---

## Working in this repository

```bash
make setup     # virtualenv and install
make check     # everything CI runs
git config core.hooksPath .githooks   # enable the pre-commit and commit-msg checks
```

Operating rules are in [AGENTS.md](AGENTS.md) and are not optional. Two are enforced
mechanically by the pre-commit hook and by CI. Work happens in one of six modes
defined in [modes/](modes/), each setting how much to ask and how much to assume.

Decisions that should not be relitigated are recorded in [docs/adr/](docs/adr/).
If you find yourself reopening a settled question, read the record first.

---

## Documentation map

| Document | Audience | Purpose |
|----------|----------|---------|
| **README.md** | Everyone | This file: what it is, who it is for, how to run it |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engineers | Layer by layer design and extension points |
| [TODO.md](TODO.md) | Team | Work queue, known gaps, change log |
| [RUNBOOK.md](RUNBOOK.md) | Operators | Diagnosing and fixing a running deployment |
| [AGENTS.md](AGENTS.md) | Contributors | Operating rules and standards |
| [docs/adr/](docs/adr/) | Engineers | Why each significant decision was made |
| [modes/](modes/) | Contributors | How much autonomy each phase of work carries |

---

## License

[MIT](LICENSE). Use it, fork it, ship it commercially. No attribution required.
