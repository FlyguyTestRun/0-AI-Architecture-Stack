# Zerostack

**A zero cost AI architecture baseline for small and midsize business applications.**

Eight layers, every one of them self hostable and free, wired together into a working
reference application. Clone it, run one command, and you have a grounded question
answering system with retrieval, tool use, persistence and tracing. Then replace the
pieces you need to replace.

This is a starting point for client and product work, not a demo. The point is that
the boring decisions are already made, tested and documented, so a new project starts
at day thirty instead of day one.

---

## Quick start

The only prerequisite is Python 3.11 or newer. No Docker. No API key. No model
download.

```bash
git clone https://github.com/FlyguyTestRun/0-AI-Architecture-Stack.git
cd 0-AI-Architecture-Stack
make demo
```

You will see the sample corpus indexed and four questions answered with citations.

```
╭─ How much can a support agent refund without approval? ─────────────────────╮
│ Support agents may issue a refund up to two hundred dollars without         │
│ approval. The support lead may approve up to two thousand dollars.          │
╰─────────────────────────────────────────────────────────────────────────────╯
plan -> retrieve -> generate | sources: support-runbook.md | 3.26ms
```

To see which backend served each layer:

```bash
make doctor
```

### Adding the real backends

```bash
make up            # Qdrant, Ollama and Phoenix in Docker
make pull-model    # pull gemma3:4b into Ollama
make doctor        # confirm the stack picked them up
```

Nothing else changes. Each layer detects its service at startup and uses it.

---

## Why it runs with nothing installed

Every layer ships a preferred backend and an offline fallback, selected automatically.

| Layer | Preferred | Offline fallback |
|-------|-----------|------------------|
| LLM | Ollama | Extractive answering over retrieved context |
| Vector store | Qdrant server | Embedded Qdrant on disk, then in process |
| Embeddings | sentence-transformers | Deterministic hashed bag of words |
| Orchestrator | LangGraph | Sequential engine over the same nodes |
| Data | Supabase | SQLite, with DuckDB for analytics |

This is not a convenience. It is what makes the test suite fast and deterministic,
what lets a colleague evaluate the stack before requesting a Docker install, and what
keeps the fallback path from rotting, because CI runs on it every commit.

The offline LLM is honest about what it is: it extracts the sentences from retrieved
context that answer the question, and says so plainly when the context does not. It
is a development aid, not a production answer engine. Start Ollama for generated
answers.

See [ADR ZS-001](docs/adr/ZS-001-offline-fallback.md) for the reasoning.

---

## The eight layers

```
                    ┌──────────────────────────────────────┐
   User input  ───▶ │  1  Frontend: Streamlit, FastAPI, CLI │
                    └───────────────────┬──────────────────┘
                                        ▼
                    ┌──────────────────────────────────────┐
                    │  2  Orchestrator: LangGraph, CrewAI   │
                    └───────────────────┬──────────────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Need external          │
                            │ knowledge?             │
                            └────┬─────────────┬────┘
                             yes │             │ no
                                 ▼             │
              ┌────────────────────────────┐   │
              │  3  RAG: Qdrant, Chroma     │   │
              │     chunking + embeddings   │   │
              └──────────────┬─────────────┘   │
                             └────────┬────────┘
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  5  Tools via MCP  ──▶  4  LLM layer  │
                    │     GitHub, Slack, DB     Ollama      │
                    └───────────────────┬──────────────────┘
                                        ▼
                    ┌──────────────────────────────────────┐
                    │  7  Data: SQLite, DuckDB, Supabase    │
                    └──────────────────────────────────────┘

   Cross cutting:  Observability (Phoenix, JSONL span log)
                   6  Code agent: Claude Code CLI, Aider
                   8  Deployment: Docker, Cloudflare Workers, Hugging Face
```

| # | Layer | What ships | Swap in |
|---|-------|-----------|---------|
| 1 | Frontend | Streamlit app, FastAPI service, Typer CLI | Next.js on Vercel |
| 2 | Orchestrator | LangGraph, CrewAI, simple, behind one adapter | Any engine over the shared nodes |
| 3 | RAG | Qdrant and Chroma, paragraph chunking, two embedding backends | Notion or Drive as a source |
| 4 | LLM | Ollama, offline extractive provider | Any OpenAI compatible endpoint |
| 5 | Tools | Registry plus MCP discovery from `mcp.json` | Any MCP server |
| 6 | Code agent | Conventions in `CLAUDE.md` and `modes/` | Aider, Claude Code |
| 7 | Data | SQLite state, DuckDB analytics | Supabase Postgres |
| 8 | Deployment | Docker Compose, GitHub Actions CI | Cloudflare Workers, Hugging Face Spaces |

Full detail in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Using it

### CLI

```bash
zerostack doctor                    # which backend is live in every layer
zerostack ingest ./my-documents     # index a file or directory
zerostack ask "What is our refund policy?" --steps
zerostack runs                      # recent runs from the data layer
zerostack serve                     # FastAPI on :8000
```

### API

```bash
make serve
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What must a postmortem contain?"}'
```

Endpoints: `/ask`, `/ingest`, `/health`, `/runs`, `/analytics`, `/tools`.
Interactive docs at `http://localhost:8000/docs`.

### Streamlit

```bash
make ui
```

The UI shows the answer, the sources, the agent steps and the raw trace spans, because
for an SMB pilot that visibility is what earns trust in the output.

### Python

```python
from zerostack.app import ZerostackApp

app = ZerostackApp()
app.ingest("./company-docs")

result = app.ask("How much can a support agent refund without approval?")
print(result.answer)
print(result.sources)
```

---

## Configuration

Copy `.env.example` to `.env`. Every value is optional. With no `.env` at all the
stack runs offline.

Each layer has a selector accepting `auto` plus explicit backend names:

```bash
ZEROSTACK_LLM_PROVIDER=auto             # auto | ollama | echo
ZEROSTACK_RAG_BACKEND=auto              # auto | qdrant | chroma | memory
ZEROSTACK_ORCHESTRATOR_KIND=auto        # auto | langgraph | crewai | simple
```

`auto` degrades to a fallback when a service is unreachable. An explicit name is
honoured even when the service is missing, so a misconfigured production deployment
fails loudly instead of quietly serving degraded answers. **Set these explicitly in
production.**

### Optional extras

```bash
pip install -e ".[langgraph]"       # LangGraph orchestrator
pip install -e ".[crewai]"          # CrewAI orchestrator
pip install -e ".[embeddings]"      # sentence-transformers
pip install -e ".[chroma]"          # Chroma vector store
pip install -e ".[ui]"              # Streamlit
pip install -e ".[observability]"   # OpenTelemetry export to Phoenix
```

### MCP tools

Copy `mcp.json.example` to `mcp.json` and declare servers using the same format as
Claude Code and Claude Desktop. Discovered tools join the registry alongside the built
in ones, and the orchestrator cannot tell them apart. See
[ADR ZS-004](docs/adr/ZS-004-mcp-for-tools.md).

---

## Development

```bash
make setup     # virtualenv and install
make check     # lint, dash policy and tests, what CI runs
make fmt       # apply formatting
make test      # tests only
```

Enable the pre-commit hook:

```bash
git config core.hooksPath .githooks
```

This repository follows the operating rules in [CLAUDE.md](CLAUDE.md). Two are
enforced mechanically: no AI attribution in committed content, and no em dashes.
`scripts/check_emdash.py` runs in the hook and in CI.

Work happens in one of six modes defined in [modes/](modes/): SCOPE, BUILD, SECURE,
TEST, FIX, DEPLOY. Each sets how much to ask and how much to assume.

---

## Project layout

```
src/zerostack/
├── app.py              Composition root, wires every layer
├── config.py           One settings section per layer
├── cli.py              Typer CLI
├── api/                FastAPI service
├── llm/                Layer 4: Ollama, offline provider, registry
├── rag/                Layer 3: chunking, embeddings, stores, pipeline
├── orchestrator/       Layer 2: shared nodes, three engines, factory
├── tools/              Layer 5: registry, built in tools, MCP adapter
├── data/               Layer 7: SQLite state, DuckDB analytics
└── observability/      Span tracing, JSONL and OTLP

apps/streamlit_app/     Layer 1 frontend
docs/adr/               Architecture decision records
modes/                  Operational mode definitions
tests/                  142 tests, no services required
```

---

## What this is not

- Not a production system. It is a baseline that production systems start from.
- Not a benchmark. The offline provider is extractive, and answer quality with a real
  model depends entirely on the model.
- Not multi tenant. Add authentication and tenant isolation before serving more than
  one customer from one deployment.

## License

MIT
