# ZS-001: Every layer ships an offline fallback

Status: Accepted
Date: 2026-08-24

## Context

The stack targets small and midsize businesses. A pilot usually starts on one
person's laptop, often before anyone has approved installing Docker, and sometimes on
a machine that cannot pull a multi gigabyte model. A baseline that only works after
three services are running is a baseline that never gets evaluated.

There is also a testing problem. If the test suite needs Qdrant, Ollama and a network
connection, CI is slow, flaky, and the offline path silently rots.

## Decision

Every layer ships at least one implementation that runs with no external service, no
network and no native dependencies:

| Layer | Preferred | Offline fallback |
|-------|-----------|------------------|
| LLM | Ollama | Extractive provider over retrieved context |
| Vector store | Qdrant server | Embedded Qdrant on disk, then in process, then pure Python |
| Embeddings | sentence-transformers | Hashed bag of words, deterministic |
| Orchestrator | LangGraph | Sequential engine over the same nodes |
| Data | Supabase | SQLite, with DuckDB for analytics |
| Analytics | DuckDB sqlite scanner | Plain SQLite aggregate |

Selection is per layer and automatic. `auto` prefers the real backend and degrades
when it is unreachable. An explicit backend name is honoured even when the service is
missing, so a misconfigured production deployment fails loudly instead of quietly
serving degraded answers.

## Consequences

Good:

- `git clone`, `make demo`, working output. No infrastructure step.
- CI needs no service containers and runs in seconds.
- The fallback path is exercised on every commit, so it cannot rot.
- `zerostack doctor` makes the active backend visible, so degraded mode is never a
  mystery.

Costs:

- More code. Each layer carries a second implementation.
- The offline LLM is extractive, not generative. It answers correctly when retrieval
  worked and says so plainly when it did not. It is a development and testing aid,
  not a production answer engine.
- Contributors must remember that a new hard dependency breaks the guarantee. This is
  called out in `CLAUDE.md` and enforced by CI running with no services.
