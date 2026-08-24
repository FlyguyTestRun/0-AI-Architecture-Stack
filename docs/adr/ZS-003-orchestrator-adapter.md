# ZS-003: Both orchestrators sit behind one adapter

Status: Accepted
Date: 2026-08-24

## Context

The reference architecture named LangGraph and CrewAI. They model agents differently:
LangGraph is a state graph, CrewAI is a crew of roles. Both are credible, and which
one fits depends on the project.

The usual way to support two frameworks is to write the agent twice. That guarantees
they drift, and it doubles the cost of every behaviour change.

## Decision

The agent logic lives once, as pure functions over a state dictionary, in
`orchestrator/nodes.py`. The three engines only decide how those functions are
executed:

- `simple` calls them in sequence. No dependencies.
- `langgraph` wires them as a `StateGraph` with a conditional edge for the
  "need external knowledge" decision.
- `crewai` runs plan, retrieve and tools through the same nodes, then delegates the
  reasoning step to a crew.

Because the nodes are shared, switching engines changes the execution strategy and
never the behaviour. A test asserts this directly: LangGraph and the simple engine
must produce the same answer, the same sources and the same step sequence for the
same question.

`auto` prefers LangGraph when installed and falls back to `simple`. `auto` never
selects CrewAI, because CrewAI requires a reachable model server and choosing it
silently would turn a working offline run into a failure.

## Consequences

Good:

- One place to change agent behaviour.
- Engine choice becomes a configuration value, not a rewrite.
- The parity test makes drift a build failure rather than a surprise.

Costs:

- The shared nodes are constrained to what all three engines can express. Engine
  specific features, LangGraph checkpointing for example, are not reachable through
  the shared path and need an engine specific extension.
- CrewAI's value is largely in multi agent delegation, and a single agent crew does
  not show that off. The adapter is the seam where a richer crew would be added.
