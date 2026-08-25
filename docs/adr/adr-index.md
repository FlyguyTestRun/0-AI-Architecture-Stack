# Architecture decision record index

ADRs record decisions that should not be relitigated without new information. Add one
whenever a choice is made that a future session would otherwise reopen.

Prefixes:

- `GLOBAL` applies across every repository in this organisation.
- `ZS` applies to this repository only.

| ID | Title | Status |
|----|-------|--------|
| GLOBAL-001 | No AI attribution in committed content | Accepted |
| GLOBAL-002 | Approval before committing | Accepted |
| GLOBAL-003 | No em dashes in repository files | Accepted |
| ZS-001 | Every layer ships an offline fallback | Accepted |
| ZS-002 | Qdrant is the default vector store | Accepted |
| ZS-003 | Both orchestrators sit behind one adapter | Accepted |
| ZS-004 | Tool access goes through MCP | Accepted |
| ZS-005 | Retrieval is hybrid by default | Accepted |
| ZS-006 | A graph tier for questions spread across documents | Accepted |
| ZS-007 | Enterprise controls are opt in, not a separate edition | Accepted |
| ZS-008 | Retrieval quality is measured, not assumed | Accepted |
