# ZS-007: Enterprise controls are opt in, not a separate edition

Status: Accepted
Date: 2026-08-25

## Context

This repository serves two audiences whose requirements look opposed. A small
team wants to clone it and ask a question within a minute, with no accounts and
no configuration. An organisation with regulated data needs authentication,
tenant isolation, role separation, rate limiting and an audit trail before it can
be used at all.

The usual answers are both bad. Maintaining two editions doubles the work and
guarantees they drift. Requiring the enterprise controls always makes the thirty
second start impossible, which is the project's main advantage.

## Decision

One codebase, with every control opt in and off by default.

With no principals configured, the API authenticates every caller as a local
administrator over all namespaces and behaves exactly as it did before the
controls existed. Configuring a single principal turns authentication, role
enforcement, namespace scoping and rate limiting on together.

Namespaces are the tenancy primitive, matching the model used by the enterprise
tier of this architecture, so a corpus organised here transfers upward rather
than being reorganised at the boundary.

Roles are ordered rather than a matrix, so a permission check is a comparison.
Namespaces are kept separate from roles, because a writer who may ingest still
must not read another department's corpus, and collapsing the two makes that
combination inexpressible.

## Consequences

Good:

- One codebase, one test suite, no drift between editions.
- Growing into the controls is configuration, not migration.
- The open path is covered by tests, so it cannot regress while attention is on
  the enterprise path.

Costs:

- The default is open, which is right for a laptop and wrong for anything
  exposed. `zerostack doctor` and the health endpoint state which mode is active
  rather than leaving it implied, but a careless deployment can still expose an
  unauthenticated service.
- Namespace filtering is applied after the search rather than pushed into each
  vector backend, so a scoped query over fetches. That is correct but not
  efficient, and a large multi tenant deployment should push the filter down.
- Rate limiting is per process. Several replicas each permit the configured rate,
  so a shared limiter is needed before horizontal scaling.
