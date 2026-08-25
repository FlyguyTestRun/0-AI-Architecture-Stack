# ZS-010: Boundaries fail closed and travel with the request

Status: Accepted
Date: 2026-08-25

## Context

An external review of the enterprise tier found three ways the tenancy boundary
could be crossed. Each was in code written specifically to draw that boundary,
which is the point worth recording: the layer that enforces a boundary is not
automatically the layer that has one.

**The namespace lived on a process wide object.** `AgentContext` carried a
`namespace` field that the application set before each run and the retrieval
nodes read during it. The API serves each request on a worker thread against one
application object, so the field was shared by every request in flight. A request
that set `hr` and then sat between the set and the read, which is what any real
model call does, retrieved inside whichever tenant set the field next.

Instrumented at the read, a request asking for `hr` performed retrieval against
`legal` once in eighty. That is not a rare event at production request rates, and
it is silent: the answer comes back well formed, citing the other tenant's
documents.

**A broken principal table opened the deployment.** `PrincipalStore.from_json`
returned an empty store when the JSON would not parse, and an empty store means
"no authentication configured", which resolves every caller to a local
administrator holding the wildcard grant. A typo in the table, a secret that
failed to mount, or an unreadable file therefore turned an authenticated
deployment into an open one, in exactly the situations where nobody is watching.

**The run log was not scoped.** `RunRecord` stored no namespace and `/runs`
filtered on nothing, so an operator restricted to one namespace could read every
tenant's questions, answers and retrieved document text through the operational
endpoints. The tenancy boundary was drawn across retrieval and not across the
log of what retrieval returned.

## Decision

**Per request data travels with the request.** The namespace is part of the
orchestrator state, threaded through `run(question, namespace=...)` into
`initial_state`, and read by nodes from `state["namespace"]`. The field on
`AgentContext` is removed rather than deprecated, so the hazard cannot be
reintroduced by a future caller setting it again. The context holds only what is
genuinely process wide: the layers themselves.

A thread local was considered and rejected. It fixes the race but depends on the
node running on the thread that set the value, which is an implementation detail
of whichever orchestrator engine is selected. If that ever stopped holding, the
failure would be a silent read of the default namespace: a wrong tenant that
looks plausible, which is the same class of failure as the bug being fixed.

**A boundary that cannot be evaluated refuses.** A principal table that was
supplied and could not be used is a distinct state from one that was never
supplied, and it resolves to no access rather than to open access. Naming a
principals file is itself the intent to authenticate, so a missing or unreadable
one fails closed too. `/health` reports the state plainly instead of implying it.

**Tenancy applies to operational data.** Runs carry the namespace they were asked
in, and the run log and the analytics aggregate are filtered by the caller's
namespaces. A wildcard grant passes `None` for unrestricted; a principal with no
namespaces restricts to nothing rather than to everything, because the empty case
must fail in the safe direction.

## Consequences

Good:

- Concurrent requests cannot observe each other's tenancy. The same probe that
  found the leak reports zero mismatches in a hundred and twenty requests.
- A configuration mistake now closes the door rather than opening it.
- The operational endpoints stop being a way around the retrieval boundary.

Bad, and accepted:

- `Orchestrator.run` takes a second argument, so the protocol and all three
  engines changed. That is the cost of making the boundary explicit, and an
  explicit argument is what makes the next engine correct by construction.
- A deployment whose principal table was quietly broken will now fail to start
  serving rather than serving openly. That is the intended behaviour and it will
  look like a regression to anyone who was unknowingly relying on the old one.
- Runs recorded before this change have no namespace and take the default, so
  they are visible only to a caller scoped to the default namespace or holding
  the wildcard. Backfilling is not possible: the information was never recorded.

## Notes

The general rule: state shared by concurrent requests must be state that is the
same for all of them. Anything that varies per request belongs in the request,
and a boundary whose inputs cannot be evaluated denies rather than permits.
