# ZS-009: Metered state is bounded and windowed

Status: Accepted
Date: 2026-08-25

## Context

The enterprise tier added four pieces of state that grow with traffic: a metric
series per namespace, a rate limit bucket per caller, a running spend total, and
a cached answer per question. Each was correct on the path it was written for.
Reviewed as a group against a deployment that runs for months rather than a
demo that runs for seconds, each had a failure that only appears with time or
volume.

The failures were found by probing rather than by reading. Three of the four
looked right on the page.

- The namespace reached a metric label without ever being validated. A caller
  holding the wildcard grant, which is the default on the single machine path,
  could name a new namespace on every request and mint a metric series for each.
  Five thousand namespaces produced five thousand series and a quarter megabyte
  of exposition output, and nothing ever evicted them.
- The rate limiter keyed its buckets on the principal's name. The name is an
  operator chosen label and nothing stops two credentials sharing one, so two
  callers configured for four requests a minute each received four between them.
- The budget was called daily and had no window. The running total only ever
  rose, so the first day that spent the allowance refused every request from
  then until the process restarted.
- Ingestion invalidated the answer cache by passing every source in the store,
  which drops every cached answer on every ingestion. The code documented the
  opposite.

## Decision

State that is metered per caller, per tenant or per period must be bounded in
the dimension it grows along, and the bound must be enforced at the boundary
where the untrusted value enters.

Concretely:

- A namespace is an identifier matching `[a-z0-9][a-z0-9_-]{0,62}`, folded to
  lower case, validated at the API and again in the application object. Case is
  folded because two partitions differing only in case read as one tenant to a
  human. The wildcard is rejected as a request target: it is a grant a principal
  holds, never a namespace documents live in.
- Anything that meters a caller keys on the credential, through
  `Principal.caller_id()`, never on the display name.
- A budget named for a period resets on that period. The counters roll at the
  UTC day boundary, behind the lock, because the rollover is a check followed by
  a write and two threads at midnight would otherwise race.
- Invalidation is driven by what a run actually wrote. `IngestReport` carries the
  sources it touched, and the cache is invalidated against those.

## Consequences

Good:

- The metric registry, the bucket table and the cache are all bounded by
  configuration rather than by caller behaviour.
- A daily budget is now a throttle that reopens rather than an outage that
  arrives on a schedule.
- Ingesting one document no longer discards every unrelated cached answer, so
  caching survives a deployment that ingests on a schedule.
- Bucket eviction moved off the hot path. Admitting one request no longer costs
  a scan of the whole caller table: at twenty thousand callers that scan was
  most of the time spent in the limiter and it grew without bound.

Bad, and accepted:

- Namespaces are now constrained, so a deployment that used mixed case or
  punctuation must rename. Nothing in the repository did, and the constraint is
  cheaper to adopt now than after data exists under the old spelling.
- The day window is UTC rather than the operator's local midnight. A per
  deployment timezone is configuration this baseline does not yet need, and a
  fixed boundary is easier to reason about than a floating one.
- Cache invalidation stays conservative: an answer is dropped if any source it
  cited was rewritten, even when the rewrite did not touch the cited passage.
  Serving a stale answer is the worse failure.

## Addendum, same day

An external review raised a fourth case of the same shape on the spend ceiling
itself. The pre call check projected only the question's tokens and a cost of
zero, so a short question passed on a nearly spent budget and then produced a
full response, overshooting before anything was recorded. A ceiling checked
against an input that excludes the largest term is not a ceiling.

The projection now bounds the completion at the configured `max_tokens` and
prices it against the model that will actually serve, which is not always the
configured one: a degraded layer runs its fallback, and pricing the configured
model would charge against something that is not running. The prompt side
remains a lower bound, because retrieved context is not known at the time of the
check. This narrows the overshoot to the retrieved passages rather than removing
it, and that is stated here so the remaining gap is not mistaken for closed.

## Notes

The general rule this encodes: any value that a caller controls and the system
retains is a growth dimension, and it needs a bound chosen deliberately rather
than one inherited from whatever the caller happens to send.
