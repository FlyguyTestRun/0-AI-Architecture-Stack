# Customer Support Escalation Runbook

## Severity Levels

Support tickets are classified into four severity levels at intake. The severity
determines the response target and who must be notified.

Severity 1 means a complete outage affecting all customers or any confirmed data loss.
The response target is fifteen minutes, twenty four hours a day. The on call engineer
and the engineering director are paged immediately.

Severity 2 means a major feature is unavailable or degraded for a significant number of
customers, with no workaround. The response target is one hour during business hours
and two hours outside them.

Severity 3 means a feature is impaired but a workaround exists. The response target is
one business day.

Severity 4 covers questions, feature requests and cosmetic issues. The response target
is three business days.

## Escalation Path

A support agent who cannot resolve a ticket within the response target escalates to the
support lead. The support lead may escalate to engineering. Only the support lead or the
engineering director may raise a ticket to severity 1.

If a severity 1 incident lasts longer than four hours, the engineering director must
post a customer facing status update and continue updating it every hour until the
incident is resolved.

## Refund Authority

Support agents may issue a refund up to two hundred dollars without approval. The
support lead may approve up to two thousand dollars. Anything above two thousand
dollars requires finance approval.

Refunds tied to a severity 1 incident are pre approved up to one full billing cycle for
any affected customer, and no individual approval is needed.

## Postmortems

Every severity 1 and severity 2 incident requires a written postmortem within five
business days. The postmortem is blameless. It must include a timeline, the root cause,
the customer impact in both duration and number of accounts, and at least two action
items with named owners and due dates.
