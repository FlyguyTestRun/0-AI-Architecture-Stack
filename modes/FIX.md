# FIX mode

Purpose: diagnose and repair a specific defect.

Autonomy: medium. Questions: some. Rules: medium.

## Behaviour

1. Reproduce the defect first. A fix for an unreproduced bug is a guess.
2. Write a failing test that captures it.
3. Find the root cause. "Flake", "race" and "environment" are symptoms, not causes.
4. Fix the cause, not the symptom.
5. Keep the fix minimal. Do not widen the change while you are in there.
6. Record anything surprising as an ADR.

## Constraints

- Never skip, disable or delete a test to get green.
- If the root cause sits outside the current scope, say so with a proposed patch
  rather than expanding the change.

## Exit criteria

- The failing test passes, the whole suite passes, and the cause is understood.
