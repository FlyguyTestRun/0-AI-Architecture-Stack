# DEPLOY mode

Purpose: package and release.

Autonomy: low. Questions: many. Rules: strict.

## Behaviour

- Never deploy without explicit approval.
- Confirm the target, the rollback plan and who is watching before starting.
- Update `RUNBOOK.md` with anything learned.

## Pre-flight checklist

- [ ] `make check` green on the release commit
- [ ] `zerostack doctor` shows the intended backends, not fallbacks
- [ ] Backend selectors set explicitly, never left on `auto` in production, so a
      missing service fails loudly instead of degrading silently
- [ ] Secrets present in the target environment and absent from the repository
- [ ] Vector collection populated and its dimensions matched to the embedding model
- [ ] Rollback procedure written and tested

## Exit criteria

- The deployment is verified and `RUNBOOK.md` reflects reality.
