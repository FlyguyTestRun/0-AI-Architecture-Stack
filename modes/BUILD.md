# BUILD mode

Purpose: ship features against settled requirements.

Autonomy: high. Questions: few. Rules: medium.

## Behaviour

- Make sensible default choices and keep moving. Ask only when the options lead to
  materially different work.
- Write the test alongside the code, not after.
- Keep each change scoped to one task in `TODO.md`.
- Update `TODO.md` as work completes, including anything discovered along the way.

## Constraints

- Never break the offline path. A new hard dependency belongs in an optional extra.
- Layer boundaries are not negotiable. See `CLAUDE.md`.
- `make check` passes before every commit.

## Exit criteria

- The feature works, is tested, and `make check` is green.
