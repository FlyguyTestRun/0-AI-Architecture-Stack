# TEST mode

Purpose: prove the system does what it claims.

Autonomy: high. Questions: few. Rules: tight.

## Behaviour

- Test behaviour, not implementation. A test that breaks on every refactor is a
  liability.
- Every bug fix starts with a failing test that reproduces it.
- Cover the failure paths, not just the happy path: missing files, unreachable
  services, malformed configuration, hostile input.
- Assertions must be able to fail. A tautological assertion is worse than no test,
  because it reports safety that does not exist.

## Constraints

- No test may require network, Docker or a model server.
- Tests must be deterministic. That is why the hashing embeddings and the extractive
  provider exist.

## Exit criteria

- `make check` is green and the new tests fail when the behaviour is reverted.
