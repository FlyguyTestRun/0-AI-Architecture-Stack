# ZS-008: Retrieval quality is measured, not assumed

Status: Accepted
Date: 2026-08-25

## Context

Every layer in this system can be tested for correctness except the one that
decides whether answers are any good. Retrieval has no exception to catch when
it goes wrong. It returns a different passage, the answer gets worse, and the
whole test suite stays green.

That makes tuning guesswork and lets a quality regression merge unnoticed.

## Decision

A golden dataset of questions with known correct sources and required answer
phrases, three retrieval metrics, and a runner that gates a merge on thresholds.

Retrieval and answer quality are scored separately, because a system can retrieve
perfectly and still answer badly, and knowing which half broke is the point.

Answer checking is substring matching rather than model grading. A graded score
needs a model, which puts the harness behind a model server and makes results
nondeterministic, so it could not gate a merge. Blunt and reproducible beats
nuanced and unrepeatable for that job.

Evaluation runs with caching disabled and persistence off, so it measures the
configuration under test rather than the cache, and does not pollute the run log.

## Consequences

Good:

- Tuning becomes measurement. The extractive window moved from four sentences to
  six because the harness showed two questions retrieving the right document and
  then omitting the answering sentence, not because six felt better.
- A quality regression fails the build instead of merging quietly.
- Forbidden phrases pin a closed leak so it cannot return unnoticed.

Costs:

- The dataset is small and hand written, so it measures the cases somebody
  thought of. It is a regression gate, not a benchmark.
- Substring matching cannot tell a correct paraphrase from a wrong answer, so a
  genuinely better answer phrased differently is scored as a failure.
- Thresholds pinned at the current baseline make the gate a ratchet. Raising
  them is deliberate work.
