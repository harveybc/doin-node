# Migration and dashboard addendum

## Observed live boundary

At the read-only sample, all four participants exposed height 40 and byte-identical
serialized block bodies. This establishes copy equality, not proof-of-optimization
validity. The primary dashboard returned zero history rows because it looked for
`verified_performance` in each individual accepted transaction. The compact
archive mode stores the score in the work transaction and an archive reference
in its companion. The read-only archive proxy recovered 39 acceptance rows,
with only 5 numerical running-best improvements. Historical repeated and
regressing acceptances remain evidence and are not rewritten.

## Additional contract checks

Two new tests failed against the earlier repair: a pending accepted optimum
changed `_domain_best` before any block, and a stage-control message reinjected
the same champion into the optimizer. Both paths now leave network champion
state untouched until a block is committed and replayed. Replaying the same
tip is idempotent for migration. The local-optimizer callback no longer
reinserts the chain champion for each inferior local proposal.

The dashboard now joins accepted work/evidence transactions using the core
acceptance projection. It resolves archived metrics with integrity checks,
reports unavailable bodies explicitly, and exposes non-improving accepted
records as a visible contract alert. A synthetic compact acceptance pair with
an inferior second score is retained as a regression test.

The live optimizer configuration has population size 20 per participant;
the inspected durable checkpoint also contains 20 individuals. Cumulative
candidate evaluation counts are not population size.

## Verification and operation

Node suite: 623 passed, 29 warnings under the isolated 1 GiB/120 s CPU scope.
Core suite from the preceding repair: 322 passed. No production code was
replaced by this addendum. The new dashboard cannot be claimed live until the
node/core rollout is coordinated across all four participants. Remote shell
access to both worker hosts was unavailable during this attempt. Do not mix
old and repaired consensus versions, reset the chain, or relabel historical
blocks as valid. After rollout, verify identical full blocks, archive readback,
and a new improving block that migrates once per receiving optimizer.
