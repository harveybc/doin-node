# Imported Work Validation

The independent rollout review reproduced an empty non-genesis acceptance
and an accepted tiny improvement carrying an effective increment of 1e12.
The latter could bias fork scoring even though champion ordering was strict.

Every non-genesis block now requires an accepted improving optimum. Raw work,
when retained, must equal the directional gain over that block's actual parent.
Effective work and the header's weighted sum cannot exceed that gain times
the domain's configured incentive bonus and weight. Existing VUW/reputation
discounts, incentive configuration, bootstrap convention and thresholds remain.
The bound prevents inflated work; it does not independently reconstruct each
peer's reputation history or claim adversarial verification of model fits.

Validation runs before local append, peer append and fork rollback. An existing
body-integrity fixture used fictitious work=10 for an improvement=1. It now
uses work=1.1, within the existing incentive cap, so it still exercises the
intended invalid-body rejection rather than the new earlier work rejection.

CPU evidence: core 328 passed; node 626 passed, 29 warnings. Fresh production
identities require the reviewed node/core pins together. The old fleet is
drained separately and its history is retained without rewriting any block.
