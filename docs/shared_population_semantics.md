# Shared Population Semantics

Status: normative for the current four-worker shared-population protocol.

These rules resolve the lease, quorum, fork-choice, and restart ambiguities
identified by `AUDIT-GS-EXEC-20260731-01`. Changing any rule requires a
versioned campaign configuration and regression tests before deployment.

## A1. Lease Renewal

Only an explicit heartbeat from the worker that owns a candidate renews its
lease. Polling or replicating a remote claim preserves the remote lease
timestamp; it never refreshes it. This prevents observation by another peer
from resurrecting an expired owner.

Implementation:

- `UnifiedNode._maintain_shared_claim_lease`
- `UnifiedNode._merge_polled_shared_claim`

Regression tests:

- `test_active_evaluation_renews_shared_claim_lease`
- `test_polled_shared_claim_preserves_remote_lease_age`

## A2. Claim Quorum

`shared_min_peers` is a fixed number of remote confirmations. It is not
recomputed as a majority of whichever workers happen to be live.

For the four-worker campaign, `shared_min_peers = 3`; therefore two live
workers cannot confirm a new claim. The intended behavior under that
partition is safety over liveness. A different quorum requires a new,
versioned campaign configuration rather than an implicit runtime downgrade.

Implementation:

- `UnifiedNode._claim_on_peers`
- `UnifiedNode._maintain_shared_claim_lease`

Regression tests:

- `test_shared_claim_lease_uses_configured_values`
- `test_phase_1_v2_fleet_initializes_before_full_compute_barrier`

## A3. Fork-Choice Tie Break

The preferred chain score order is:

1. greater finalized height;
2. greater chain height;
3. greater total improvement;
4. lower tip hash for an exact score tie.

`ChainScore.__lt__` in `doin-core` intentionally compares
`self.tip_hash > other.tip_hash`. The comparator is inverted only for the
hash field so that descending sort or maximum selection prefers the lower
hash. The call-site regression test is
`test_select_best_uses_lower_hash_for_an_exact_score_tie` in
`doin-core/tests/test_fork_choice.py`.

## A4. Restart and Barrier Re-entry

After a mid-generation process restart, a worker:

1. recovers the canonical population checkpoint and compatible completed
   results from the chain;
2. discards process-local candidate ownership and claims;
3. recomputes the exact generation fingerprint;
4. enters a fresh generation barrier against its peers;
5. claims no candidate until that barrier satisfies the fixed quorum.

Cached `join_ready` or stale local lineage state cannot bypass the fresh
barrier. Population compatibility is determined by generation and exact
generation fingerprint, not by generation number alone.

Implementation:

- `UnifiedNode._recover_shared_state_from_chain`
- `_shared_generation_fingerprint`
- `UnifiedNode._wait_for_shared_generation_peers`

Regression tests:

- `test_shared_results_survive_full_process_restart`
- `test_shared_generation_fingerprint_ignores_live_fitness_only`
- `test_ordered_shared_initialization_precedes_compute_peer_barrier`
