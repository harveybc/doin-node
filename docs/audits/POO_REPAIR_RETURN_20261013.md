# Proof-of-optimization repair return

## State

IMPLEMENTED_AND_TESTED_IN_ISOLATION. NOT_DEPLOYED.

Core pin: `2c42883` on `codex/poo-champion-validation-20261013`.
Node work: `codex/poo-contract-audit-20261013`, following audit `e0760ac`.
Both are local commits; no remote publication or live rollout is claimed.
No live service, private key, campaign dataset, TEST split, historical chain or
plot was changed. No fitting, GPU work or experiment repetition was performed.

## Result

The six original contract tests changed from five failures/one pass to six
passes. Final whole-node suite: **620 passed, 26 warnings, 20.16 seconds**.
Final whole-core suite: **322 passed, 0.74 seconds**. CPU verification was
bounded by 1 GiB RAM and 120 seconds per command. The warning count is retained;
this is not a claim of a warning-free, lint-clean or type-clean codebase.

The first whole-node run exposed an existing version-list assertion excluding
the predictor package, reproduced on the baseline. Its fixture now reflects
the package already present in the dashboard; no dashboard code was changed.
The archive integration tests require the data-lake package, supplied on
PYTHONPATH rather than changing any shared installed environment. Multi-node
test discovery is disabled so localhost fixtures do not scan the LAN.

## Traceability

| Requirement | Implementation | Retained evidence |
| --- | --- | --- |
| R1 Originator-only work | Reveal/quorum paths do not enqueue remote mintable acceptance | `test_network_champion_contract_audit.py`, four-node test |
| R2 Strictly improving committed score | Core `advance_champions`, node append/import/preflight | `test_champion_transition.py`, regression import test |
| R3 No repeated optimum credit | Domain/optimum identity guard, paired fork-work deduplication | Core transition/fork tests |
| R4 Body integrity | JSON/SQLite append, read, save and load validate accepted transaction IDs | `test_acceptance_body_integrity.py` |
| R5 Reference recovery | Integrity-checked archive projection; small consensus fields retained on chain | `test_archive_champion_projection.py` (61 tests) |
| R6 Restart/reorg recovery | Unconditional startup recovery; cache replacement, not stale best retention | `test_archive_champion_node_recovery.py` (16 tests) |
| R7 Validate before rollback | Complete peer-branch preflight; work-based fork choice for longer and equal forks | Fork rejection and adoption tests |
| R8 Identical adoption | Receivers append the originator's exact serialized block | Four-node synthetic sequence and real HTTP sync test |

The pure core module does not access archives, storage, networks or models.
It joins historical work/evidence transaction pairs without treating them as
two improvements. Its state inputs are the actual parent branch, not a local
optimizer's uncommitted best.

## Additional Corrections

- Verified zero is preserved instead of being replaced by the reported score.
- Subsequent work uses directional performance improvement, not absolute
  fitness. Legacy first-observation bootstrap work remains unchanged. Existing
  incentive factors and dynamic threshold remain in use; this is not a change
  to predictor fitness or its stopping criteria.
- Superseded pending acceptance work is removed. A still-better pending
  proposal is rebased against an adopted parent without redoing a model fit;
  only uncommitted transaction IDs are recomputed.
- A shared-population checkpoint is `DOMAIN_UPDATED` metadata, not an accepted
  optimum and not a source of artificial optimization credit.
- Champion-response and stage-control hints no longer overwrite the committed
  champion merely because a peer reports a better number.
- Fork scoring counts paired acceptance work once and rejects conflicting
  duplicate work. Legacy reference records recover work from their committed
  companion for scoring only, without editing historical block bytes.
- Destructive transaction pruning is disabled explicitly. Current snapshots
  cannot replace the transaction bodies needed for Merkle reconstruction and
  champion replay. Hash-only candidate storage still reduces payload size.

An independent review identified the shared-population, pruning and duplicate
fork-credit cases; all three received corrections and regression tests before
the final whole-suite run.

## Reproduction

Use an isolated environment with matching node/core sources and the existing
plugin and predictor archive-schema dependencies. DATA_LAKE_ROOT supplies the
test-only service adapter package; these tests create disposable stores.

```bash
# In the core checkout pinned above:
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 \
  crispdm-run -m 1G -t 120s -n poo-core-check -- python -m pytest -q tests

# In the node repair checkout:
CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$PWD/src:$DOIN_CORE_SRC:$DATA_LAKE_ROOT" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  crispdm-run -m 1G -t 120s -n poo-node-check -- \
  python -m pytest -q tests --tb=short --disable-warnings
```

## Deployment Boundary

These are coordinated consensus repairs, not a rolling dashboard patch.
Do not mix the repaired core/node with old participants. Existing repository
instructions prohibit touching fleet workers during verification; no live
restart or migration was performed.

Before rollout:

1. Retain the existing identical chain as historical evidence. Do not delete,
   rewrite, filter or silently relabel its duplicate/regressing blocks.
2. Check availability and integrity of archived candidate bodies on the target
   environment, using its configured archive adapter without disclosing tokens.
3. Verify that required historical transaction bodies are present. This patch
   cannot reconstruct bodies already deleted by an earlier pruning operation.
4. Pin repaired node and core identically for all participants. Preserve
   predictor/plugin, domain, seed, config, evaluation and archive identities.
5. Establish the explicit activation boundary for stricter validation. Recovery
   can read structurally valid old records, but that does not retroactively
   certify their proof-of-optimization semantics. No new genesis or fabricated
   historical checkpoint was created by this work.
6. After coordinated activation, obtain stable snapshots from all four nodes:
   network ID, component revisions, every block hash/body, height and champion.
   Verify the next genuine accepted optimum is committed once and adopted
   unchanged by every peer, including after a reconnect/restart.

The HTTP test proves exact block adoption between disposable nodes. It does
not substitute for step 6 on the live fleet. Trusted mode still skips peer
model retraining; this repair does not claim adversarial quorum/model-replay
certification or bitwise GPU training guarantees.
