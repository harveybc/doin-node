# Network champion contract audit

## Verdict

FAIL: identical replicated chains do not establish correct proof-of-optimization
acceptance. The deployed implementation violates the requested single-originator,
strictly improving network-champion contract. No production code, services,
chain, candidate archive, training or dashboard was changed by this audit.

This document concerns one configured minimizing domain. It does not redefine
the multi-domain consensus protocol or propose rewriting historical blocks.

## Observed fleet

Read-only HTTP snapshot: 2026-10-14T00:56:33Z (2026-10-13 19:56:33 UTC-05).
Four endpoints returned identical full parsed block arrays, stable across each
endpoint's before/after height and tip checks. These are observations at that
snapshot, not a claim of perpetual simultaneous delivery over a network.

- Height: 37 on all nodes, including genesis.
- Tip: `68eb0c11c4c917fbdd31ea8c4481848cca06c2f0416c565e99ccb74ca6ee7ab7`.
- Canonical JSON block-array digest:
  `3e437025c4b468ea1398ac5197343ef32cbf47abc9f90ddfef670b3f8067784c`.
- In-memory best performance: `-0.014065407128813545` on all four.
- Identical reported revisions: node `7942197`, core `8e7bd82`, plugins
  `2fc0523`, predictor `cbbe1f1`.
- 36 accepted-optimum records, 20 distinct optimum IDs, 16 repeated inclusions.
- 18 blocks have a generator different from the accepted optimum's discoverer.
- 32 blocks do not establish a new best. Four best observations include the
  first observed score, not four improvements beyond an existing initial best.
- No header/link/Merkle or acceptance transaction-body digest mismatch was
  found in this actual snapshot. The mutation vulnerability below was synthetic.

Examples independently read from the chain, not inferred from plot colors:

| Blocks | Observation |
| --- | --- |
| 3, 4, 5 | Same optimum ID included three times; different block generators |
| 6 | Accepted performance `-0.010586766321054819` |
| 7 | Accepted performance `-0.007545515147333703`, worse than block 6 |
| 33 through 36 | Repeated optimum with performance `-0.014065407128813545` |

Full anonymized snapshot analysis: `network-champion-live-report.json` beside
this document. Endpoint addresses are deliberately not retained here.

## Findings

### P1: External references broke recovery and fork-work projection

Node commit `2a3e725654945ae63afcd45248f696b53c6b606f` replaced inline
acceptance payloads with an archive reference without updating their consumers.
`src/doin_node/unified.py:1259` writes the reference; recovery at line 1367
still requires top-level `verified_performance` and `parameters`. It does not
resolve the referenced body. The companion transaction's `performance` alone
cannot restore the genotype. Recovery can leave the champion empty or stale.
Startup also does not unconditionally recover the local committed champion;
the recovery calls are in peer synchronization paths.

The replacement drops `effective_increment` in both trusted and quorum paths
(lines 1841 and 2268). Core `consensus/fork_choice.py:64` scores that field,
not the companion's `increment` or header work. An independent bounded CPU
reproduction compared the predecessor, introducing commit and live revision:
the predecessor recovered score 0.9 and fork work 0.9; both newer revisions
recovered no champion and fork work 0.0, despite header work 0.9 and resolvable
archive bytes. This independent probe was ephemeral, not a retained test file.

This is a demonstrated recent regression. It does not establish the cause of
every repeated historical block by itself.

### P1: Receivers mint their own blocks for another node's optimum

Trusted reveal acceptance calls `try_generate_block()` unconditionally at
`unified.py:1900`; generation at line 2481 uses the receiver's peer identity.
Two temporary nodes given the same commitment/reveal create different blocks.
This behavior predates the archive change (commit `7e77958`, February 18).

Merely reaching eventual fork convergence does not satisfy the requested
originator-only proposal and identical-block adoption contract.

### P1: Block import accepts a regression of the committed champion

The retained test imports score -0.3, restores it as the committed best, then
imports a correctly hashed -0.2 block. Both append successfully. Structural
block validation does not enforce the semantic champion transition.

### P1: Acceptance transaction body can change without invalidating import

The retained mutation changes an accepted score after its transaction ID and
block Merkle root were computed. Both JSON and SQLite imports accept it.
Their structural checks do not rebind this acceptance payload to its ID.
No such tampering was observed in the live chain.

### Important distinctions

- The reveal handler DOES compare with `_domain_best` before trusted acceptance
  (line 1631). The inferior-reveal control passes when that state is correct.
  An earlier claim that no comparison exists was wrong.
- Core block generation clears pending work. Repeated semantic optimum IDs
  under newly created transactions are not proof of an uncleared mempool.
- Trusted mode is configured: peer model retraining is skipped, not proven
  reproducible by this audit. Scientific TEST was not opened. Training seeds
  and independent island search seeds have different roles.
- Acceptance currently supplies `abs(reported)` as work, not a measured delta
  from the predecessor champion. This older behavior needs explicit contract
  review; it was not introduced by the external archive commit.

## Executable checks

`tests/test_network_champion_contract_audit.py` is a deliberately red contract
suite against the deployed revision, not a production repair. It uses temporary
nodes with no started sockets and no training. Expected current result:
five failures and one passing inferior-reveal control.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q \
  tests/test_network_champion_contract_audit.py
```

Read-only fleet recheck, with actual endpoints supplied privately:

```bash
python tools/audit_network_champion.py \
  --node node-a=http://HOST_A:PORT_A \
  --node node-b=http://HOST_B:PORT_B \
  --node node-c=http://HOST_C:PORT_C \
  --node node-d=http://HOST_D:PORT_D \
  --domain YOUR_DOMAIN --output /private/path/chain-report.json
```

Exit 0 means the script's checked properties pass, 2 means a violation and 3
means an incomplete/changing snapshot. It cannot certify model evaluation,
archive contents or every consensus rule. Node labels must be nonprivate.

## Correction requirements, before certification

1. Restore a reference-aware, integrity-checked acceptance projection for
   recovery and fork scoring. Preserve the historical acceptance semantics,
   including verified score and work, rather than assuming the raw candidate
   body contains all consensus evidence. Missing evidence must remain unresolved.
2. Restore committed champion knowledge before accepting new candidates,
   including restart when already caught up with peers.
3. Ensure only the discovering optimizer proposes its block; remote acceptance
   must not leave duplicate mintable work. Validate transitions and prevent
   duplicate optimum credit on import, synchronization and fork adoption.
4. Bind acceptance bodies to their transaction IDs with compatibility tests for
   legacy/custom transaction ID schemes, not a blind global ID rewrite.
5. Exercise restart, stale messages, two simultaneous improvements, delayed
   peers, archive failures and reconnect. Compare every committed block hash
   after convergence. Repeat this fleet audit after deployment.

Do not modify block history or replace the chart with a cumulative minimum to
conceal these findings. Do not change optimization fitness, candidate seeds,
scientific data splits or proof rules while treating the work as a UI fix.
