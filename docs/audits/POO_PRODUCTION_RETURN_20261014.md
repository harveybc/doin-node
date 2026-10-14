# Production Alpha Return

The owner authorized a coordinated clean H4 restart after the consensus audit.
Retired chains, receipts, populations and code remain in their old namespaces.
Running evaluators finished before their units were stopped; the wrapper refused
the next evaluator, and no old population or checkpoint initialized the new run.

## Exact Pins

| Component | Live revision |
| --- | --- |
| doin-node | 393049dd4591076d4012f6ff1974aa42c4a390cf |
| doin-core | 9f9d3f757c5ee69c2995352a2370021fc837534c |
| doin-plugins | 2fc0523e6ee6fa371cee4d7862ca61c087a852fc |
| predictor | 3e5d8b030f10b112dce37301e1b05c2264cd2bab |

All four live participants exposed these exact revisions, seed 6102026 and
scientific config 4a1850ca123298397f597ad4f852993439f5c263aa1e9aac9bfd95f29d5ad2fd.
The new manifest contains 19 features and excludes STLFSI4. H4 origin hashes,
counts and naive sums are unchanged; its support digest changes appropriately
because the input feature identity changed. Feature provenance gates before
TEST remain outstanding; this run is DEVELOPMENT_TUNING.

## Executed Live Acceptance

The first real optimum produced block 1 (height 2 including genesis), with tip
`b44ca942ab20ead652fc349b8809da3f4d529ee488799dfa4c0f4e52318bd634`.
All four complete serialized chains matched, SHA-256
`1747c18ea902e5cf55d0ec9ab141d490f11125802d1c905a9ffecf05db80244b`.
The independent controller replayed strict champion ordering and exact work
from parent history, verified archived bytes and recovered the metric vector.
The dashboard returned one champion point, zero invalid acceptances and zero
archive failures. The development objective was -0.004466807460606459; this
combines TRAIN and INNER fold ratios and is not prospective performance.

The first candidate was projected into the deployed warehouse and exactly read
back. Subsequent live candidates also reached OFFCHAIN_VERIFIED. They remain
explicitly unfinalized until legitimate later champion blocks provide depth.
Warehouse delivery uses private loopback transports; no public bind, credential
printing or warehouse restart was needed.

## Automation

Each peer has a bounded one-shot archive timer. The coordinator's one-shot
controller checks exact full chains, code/config/seed, work proofs and archive
readback, records host RAM/disk/cgroup OOM counters, collects coherent terminals,
seals outboxes, closes corpus readback and invokes the reviewed H72 successor.
Jobs survive agent exit. Two memory-heavy evaluators on the dual-GPU host are
serialized with priority for the preferred GPU. All four peers remain online.

Live controller code separately represents absence of a finality checkpoint
as height -1 and empty hash. No checkpoint or housekeeping block is fabricated.
The retired successor timers are disabled so they cannot activate an old run
after a power outage. Private runtime paths, endpoints and tokens are not
stored in this public report.

## Verification And Remaining Acceptance

Core: 336 tests. Node: 627 tests, 37 warnings. Predictor archive/provider/restart
integration: 85 tests; updated closure/successor policies: 42 tests. Controller
checkpoint handling: 83 tests plus 20 collector tests. Live cgroup counters
reported zero OOM kills, and all hosts retained free RAM and disk.

H4 completion, all terminal corpus closures and the actual H72 handoff are
future acceptance points. Only the first committed improvement and ongoing
archive delivery are certified by the live evidence above. Dashboard network
version warnings still compare unrelated installed packages; the four required
campaign components match through the canonical chain API and fleet verifier.
