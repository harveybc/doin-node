# doin-node

**Status: ACTIVE — the unified participant runtime.**
It supersedes the retired standalone
[doin-optimizer](https://github.com/harveybc/doin-optimizer) and
[doin-evaluator](https://github.com/harveybc/doin-evaluator) clients: optimizer,
evaluator/inference-worker, and network-node responsibilities all run as
configured roles inside this single process.

`doin-node` is the runnable node of DOIN, the Decentralized Optimization and
Inference Network (older code docstrings abbreviate it **DON**). One process
per machine loads a JSON config that selects, per optimization domain, whether
this machine optimizes, evaluates, or only relays; which plugins implement the
domain; and how it joins the peer network. Consensus is proof-of-optimization:
blocks are generated when verified optimization improvements cross a dynamic
threshold. Per-round optimization metrics are recorded to an embedded SQLite
star schema as the node runs.

## Research origin

DOIN derives from Harvey Demian Bastidas Caicedo's 2018 Master's in
Engineering thesis at Pontificia Universidad Javeriana Cali, *Computación
Evolutiva Descentralizada de Modelo Híbrido usando Blockchain y Prueba de
Trabajo de Optimización*. The present runtime is a later implementation: it
unifies the optimizer, evaluator and network roles that the thesis and early
repositories represented as separate participants.

Read the original thesis in the doin-core repository:
[Hybrid-Model Decentralized Evolutionary Computing Using Blockchain and Proof-of-Work Optimization](https://github.com/harveybc/doin-core/blob/master/docs/Hybrid-Model%20Decentralized%20Evolutionary%20Computing%20Using%20Blockchain%20and%20Proof-of-Work%20Optimization.pdf).

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent
with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart**
> section end to end: set up the environment, run the smoke test, start the
> example single-node quadratic run, then tell me the exact URL or file paths
> where I can see the results and one query I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by
most coding agents.

## Role and non-responsibilities

**Role:** the event loops for the optimizer/evaluator roles, HTTP transport and
peer protocols, blockchain storage and sync, shared-population coordination,
the monitoring dashboard, and the analytics layer
([`src/doin_node/stats/`](src/doin_node/stats)).

**Not in this repository:**

- Protocol primitives (consensus rules, models, wire schema, crypto, plugin
  ABCs and entry-point groups) are defined in
  [doin-core](https://github.com/harveybc/doin-core) and imported here.
- Plugin implementations live in
  [doin-plugins](https://github.com/harveybc/doin-plugins) or any external
  package registering the `doin.*` entry-point groups; `doin-node` declares no
  plugins of its own.
- Domain optimizers/models remain external installable packages (for example
  [predictor](https://github.com/harveybc/predictor) and
  [agent-multi](https://github.com/harveybc/agent-multi)) that work locally
  without DOIN.

## Architecture

| Module | Responsibility |
|---|---|
| [`src/doin_node/unified.py`](src/doin_node/unified.py) | `UnifiedNode`: role dispatch (optimizer loop, shared-population optimizer loop, evaluator loop, gossip/discovery/maintenance loops), champion migration, candidate leasing, dedup |
| [`src/doin_node/cli.py`](src/doin_node/cli.py) | `doin-node` console script: config parsing/validation, plugin loading, lifecycle |
| [`src/doin_node/blockchain/`](src/doin_node/blockchain) | Chain state, block application, sync |
| [`src/doin_node/network/`](src/doin_node/network) | aiohttp transport, flooding and gossipsub protocols, peer discovery |
| [`src/doin_node/storage/`](src/doin_node/storage) | Chain persistence backends (`sqlite` via aiosqlite, or `json`) |
| [`src/doin_node/stats/`](src/doin_node/stats) | OLAP star schema (v3: `dim_domain`, `dim_experiment`, `fact_round`, `fact_experiment_summary`, `fact_chain_optimae`), experiment tracker (CSV + SQLite dual-write), chain metrics, SQLite→PostgreSQL sync. Note that `fact_chain_optimae` is populated only by `ingest_from_chain()`, which currently has no runtime caller |
| [`src/doin_node/dashboard/`](src/doin_node/dashboard) | Web monitoring UI served at `/dashboard` |
| [`src/doin_node/scheduling/`](src/doin_node/scheduling), [`validation/`](src/doin_node/validation), [`benchmarks/`](src/doin_node/benchmarks) | GPU/job scheduling, input validation, benchmark harnesses |

[`src/doin_node/node.py`](src/doin_node/node.py) is an earlier orchestrator
kept for reference; the CLI runs `UnifiedNode` from `unified.py`.

## Requirements

From [`pyproject.toml`](pyproject.toml):

- Python `>=3.10`
- `doin-core>=0.1.0`, `aiohttp>=3.9`, `aiosqlite>=0.20`, `psutil>=5.9`
- Dev extras: `pytest`, `pytest-cov`, `pytest-asyncio`, `mypy`, `ruff`
- Domain plugins bring their own requirements (the quadratic reference domain
  needs only `doin-plugins`; predictor/trading domains need their external ML
  stacks)

## Installation

Install the three packages in dependency order (no PyPI releases; source
installs):

```bash
git clone https://github.com/harveybc/doin-core.git
git clone https://github.com/harveybc/doin-plugins.git
git clone https://github.com/harveybc/doin-node.git
pip install -e doin-core -e doin-plugins -e doin-node
```

## Quickstart: single-node quadratic run

Runs one process that optimizes *and* evaluates the self-contained quadratic
reference domain, using the repository-owned config
[`examples/quadratic_single_node.json`](examples/quadratic_single_node.json):

```bash
cd doin-node
doin-node --config examples/quadratic_single_node.json
# equivalently: python -m doin_node.cli --config examples/quadratic_single_node.json
```

The node creates OLAP schema v3 at `doin-data-single/olap.db`, registers the
`quadratic` domain (optimize=evaluate=true), loads the `simple_quadratic`
optimizer/evaluator/synthetic plugins, initializes a chain with a genesis
block, and serves the dashboard at `http://localhost:8470/dashboard`. Stop it
with Ctrl+C.

Note that starting a node kills any process **owned by the same user** already
listening on its port (`Transport.start()` → `_kill_stale_process_on_port`).
Check the port is free, or pass `--port`.

Useful CLI flags (see `doin-node --help`): `--port`, `--data-dir`, `--peers`,
`--identity`, `--stats-file`, `--olap-db`, `--reset-chain`, `--log-level`.

## Unified role configuration (per-machine JSON)

Each machine runs one `doin-node` process with one JSON config. Top-level keys
map to `UnifiedNodeConfig` in
[`src/doin_node/unified.py`](src/doin_node/unified.py); every key is optional
and defaults are defined there.

| Area | Keys |
|---|---|
| Identity / network | `node_label`, `host`, `port`, `data_dir`, `identity_file`, `bootstrap_peers`, `network_protocol` (`gossipsub` or `flooding`), `gossip_heartbeat_interval`, `discovery_enabled`, `discovery_interval` |
| Consensus | `target_block_time`, `initial_threshold`, `acceptance_tolerance`, `quorum_min_evaluators`, `quorum_fraction`, `quorum_tolerance`, `commit_reveal_max_age`, `finality_confirmation_depth`, `external_anchor_interval`, `require_deterministic_seed` |
| Role loops | `optimizer_loop_interval`, `eval_poll_interval`, `eval_max_concurrent` |
| Shared population | `shared_min_peers`, `shared_claim_timeout`, `shared_claim_result_patience`, `shared_claim_settle_seconds`, `shared_claim_confirmation_rounds`, `shared_initialize_before_peers`, `shared_peer_wait_timeout` |
| Storage / analytics | `storage_backend` (`sqlite` or `json`), `db_path`, `snapshot_interval`, `prune_keep_blocks`, `experiment_stats_file`, `olap_db_path`, `dashboard_enabled`, `reset_chain` |
| Economics (optional) | `fee_market_enabled`, `fee_config` |

The `domains` list assigns roles and plugins per domain (parsed into
`DomainRole` by [`src/doin_node/cli.py`](src/doin_node/cli.py)):

```json
{
  "domain_id": "quadratic",
  "optimize": true,
  "evaluate": true,
  "optimization_plugin": "simple_quadratic",
  "inference_plugin": "simple_quadratic",
  "synthetic_data_plugin": "simple_quadratic",
  "has_synthetic_data": true,
  "optimization_config": { "n_params": 10, "step_size": 0.5 },
  "param_bounds": { "x": [-20.0, 20.0] }
}
```

Additional per-domain keys: `inference_config`, `synthetic_data_config`,
`synthetic_data_validation`, `higher_is_better`, `metric_type`,
`resource_limits`, `incentive_config`, `target_performance` (convergence
stop). Plugin names are resolved through the `doin.optimization` /
`doin.inference` / `doin.synthetic_data` entry-point groups defined by
doin-core; implementations come from
[doin-plugins](https://github.com/harveybc/doin-plugins) or any external
package. A domain without a synthetic-data plugin gets zero consensus weight.

## Distributed usage

**Two machines / two processes, quadratic domain:**
[`examples/quadratic_node_a.json`](examples/quadratic_node_a.json) and
[`examples/quadratic_node_b.json`](examples/quadratic_node_b.json) (node B
bootstraps to node A). The helper
[`scripts/run_two_node_test.sh`](scripts/run_two_node_test.sh) installs the
three packages, launches both nodes locally, and polls `/status`.

**Predictor and trading campaigns:** [`examples/`](examples) contains
per-machine configs for timeseries-predictor domains, and
[`examples/trading/`](examples/trading) holds the per-machine configs (93
files, grouped per campaign phase) used for shared-population trading
experiments driven by the external agent-multi package. In a shared-population
campaign every participating machine's JSON points at the same `domain_id`,
deterministic-seed/genome contract, and shared-population settings, and each
machine differs only in its identity, port, data directory, and role booleans.

**Shared-population semantics** (champion migration, candidate claim leasing,
duplicate-evaluation avoidance, fork-choice tie-breaks, restart recovery) are
specified normatively in
[`docs/shared_population_semantics.md`](docs/shared_population_semantics.md)
and implemented in `unified.py`. In brief: candidates are claimed through
lease-based coordination over HTTP (`/api/shared/candidates`, `.../claim`,
`.../release`, `.../result`) where only the claim owner's heartbeat renews a
lease; results deduplicate by transaction id with first-result-wins conflict
handling; champions propagate optimistically and at stage boundaries; a
restarting node recovers the canonical population from the chain and rejoins
quorum before claiming again.

## Tests

```bash
pip install -e .[dev]
pytest -q
```

`pytest -q` reports **409 passed** in about 10 seconds, across
[`tests/`](tests). `ruff` and `mypy` are configured in `pyproject.toml` and
shipped in the dev extras, but CI runs neither, so the codebase is not known to
be lint- or type-clean.

## Artifacts, outputs, and reproducibility

Per run, under the configured `data_dir`:

- `identity.json` — the node's private key (see security notes)
- `chain.json` (json backend) or `chain.db` (sqlite backend) — the blockchain
- `olap.db` — OLAP star schema (schema v3), optionally synced to PostgreSQL
  via [`src/doin_node/stats/olap_sync.py`](src/doin_node/stats/olap_sync.py)

Plus the experiment stats CSV and a `.summary.json` next to it (path set by
`experiment_stats_file` / `--stats-file`). The dashboard serves live state at
`/dashboard`.

Reproducibility: with `require_deterministic_seed` enabled, evaluation seeds
are derived from commitment hashes (doin-core `deterministic_seed`), so
verifications are reproducible from on-chain data. The genesis block is fixed,
so a fresh experiment means a new `domain_id` and a new `data_dir` — never
reuse a data directory across incompatible configs. Sample sealed results from
a completed campaign are kept under [`examples/results/`](examples/results).

## Security and safety notes

- `identity.json` holds the node's private key: keep it out of version
  control and restrict file permissions.
- Consensus hardening (from doin-core, wired in here): commit-reveal for
  optimae, quorum verification with tolerance, asymmetric reputation,
  resource-limit validation, finality checkpoints, external anchoring,
  deterministic per-evaluator seeds.
- The HTTP transport is plain HTTP intended for trusted/private networks; do
  not expose node ports to untrusted networks.
- No exchange, broker, or API credentials are required or read by this
  repository. Trading domains operate purely on historical/synthetic data
  through simulation and backtesting; no live orders are placed.

## Limitations and legacy notes

- Version `0.1.0` (alpha); no PyPI releases; wire compatibility between
  versions not guaranteed — run matching versions on all peers.
- The standalone [doin-optimizer](https://github.com/harveybc/doin-optimizer)
  and [doin-evaluator](https://github.com/harveybc/doin-evaluator) clients are
  retired and are **not** required for any current deployment; their roles are
  the `optimize` / `evaluate` booleans of this package's domain config.
- Both `gossipsub` and `flooding` protocols are implemented; the in-repo
  example configs (including all shared-population trading examples) use
  `flooding`.
- `src/doin_node/node.py` is a pre-unified orchestrator kept for reference
  only.

## Related repositories and docs

- [doin-core](https://github.com/harveybc/doin-core) — protocol primitives
  this node implements; its `docs/` folder holds the network/security/
  scalability papers
- [doin-plugins](https://github.com/harveybc/doin-plugins) — plugin
  implementations loaded by entry-point name
- [predictor](https://github.com/harveybc/predictor),
  [agent-multi](https://github.com/harveybc/agent-multi) — external domain
  packages used by the predictor/trading domains
- [`docs/shared_population_semantics.md`](docs/shared_population_semantics.md)
  — normative shared-population specification

## License

Declared MIT in [`pyproject.toml`](pyproject.toml); the repository does not
currently ship a standalone `LICENSE` file.
