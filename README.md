# doin-node

Unified DOIN node — single configurable process that can optimize, evaluate, and relay.

## What is DOIN?

DOIN (Decentralized Optimization and Inference Network) is a blockchain-based system where nodes collaboratively optimize ML models. Block generation is triggered by verified optimization work — **Proof of Optimization**.

Visit [doin.network](https://doin.network) for the full overview.

## This Package

`doin-node` is the main entry point for running a DOIN node. Like a Bitcoin node that can mine + validate + relay, a DOIN node can optimize + evaluate + relay — all configurable per domain via JSON.

### Features
- **Unified architecture** — single process, configurable roles per domain
- **HTTP transport** — aiohttp-based, simple and debuggable
- **GossipSub protocol** — O(log N) message propagation through mesh-based gossip
- **Peer discovery** — LAN scan + bootstrap nodes + PEX (peer exchange)
- **Block sync protocol** — initial sync on startup, catch-up on announcements
- **Full security pipeline** — all 10 hardening measures wired in
- **Island model migration** — champion solutions shared via on-chain optimae, injected into other nodes' populations
- **DEAP GA wrapper** — predictor plugin wraps full DEAP genetic algorithm via callback hooks (doesn't replace it)
- **Pull-based task queue** — evaluators poll for work, priority-ordered
- **Real-time dashboard** — web UI at `:8470/dashboard` for monitoring optimization, training, evaluations, chain, peers
- **Experiment tracking** — per-round CSV + SQLite OLAP dual-write from round 1
- **On-chain experiment metrics** — OPTIMAE_ACCEPTED transactions carry experiment metadata
- **Per-domain convergence** — `target_performance` stop criteria per domain
- **Domain sharding** — nodes only process subscribed domains
- **EIP-1559 fee market** — dynamic base fee adjusts with demand
- **GPU scheduler** — resource marketplace matching jobs to hardware
- **L2 payment channels** — off-chain micropayments for inference requests
- **PostgreSQL sync** — export OLAP data to PostgreSQL for Metabase dashboards

### Security Systems (all wired in)
1. Commit-reveal for optimae
2. Random quorum selection
3. Asymmetric reputation penalties
4. Resource limits + bounds validation
5. Finality checkpoints
6. Reputation decay (EMA)
7. Min reputation threshold
8. External checkpoint anchoring
9. Fork choice rule (heaviest chain)
10. Deterministic per-evaluator seeds

## Install

```bash
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-node.git
```

## Usage

```bash
doin-node --config config.json
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config` | Path to JSON config file |
| `--stats-file` | CSV experiment stats output path |
| `--olap-db` | SQLite OLAP database path |

See [INSTALL.md](https://github.com/harveybc/doin-node/blob/master/docs/INSTALL.md) for configuration examples.

### Configuration Example

```json
{
  "host": "0.0.0.0",
  "port": 8470,
  "data_dir": "./doin-data",
  "bootstrap_peers": ["seed1.doin.network:8470"],
  "experiment_stats_file": "./stats.csv",
  "olap_db_path": "./olap.sqlite",
  "domains": [
    {
      "domain_id": "my-ml-domain",
      "optimize": true,
      "evaluate": true,
      "has_synthetic_data": true,
      "optimization_plugin": "my_optimizer",
      "synthetic_data_plugin": "my_synth_gen",
      "target_performance": -1.0
    }
  ]
}
```

## Dashboard

Access the real-time monitoring dashboard at `http://localhost:8470/dashboard`.

Tracks optimization progress, training status, evaluations, chain state, events, and peer connections in a single web UI.

### Dashboard API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/dashboard` | Web UI (HTML dashboard) |
| `/api/node` | Node identity and configuration |
| `/api/peers` | Connected peers and mesh topology |
| `/api/optimization` | Current optimization state per domain |
| `/api/training` | Active training jobs and progress |
| `/api/evaluations` | Evaluation queue and results |
| `/api/metrics` | Performance metrics and statistics |
| `/api/events` | Real-time event stream |
| `/api/chain` | Chain state, height, recent blocks |
| `/api/plugins` | Loaded plugins and domain configuration |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Node status (chain, peers, tasks, security) |
| `/chain/status` | GET | Chain height, tip hash, finalized height |
| `/chain/blocks?from=X&to=Y` | GET | Fetch blocks by range (max 50) |
| `/chain/block/{index}` | GET | Fetch single block |
| `/tasks/pending` | GET | List pending tasks |
| `/tasks/claim` | POST | Claim a task |
| `/tasks/complete` | POST | Complete a task |
| `/inference` | POST | Submit inference request |
| `/stats` | GET | Experiment tracker stats + OLAP data |
| `/stats/experiments` | GET | List all experiments with summaries |
| `/stats/rounds?experiment_id=X&limit=N` | GET | Round history for an experiment |
| `/stats/chain-metrics?domain_id=X` | GET | On-chain experiment metrics |
| `/stats/export` | GET | Download OLAP database |
| `/fees` | GET | Fee market stats |
| `/peers` | GET | Peer list |

### Coin & Difficulty
- Native DOIN coin with block rewards (65% optimizers, 30% evaluators, 5% generator)
- Bitcoin/Ethereum hybrid difficulty adjustment (epoch + per-block EMA)
- Balance tracker with transfers, fees, nonce replay protection

## Stats & Analytics

DOIN nodes automatically track experiment data via a dual-write pipeline:

1. **CSV** — Per-round flat file with 28+ columns (ExperimentTracker). Human-readable, easy to grep.
2. **SQLite OLAP** — Star schema database written locally from round 1, zero configuration needed.

### OLAP Star Schema

| Table | Type | Description |
|-------|------|-------------|
| `dim_experiment` | Dimension | Experiment metadata |
| `dim_domain` | Dimension | Domain configuration |
| `fact_round` | Fact | Per-round metrics (28+ columns) |
| `fact_experiment_summary` | Fact | Aggregated experiment results |
| `fact_chain_optimae` | Fact | On-chain accepted optimae metrics |

### The Chain as OLAP Cube

OPTIMAE_ACCEPTED transactions carry experiment metrics on-chain:
- `experiment_id`, `round_number`, `time_to_this_result_seconds`
- `optimization_config_hash`, `data_hash` (hashes only — no raw data)

Every node syncing the chain gets the full experiment history of ALL participants. This enables:
- **Cross-node analytics** via Metabase or any BI tool
- **L3 meta-optimizer** training on collective optimization data
- **PostgreSQL sync** for production dashboards

### Three-Level Optimization Pipeline
- **L1**: Keras/AdamW — individual model training
- **L2**: Genetic algorithms / optimization plugins — what DOIN decentralizes
- **L3**: Deep learning meta-optimizer trained on OLAP data from all network participants

## Benchmarks

### 3-Node Island Model Benchmark

Running on: **Dragon** (RTX 4090) + **Omega** (RTX 4070) + **Delta** (CPU-only, SLI 2× GFX 550M)

| Setup | Rounds to Converge | Speedup |
|-------|-------------------|---------|
| Single node (Omega, RTX 4070) | 39 | 1× |
| Two nodes (Dragon + Omega) | 5–6 | **~7×** |
| Delta solo (CPU) | not converged at 1680s | — |
| Dragon + Omega (hard target) | 78 rounds, 1292s | **19% faster** |

Champion migration via on-chain optimae exchange is working: when one node finds a better solution, it broadcasts parameters and other nodes inject them into their populations (island model). Delta (CPU-only, 3–4× slower convergence) benefits most from receiving champions.

Auto-discovery via LAN scan + PEX enables zero-config peer connection.

## Tests

```bash
python -m pytest tests/ -v
# 289 tests passing
```

## Island Model Migration

DOIN implements the **island model** from evolutionary computation over a real blockchain:

1. Each node runs its own optimization (e.g., DEAP genetic algorithm via the predictor plugin)
2. When a node finds a champion solution, it broadcasts parameters via on-chain optimae
3. Other nodes receive the champion and inject it into their local populations
4. The predictor plugin wraps DEAP's full GA via callback hooks: `on_generation_start`, `on_generation_end`, `on_between_candidates`, `on_champion_found`
5. DOIN doesn't replace the optimizer — it wraps it, adding decentralized champion sharing

This means any evolutionary optimizer (DEAP, NEAT, custom GA) gets automatic island-model parallelism just by running on multiple DOIN nodes.

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — This package
- [doin-optimizer](https://github.com/harveybc/doin-optimizer) — Standalone optimizer runner
- [doin-evaluator](https://github.com/harveybc/doin-evaluator) — Standalone evaluator service
- [doin-plugins](https://github.com/harveybc/doin-plugins) — Domain plugins
- [doin.network](https://doin.network) — Project homepage
