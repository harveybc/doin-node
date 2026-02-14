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
- **Block sync protocol** — initial sync on startup, catch-up on announcements
- **Full security pipeline** — all 10 hardening measures wired in
- **Pull-based task queue** — evaluators poll for work, priority-ordered
- **Experiment tracking** — per-round CSV + SQLite OLAP dual-write from round 1
- **On-chain experiment metrics** — OPTIMAE_ACCEPTED transactions carry experiment metadata
- **Per-domain convergence** — `target_performance` stop criteria per domain
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

## Tests

```bash
python -m pytest tests/ -v
# 290 tests passing
```

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — This package
- [doin-plugins](https://github.com/harveybc/doin-plugins) — Domain plugins
- [doin.network](https://doin.network) — Project homepage
