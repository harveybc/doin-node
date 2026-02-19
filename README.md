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

## Multi-Node Predictor Deployment (Real Example)

This deploys the [harveybc/predictor](https://github.com/harveybc/predictor) timeseries system across multiple machines with DOIN handling island-model migration.

### Prerequisites (each machine)

```bash
# Install DOIN packages
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-node.git
pip install git+https://github.com/harveybc/doin-plugins.git

# Clone predictor (the ML system DOIN wraps)
git clone --branch main --single-branch --depth 1 https://github.com/harveybc/predictor.git
cd predictor && pip install -e .
```

### Node 1 — First Node (seed)

Create `config_node1.json`:

```json
{
  "host": "0.0.0.0",
  "port": 8470,
  "data_dir": "./doin-data-predictor",
  "bootstrap_peers": [],
  "network_protocol": "gossipsub",
  "discovery_enabled": true,
  "initial_threshold": 1e-6,
  "quorum_min_evaluators": 1,
  "storage_backend": "sqlite",
  "fee_market_enabled": false,
  "domains": [{
    "domain_id": "predictor-timeseries",
    "optimize": true,
    "evaluate": true,
    "optimization_plugin": "predictor",
    "inference_plugin": "predictor",
    "has_synthetic_data": true,
    "synthetic_data_validation": false,
    "target_performance": 999.0,
    "optimization_config": {
      "predictor_root": "/path/to/predictor",
      "load_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
      "predictor_plugin": "mimo",
      "preprocessor_plugin": "stl_preprocessor",
      "target_plugin": "default_target",
      "pipeline_plugin": "stl_pipeline",
      "step_size_fraction": 0.15,
      "epochs": 50,
      "batch_size": 32,
      "population_size": 10,
      "n_generations": 5,
      "early_patience": 15,
      "early_stopping_patience": 2
    },
    "param_bounds": {
      "encoder_conv_layers": [1, 3],
      "encoder_base_filters": [16, 64],
      "encoder_lstm_units": [8, 32],
      "learning_rate": [1e-5, 0.01],
      "batch_size": [16, 64],
      "l2_reg": [1e-7, 0.001],
      "decoder_dropout": [0.0, 0.5]
    },
    "resource_limits": {
      "max_training_seconds": 3600,
      "max_memory_mb": 14000,
      "max_epochs": 2000
    }
  }],
  "experiment_stats_file": "./predictor_stats.csv"
}
```

Start:

```bash
cd /path/to/predictor
doin-node --config /path/to/config_node1.json --log-level INFO --olap-db predictor_olap.db
```

### Node 2+ — Additional Nodes

Same config, but add `bootstrap_peers` pointing to Node 1:

```json
{
  "bootstrap_peers": ["192.168.1.100:8470"],
  ...
}
```

Nodes discover each other via LAN scan + bootstrap. Each runs the **full DEAP genetic algorithm** independently. When a node finds a champion, it broadcasts parameters on-chain; other nodes auto-accept if better and inject into their population (island model migration).

### Three-Level Patience System

DOIN's optimization pipeline uses three distinct patience/stopping levels. Understanding these is critical for tuning:

| Level | Name | Config Key | What It Controls | Default |
|-------|------|------------|-----------------|---------|
| **L1** | Candidate Training | `early_patience` | Keras `model.fit()` early stopping — epochs without val_loss improvement before stopping ONE candidate | 80–100 |
| **L2** | Stage Progression | `optimization_patience` | DEAP GA — generations without best-fitness improvement before advancing to the next incremental stage | 8–10 |
| **L3** | Meta-Optimizer | *(not yet implemented)* | Network-level performance predictor trained on (params→performance) from many L2 experiments via OLAP data | — |

- **L1** (`early_patience`): Low values (15) = fast but shallow training per candidate. High values (100) = thorough training, slower per candidate.
- **L2** (`optimization_patience`): Low values (2) = quickly advance through stages. High values (10) = more generations to find improvements per stage.
- **L3**: Will train on the on-chain OLAP cube data from ALL network participants to predict promising hyperparameter regions.

### Key Configuration Options

| Field | Description | Default |
|-------|-------------|---------|
| `synthetic_data_validation` | `false` = auto-accept/reject by reported MAE (skip quorum). `true` = full evaluator verification | `true` |
| `population_size` | GA population per generation | 10 |
| `n_generations` | Generations per stage | 5 |
| `epochs` | Training epochs per candidate | 50 |
| `discovery_enabled` | Auto-discover LAN peers | `true` |
| `network_protocol` | `gossipsub` (production) or `flooding` (legacy) | `gossipsub` |
| `initial_threshold` | Min threshold for block generation | `1e-6` |

### What You'll See

1. **Dashboard** at `http://<ip>:8470/dashboard` — live events, peers, optimization progress
2. **Optimizer Events tab** — champion discoveries (🏆), broadcasts (📡), auto-accepts (✅), auto-rejects (❌), peer connections (🔗)
3. **Domains tab** — current champion with train/val/test MAE vs naive baselines
4. **Log** — `Broadcast optimae_reveal → 2 peers` confirms migration is flowing

### Example: 3-Node LAN Deployment

See ready-to-use config files in `examples/`:
- `predictor_single_node.json` — seed node (Dragon, RTX 4090)
- `predictor_omega_node.json` — GPU node (Omega, RTX 4070)
- `predictor_delta_node.json` — CPU-only node (Delta)

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
