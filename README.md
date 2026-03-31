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

## Requirements

- **Python** >= 3.10 (tested on 3.12)
- **OS**: Linux (Ubuntu 22.04+, Debian 12+), macOS. Windows via WSL2.
- **GPU** (optional): NVIDIA GPU with CUDA support for TensorFlow acceleration

### Core Dependencies

| Package | Required By | Purpose |
|---------|------------|---------|
| `pydantic>=2.0` | doin-core | Model validation & serialization |
| `cryptography>=41.0` | doin-core | Identity keys, hashing |
| `aiohttp>=3.9` | doin-node | HTTP transport, dashboard |
| `aiosqlite>=0.20` | doin-node | SQLite OLAP storage |
| `numpy>=1.24` | doin-plugins | Numerical operations |

### Predictor Plugin Dependencies (for ML optimization)

| Package | Purpose |
|---------|---------|
| `tensorflow` | Deep learning backend (Keras) |
| `nvidia-cudnn-cu12` | CUDA acceleration (GPU nodes only) |
| `numpy`, `pandas`, `scipy` | Data processing |
| `deap` | Genetic algorithm (DEAP GA optimizer) |
| `h5py` | Model serialization |
| `tensorflow-probability` | Bayesian inference |
| `PyWavelets`, `pmdarima` | Signal decomposition |
| `tqdm`, `matplotlib` | Progress bars, plotting |

See the full list in [`predictor/requirements.txt`](https://github.com/harveybc/predictor/blob/main/requirements.txt).

## Install

> **Important**: Modern Linux distros (Debian 12+, Ubuntu 23.04+) enforce PEP 668
> which blocks system-wide pip installs. You **must** use a virtual environment.

### Option A: Conda (recommended for GPU support)

```bash
# Create and activate environment
conda create -n doin python=3.12 -y
conda activate doin

# Install TensorFlow with GPU support (if NVIDIA GPU available)
pip install tensorflow[and-cuda]
# Or CPU-only:
# pip install tensorflow

# Install DOIN packages (order matters — core first)
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-node.git
pip install git+https://github.com/harveybc/doin-plugins.git

# Clone and install predictor (the ML system DOIN wraps)
git clone --branch main --single-branch --depth 1 https://github.com/harveybc/predictor.git
cd predictor
pip install -r requirements.txt
pip install -e .
cd ..
```

### Option B: Python venv

```bash
# Create and activate virtual environment
python3 -m venv ~/doin-env
source ~/doin-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install TensorFlow
pip install tensorflow

# Install DOIN packages (order matters — core first)
pip install git+https://github.com/harveybc/doin-core.git
pip install git+https://github.com/harveybc/doin-node.git
pip install git+https://github.com/harveybc/doin-plugins.git

# Clone and install predictor
git clone --branch main --single-branch --depth 1 https://github.com/harveybc/predictor.git
cd predictor
pip install -r requirements.txt
pip install -e .
cd ..
```

### Verify Installation

```bash
# Check DOIN node is available
doin-node --help

# Check plugins are registered
python -c "from importlib.metadata import entry_points; eps = entry_points(); print([ep.name for ep in eps.select(group='doin.optimization')])"
# Expected: ['simple_quadratic', 'predictor']

# Check TensorFlow GPU (optional)
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
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

## Running a TCN NEAT Experiment

This section covers the deployment of a DOIN network for distributed NEAT optimization of TCN time series models using the [harveybc/predictor](https://github.com/harveybc/predictor).

### Prerequisites (each machine)

#### 1. Install conda environment

```bash
conda create -n tensorflow python=3.12 -y
conda activate tensorflow
pip install tensorflow[and-cuda]
```

#### 2. Clone and install all 4 repositories (order matters)

```bash
cd ~/Documents/GitHub   # or any preferred directory

git clone https://github.com/harveybc/doin-core.git
cd doin-core && pip install -e . && cd ..

git clone https://github.com/harveybc/doin-node.git
cd doin-node && pip install -e . && cd ..

git clone https://github.com/harveybc/doin-plugins.git
cd doin-plugins && pip install -e . && cd ..

git clone https://github.com/harveybc/predictor.git
cd predictor && pip install -r requirements.txt && pip install -e . && cd ..
```

#### 3. Configure GPU (CRITICAL — read carefully)

TensorFlow requires three environment variables. Without them, training runs on CPU or crashes with OOM.

**Variable 1 — `TF_FORCE_GPU_ALLOW_GROWTH`**: Prevents the parent process from pre-allocating all GPU memory (which starves training subprocesses). **Must be `true`, not `1`** — TensorFlow rejects the value `"1"` with a parsing error and silently falls back to pre-allocation.

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true    # CORRECT
# export TF_FORCE_GPU_ALLOW_GROWTH=1     # WRONG — TF rejects this silently
```

**Variable 2 — `TF_GPU_ALLOCATOR`**: Enables the async CUDA memory allocator for better multi-process GPU sharing.

```bash
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

**Variable 3 — `LD_LIBRARY_PATH` (machines without system CUDA)**: If CUDA was installed via `pip install tensorflow[and-cuda]` (NOT via system `/usr/local/cuda`), the CUDA libraries live inside the conda environment's `site-packages/nvidia/` directory. TensorFlow cannot find them without this:

```bash
NB=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="${NB}/cudnn/lib:${NB}/cublas/lib:${NB}/cuda_runtime/lib:${NB}/cufft/lib:${NB}/curand/lib:${NB}/cusolver/lib:${NB}/cusparse/lib:${NB}/cuda_cupti/lib:${NB}/nvjitlink/lib:${NB}/cuda_nvrtc/lib:${NB}/nccl/lib"
```

> **How to tell if you need this**: If `/usr/local/cuda/lib64/libcudart.so` exists, you have system CUDA and do NOT need `LD_LIBRARY_PATH`. If it doesn't exist, you need the export above. When in doubt, set it — it does no harm on machines with system CUDA.

> **Blackwell GPUs (RTX 50xx)**: TensorFlow 2.21 does not ship pre-compiled CUDA kernels for compute capability 12.x. On first launch, all kernels are JIT-compiled from PTX, which adds ~30 minutes of startup delay. This only happens once per kernel shape.

#### 4. Verify GPU setup

```bash
conda activate tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
# Set LD_LIBRARY_PATH if needed (see step 3)

python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); assert len(gpus) > 0, 'NO GPU DETECTED'"
```

If this prints `GPUs: []`, fix the `LD_LIBRARY_PATH` before proceeding.

#### 5. Verify DOIN installation

```bash
python -c "from doin_plugins.predictor.optimizer import PredictorOptimizer; print('OK')"
```

#### 6. Edit the node config

Open one of the example configs in `doin-node/examples/` and set `predictor_root` to the **absolute path** of the predictor repository on this machine. This path appears in both `optimization_config` and `inference_config`:

```json
"predictor_root": "/home/harveybc/Documents/GitHub/predictor"
```

Each node config must also have a **unique `node_seed_offset`** (0, 1, 2, ...) for different random seeds.

### Starting a New Experiment from Scratch

This starts a fresh optimization with an empty blockchain. All nodes begin with random NEAT populations and collaborate by sharing champions.

The complete launch sequence for every node is:

```bash
# 1. Activate environment
conda activate tensorflow

# 2. Set GPU environment variables (ALL THREE — see Prerequisites step 3)
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
# If no system CUDA (see step 3):
NB=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="${NB}/cudnn/lib:${NB}/cublas/lib:${NB}/cuda_runtime/lib:${NB}/cufft/lib:${NB}/curand/lib:${NB}/cusolver/lib:${NB}/cusparse/lib:${NB}/cuda_cupti/lib:${NB}/nvjitlink/lib:${NB}/cuda_nvrtc/lib:${NB}/nccl/lib"

# 3. Navigate to doin-node
cd ~/Documents/GitHub/doin-node

# 4. Launch (replace <config> with your node's config file)
nohup python -m doin_node.cli \
  --config examples/<config>.json \
  > /tmp/doin_launch.log 2>&1 &
echo "PID: $!"
```

#### Step 1: Start the Seed Node

Launch the first node. It creates the genesis block. The seed node's `bootstrap_peers` should list the other nodes' IPs (they don't need to be running yet — discovery will find them once they start).

```bash
# On Omega (seed node)
nohup python -m doin_node.cli \
  --config examples/predictor_omega_node_tcn_neat.json \
  > /tmp/doin_omega.log 2>&1 &
```

Verify it started correctly:

```bash
# Wait ~20 seconds for first candidate to start, then:
grep "Created device" /tmp/doin_omega.log
# Expected: "Created device /job:localhost/.../GPU:0 with XXXX MB memory"
# If you see "Skipping registering GPU devices" → GPU NOT working (fix LD_LIBRARY_PATH)

grep "input_layer" /tmp/doin_omega.log
# Expected: "(None, 51, 7)" — 7 features = 1 price + 6 temporal sincos

grep "could not be parsed" /tmp/doin_omega.log
# Should return nothing. If it shows TF_FORCE_GPU_ALLOW_GROWTH error → you used "1" instead of "true"
```

#### Step 2: Join Additional Nodes

On each additional machine, ensure `bootstrap_peers` in the config points to running nodes. The example configs are pre-configured for the 3-node LAN setup.

```bash
# On Gamma
nohup python -m doin_node.cli \
  --config examples/predictor_gamma_node_tcn_neat.json \
  > /tmp/doin_gamma.log 2>&1 &
```

```bash
# On Dragon
nohup python -m doin_node.cli \
  --config examples/predictor_dragon_node_tcn_neat.json \
  > /tmp/doin_dragon.log 2>&1 &
```

#### Step 3: Verify the Network

Run these checks on each node immediately after launch:

```bash
# 1. GPU is being used (NOT CPU)
grep "Created device" /tmp/doin_<node>.log
# Must show GPU name and memory. If missing → training is on CPU.

# 2. No version mismatch (ALL nodes must run identical code)
grep "version mismatch" /tmp/doin_<node>.log
# Must return nothing. If it shows rejections → pull & reinstall on ALL machines.

# 3. Peer connectivity
curl -s http://localhost:8470/chain/status | python3 -m json.tool
# Check chain_height and tip_hash match across nodes

# 4. GPU utilization
nvidia-smi
# Expect 50-95% GPU utilization during training
```

> **Critical**: If `nvidia-smi` shows 0% GPU but the node is running, training is happening on CPU. This means `LD_LIBRARY_PATH` is not set correctly. Kill the node, fix the environment, and relaunch.

#### Step 4: Monitor Progress

- **Dashboard**: `http://<any-node-ip>:8470/dashboard`
- **Logs**: `tail -f /tmp/doin_<node>.log`
- **GPU**: `nvidia-smi` — 50-95% during training, drops briefly between candidates
- **Peers**: `curl -s http://localhost:8470/api/peers`

### Joining a Running Optimization

To add a new node to an existing experiment (nodes already running, blockchain has blocks):

1. **Ensure `reset_chain: true`** in your config — the new node starts with a fresh local chain and syncs from peers.
2. **Set `bootstrap_peers`** to at least one running node's address.
3. **Set environment variables** (all three — see Prerequisites step 3).
4. **Launch normally**:

```bash
conda activate tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
# Set LD_LIBRARY_PATH if needed (see Prerequisites step 3)

cd ~/Documents/GitHub/doin-node
nohup python -m doin_node.cli \
  --config examples/predictor_dragon_node_tcn_neat.json \
  > /tmp/doin_dragon.log 2>&1 &
```

The new node will:
- Connect to peers and sync the blockchain
- Import the current best champion from the chain into its population
- Start its own NEAT optimization, contributing candidates to the network
- Receive future champion broadcasts from other nodes

### Updating a Running Node

> **Warning — version mismatch**: DOIN nodes enforce strict version checking. If ANY of the 4 repositories (doin-core, doin-node, doin-plugins, predictor) differ by even one commit between nodes, ALL inter-node messages are rejected with `🚫 Rejecting ... version mismatch`. After pulling code, you **must update and restart ALL running nodes** — not just the one you changed.

```bash
# 1. Pull latest code on EVERY machine (not just one)
cd ~/Documents/GitHub/predictor && git pull && pip install -e .
cd ~/Documents/GitHub/doin-plugins && git pull && pip install -e .
cd ~/Documents/GitHub/doin-core && git pull && pip install -e .
cd ~/Documents/GitHub/doin-node && git pull && pip install -e .

# 2. Kill the running node
pgrep -f doin_node.cli
kill <PID>   # or: python3 -c "import os,signal; os.kill(<PID>, signal.SIGKILL)"

# 3. Relaunch with ALL environment variables
conda activate tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
# Set LD_LIBRARY_PATH if needed (see Prerequisites step 3)

cd ~/Documents/GitHub/doin-node
nohup python -m doin_node.cli \
  --config examples/<node_config>.json \
  > /tmp/doin_<node>.log 2>&1 &
```

Repeat on **every machine** before any node will communicate with the others.

### SSH Launch (Remote Machines)

For machines accessible only via SSH, use this one-liner pattern. It activates conda, sets all GPU environment variables, and launches in the background:

```bash
ssh -p 62024 harveybc@<IP> 'source /home/harveybc/anaconda3/etc/profile.d/conda.sh && \
  conda activate tensorflow && \
  NB=/home/harveybc/anaconda3/envs/tensorflow/lib/python3.12/site-packages/nvidia && \
  export LD_LIBRARY_PATH="${NB}/cudnn/lib:${NB}/cublas/lib:${NB}/cuda_runtime/lib:${NB}/cufft/lib:${NB}/curand/lib:${NB}/cusolver/lib:${NB}/cusparse/lib:${NB}/cuda_cupti/lib:${NB}/nvjitlink/lib:${NB}/cuda_nvrtc/lib:${NB}/nccl/lib" && \
  export TF_FORCE_GPU_ALLOW_GROWTH=true && \
  export TF_GPU_ALLOCATOR=cuda_malloc_async && \
  cd /home/harveybc/Documents/GitHub/doin-node && \
  nohup python -m doin_node.cli --config examples/<config>.json \
  > /tmp/doin_<node>.log 2>&1 & echo "PID: $!"'
```

### 3-Node LAN Config Files

Ready-to-use configurations for a 3-node setup:

| Config | Machine | IP | GPU | VRAM | `node_seed_offset` |
|--------|---------|-----|-----|------|-------------------|
| `predictor_omega_node_tcn_neat.json` | Omega | 192.168.0.105 | RTX 4070 Laptop | 8 GB | 0 |
| `predictor_dragon_node_tcn_neat.json` | Dragon | 192.168.0.107 | RTX 4090 Laptop | 16 GB | 1 |
| `predictor_gamma_node_tcn_neat.json` | Gamma | 192.168.0.106 | RTX 5070 Ti Laptop | 12 GB | 2 |

Each config references the predictor optimization config at `examples/config/phase_1_daily/optimization/phase_1_tcn_neat_1d_optimization_config.json` within the predictor repo. The `predictor_root` field must point to the local predictor repository path on each machine.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-smi` shows 0% GPU | `LD_LIBRARY_PATH` not set, CUDA libs not found | Set `LD_LIBRARY_PATH` to nvidia pip package dirs (see Prerequisites step 3) |
| `Skipping registering GPU devices` in log | Same — TF can't load CUDA shared objects | Same fix — set `LD_LIBRARY_PATH` |
| `TF_FORCE_GPU_ALLOW_GROWTH ... could not be parsed: "1"` | Used `=1` instead of `=true` | Change to `export TF_FORCE_GPU_ALLOW_GROWTH=true` |
| `🚫 Rejecting ... version mismatch` | Nodes running different code versions | Pull + reinstall + restart ALL nodes (every repo, every machine) |
| `OOM when allocating tensor` | GPU memory exhausted by parent process | Set `TF_FORCE_GPU_ALLOW_GROWTH=true` BEFORE launching |
| `Cannot connect to host X:8470` at startup | Other nodes not running yet | Normal during startup — discovery retries automatically |
| `GPUs: []` from `tf.config.list_physical_devices` | CUDA not installed or `LD_LIBRARY_PATH` missing | Run the GPU verification from Prerequisites step 4 |
| JIT compilation takes 30 min on first launch | Blackwell GPU (RTX 50xx, compute 12.x) | Normal — PTX → native compilation. Only on first run per kernel shape |

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

See the [Running a TCN NEAT Experiment](#running-a-tcn-neat-experiment) section above for ready-to-use configs:
- `predictor_omega_node_tcn_neat.json` — Omega (RTX 4070 Laptop)
- `predictor_gamma_node_tcn_neat.json` — Gamma (RTX 5070 Ti Laptop)
- `predictor_dragon_node_tcn_neat.json` — Dragon (RTX 5070 Ti)

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

### Importing Blockchain Data to Metabase

The `chain.db` SQLite database produced by each node can be imported directly into [Metabase](https://www.metabase.com/) for visualization and analysis. An example database from a Phase 1 TCN+NEAT optimization run is included in `examples/results/phase_1_daily/blockchain/`.

**Quick Start:**

1. **Install Metabase** (Docker is simplest):
   ```bash
   docker run -d -p 3000:3000 --name metabase metabase/metabase
   ```

2. **Copy the blockchain database** to a known path:
   ```bash
   cp examples/results/phase_1_daily/blockchain/chain.db /tmp/chain.db
   ```

3. **Add SQLite database in Metabase**:
   - Open `http://localhost:3000` → Admin → Databases → Add database
   - Type: **SQLite**
   - Filename: `/tmp/chain.db`

4. **Create questions/dashboards** using these tables:
   - `blocks` — One row per accepted block (block_index, timestamp, weighted_performance_sum, generator_id)
   - `transactions` — Two rows per block: summary (tx_index=0) and verified result (tx_index=1). The payload JSON at tx_index=1 contains `verified_performance`, `val_mae`, `test_mae`, `train_mae`, and all naive baselines
   - `peers` — Network peer registry

5. **Example queries**:
   ```sql
   -- Performance progression across blocks
   SELECT b.block_index, b.timestamp, b.weighted_performance_sum,
          json_extract(t.payload, '$.val_mae') as val_mae,
          json_extract(t.payload, '$.test_mae') as test_mae,
          json_extract(t.payload, '$.val_naive_mae') as val_naive_mae
   FROM blocks b
   JOIN transactions t ON t.block_index = b.block_index AND t.tx_index = 1
   WHERE b.block_index > 0
   ORDER BY b.block_index;
   ```

A pre-exported flat CSV (`optimization_metrics.csv`) is also included for tools that don't support SQLite directly.

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
