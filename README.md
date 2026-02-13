# doin-node

Unified DOIN node — single configurable process that can optimize, evaluate, and relay.

## What is DOIN?

DOIN (Decentralized Optimization and Inference Network) is a blockchain-based system where nodes collaboratively optimize ML models. Block generation is triggered by verified optimization work — **Proof of Optimization**.

## This Package

`doin-node` is the main entry point for running a DOIN node. Like a Bitcoin node that can mine + validate + relay, a DOIN node can optimize + evaluate + relay — all configurable per domain via JSON.

### Features
- **Unified architecture** — single process, configurable roles per domain
- **HTTP transport** — aiohttp-based, simple and debuggable
- **Controlled flooding** — TTL-based message propagation with dedup
- **Block sync protocol** — initial sync on startup, catch-up on announcements
- **Full security pipeline** — all 10 hardening measures wired in
- **Pull-based task queue** — evaluators poll for work, priority-ordered

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

See [INSTALL.md](https://github.com/harveybc/doin-node/blob/master/docs/INSTALL.md) for configuration examples.

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

## Tests

```bash
python -m pytest tests/ -v
# 83 tests (including multi-node network tests)
```

## Part of DOIN

- [doin-core](https://github.com/harveybc/doin-core) — Consensus, models, crypto
- [doin-node](https://github.com/harveybc/doin-node) — This package
- [doin-plugins](https://github.com/harveybc/doin-plugins) — Domain plugins
