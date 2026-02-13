# DON Node

**P2P node for the Decentralized Optimization Network (DON)**

Handles peer-to-peer networking, controlled flooding, blockchain management, optimae validation, and block generation via proof of optimization.

## Features

- **Controlled Flooding** — Messages propagate to neighbors with TTL-based hop limiting and dedup
- **Blockchain** — Append-only chain with validation, persistence, and Merkle proofs
- **Optimae Validation** — Coordinates with evaluators to verify reported performance
- **Block Generation** — Triggered by proof-of-optimization consensus threshold
- **HTTP Transport** — Simple, debuggable HTTP-based peer communication

## Installation

```bash
pip install -e ".[dev]"
```

Requires `doin-core` to be installed first.

## Usage

```bash
# Start a node
doin-node --port 8470 --data-dir ./node-data --peers 192.168.1.10:8470

# With domain configuration
doin-node --port 8470 --domains domains.json --target-block-time 300

# Full options
doin-node --help
```

### Domain Configuration (domains.json)

```json
[
    {
        "id": "predictor-v1",
        "name": "Time Series Predictor",
        "performance_metric": "mse",
        "higher_is_better": false,
        "weight": 1.0,
        "config": {
            "optimization_plugin": "genetic_optimizer",
            "inference_plugin": "keras_predictor",
            "synthetic_data_plugin": "timeseries_generator"
        }
    }
]
```

## Architecture

```
doin-node/
├── network/
│   ├── peer.py          # Peer state management
│   ├── flooding.py      # Controlled flooding with TTL + dedup
│   └── transport.py     # HTTP-based message transport
├── blockchain/
│   └── chain.py         # Chain storage, validation, persistence
├── validation/
│   └── validator.py     # Optimae verification coordination
├── node.py              # Main node orchestrator
└── cli.py               # Command-line entry point
```

## License

MIT
