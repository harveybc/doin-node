# AGENTS.md — doin-node

Guidance for AI coding agents working in this repository. See [agents.md](https://agents.md).

## Project overview

doin-node is the runnable participant process of DOIN, the Decentralized
Optimization and Inference Network (older docstrings abbreviate it **DON**). One
process per machine reads one JSON config that selects, per optimization domain,
whether that machine optimizes, evaluates, or only relays; which plugins
implement the domain; and how it joins the peer network. Consensus is
proof-of-optimization: blocks are minted when verified performance improvements
cross a dynamic threshold. The process also serves an HTTP API and a web
dashboard, persists the chain to SQLite or JSON, and records per-round
optimization metrics into an embedded SQLite star schema.

It does **not** define the protocol — consensus rules, data models, the wire
schema, crypto and the plugin ABCs come from
[doin-core](https://github.com/harveybc/doin-core) and are imported. It ships
**no plugins of its own**; implementations come from
[doin-plugins](https://github.com/harveybc/doin-plugins) or any package
registering the `doin.*` entry-point groups. Domain optimizers such as
[predictor](https://github.com/harveybc/predictor) and
[agent-multi](https://github.com/harveybc/agent-multi) stay external and work
standalone without DOIN.

## Agent quickstart (install → run → show the user results)

Verified on 2026-08-16 with Python 3.12.13.

### 1. Environment

`requires-python = ">=3.10"`; CI runs 3.12. Install the three packages in
dependency order — there are no PyPI releases.

```bash
conda create -n doin python=3.12 -y && conda activate doin
git clone https://github.com/harveybc/doin-core.git
git clone https://github.com/harveybc/doin-plugins.git
git clone https://github.com/harveybc/doin-node.git
pip install -e doin-core -e doin-plugins -e doin-node
```

> Unverified: the clone-and-install sequence was not re-executed from scratch
> for this document. Everything below was verified in an existing environment
> that already had the three packages installed.

Confirm the reference plugins are discoverable:

```bash
python -c "
from importlib.metadata import entry_points
print([e.name for e in entry_points(group='doin.optimization')])"
# verified: ['binary_predictor', 'predictor', 'simple_quadratic', 'trading_asset']
```

### 2. Smoke test

```bash
pytest -q
```

Verified: **409 passed in 10.49 s**. The suite is fast and green — treat any
failure as a real regression.

### 3. Representative run: single-node quadratic domain

`examples/quadratic_single_node.json` is the only fully self-contained config:
one process that both optimizes and evaluates the `simple_quadratic` reference
domain, with `discovery_enabled: false`, no bootstrap peers and the fee market
off. Everything else under `examples/` needs external ML packages, absolute
machine-specific paths, or live peers.

```bash
doin-node --config examples/quadratic_single_node.json
# equivalently: python -m doin_node.cli --config examples/quadratic_single_node.json
```

To keep artifacts out of the working tree, redirect them:

```bash
doin-node --config examples/quadratic_single_node.json \
  --data-dir /tmp/doin-demo/data \
  --olap-db  /tmp/doin-demo/olap.db \
  --stats-file /tmp/doin-demo/stats.csv \
  --log-level INFO
```

Verified: the node came up in ~2 s, registered the `quadratic` domain, loaded
the `simple_quadratic` optimizer/evaluator/synthetic plugins, initialized a
chain with a genesis block, created OLAP schema v3, and served HTTP on
`:8470`. Stop it with Ctrl+C.

Useful flags (`doin-node --help`): `--port`, `--data-dir`, `--peers`,
`--identity`, `--stats-file`, `--olap-db`, `--reset-chain`, `--log-level`.
Environment overrides: `DON_DATA_DIR`, `DON_PORT`, `DON_PEERS`.

> **Starting a node kills whatever is already on its port.**
> `Transport.start()` calls `_kill_stale_process_on_port(self.port)`, which
> runs `lsof` and SIGKILLs any process **owned by the same user** listening
> there. Default port is **8470**. Before starting a node, check the port is
> free (`ss -ltn | grep 8470`) and pass `--port` if it is not. Never start a
> node "to see what happens" on a machine that may be running fleet workers.

### 4. Web dashboard and HTTP API

The dashboard is **on by default** (`dashboard_enabled: true`) and shares the
single aiohttp port with the peer protocol and the JSON API. Framework is
aiohttp only — there is no FastAPI, Flask, uvicorn or ASGI server anywhere.

**URL: `http://localhost:8470/dashboard`** — verified HTTP 200,
`<title>DOIN Node Dashboard</title>`.

The page is AdminLTE/Bootstrap/Chart.js loaded from public CDNs, so it needs
internet access to render its styling.

Endpoints verified against a live node:

| Endpoint | Verified response |
|---|---|
| `GET /health` | `{"status": "healthy", "port": 8470}` |
| `GET /status` | peer id, chain height, per-domain optimize/evaluate/target, peers, task queue |
| `GET /chain/status` | tip hash, height, finalized height, component git revisions |
| `GET /stats` | live per-domain experiment record (rounds, best performance, elapsed) |
| `GET /dashboard` | HTML monitoring UI |

Also registered (not individually exercised): `GET /peers`, `GET /fees`,
`GET /chain/blocks`, `GET /chain/block/{index}`, `POST /inference`,
`GET|POST /tasks/*`, `GET /stats/experiments`, `GET /stats/rounds`,
`GET /stats/export`, the `/api/shared/*` coordination API, and the dashboard's
`/api/*` JSON feeds.

> **Known defect: `GET /stats/chain-metrics` always returns HTTP 500.**
> `_http_stats_chain_metrics` at `src/doin_node/unified.py:5178` reads
> `self.consensus.chain.blocks`, but `ProofOfOptimization` has no `chain`
> attribute — verified: `AttributeError: 'ProofOfOptimization' object has no
> attribute 'chain'`. Every call fails. The node holds its chain elsewhere; the
> handler needs the correct reference.

### 5. OLAP

The node writes an embedded SQLite star schema, **schema version 3**, verified
auto-created at the `--olap-db` path on startup:

| Table | Grain |
|---|---|
| `dim_domain` | domain id, type, description |
| `dim_experiment` | experiment id, domain, node id, hostname, optimizer plugin, metric schema, bounds, target, start/finish, `doin_version` |
| `fact_round` | one row per optimization round: performance, best so far, delta, `is_improvement`, parameters, wall clock, chain height, peers, block reward, and per-split MAE columns |
| `fact_experiment_summary` | one row per experiment: totals, best performance, time to target, rounds to convergence |
| `fact_chain_optimae` | one row per accepted on-chain optimae, keyed by `optimae_id` |

**How the cube is actually filled — read this before documenting a pipeline.**
There is no ETL step for the live path. `ExperimentTracker` dual-writes each
round to CSV *and* SQLite as the optimizer runs, so `dim_domain`,
`dim_experiment`, `fact_round` and `fact_experiment_summary` populate
themselves while the node is up. Verified after a short run: `dim_domain` 1 row,
`dim_experiment` 1 row, `fact_experiment_summary` 1 row, `fact_round` 0 rows
(the run was too short to complete a round).

**Chain → OLAP projection exists but is never invoked.**
`OLAPDatabase.ingest_from_chain(blocks)` in `src/doin_node/stats/olap_db.py:293`
walks blocks, keeps `optimae_accepted` transactions and upserts them into
`fact_chain_optimae` with `INSERT OR IGNORE` on `optimae_id`. Verified with
`grep -rn "ingest_from_chain" --include="*.py"`: the only call sites are four
in `tests/test_chain_metrics.py`. **No CLI, no loop and no endpoint calls it**,
so `fact_chain_optimae` stays empty at runtime and there is no "replay blocks
into the cube" command to document. There is also **no reorg handling** in the
OLAP layer — rows are never invalidated when the chain reorganizes. Do not
describe chain projection as a working pipeline; it is library code awaiting a
caller.

To analyze in Metabase, sync SQLite to PostgreSQL:

```bash
pip install psycopg2-binary   # NOT a declared dependency of this package
python -m doin_node.stats.olap_sync \
  --source /tmp/doin-demo/olap.db \
  --target postgresql://<user>:<password>@<your-host>:5432/doin_olap
```

Verified: the CLI exists and its `--source` / `--target` / `--log-level`
interface is as above. **Unverified:** an actual sync was not executed, because
`psycopg2` is not installed by this package and the target database did not
exist. The module recreates the table DDL in PostgreSQL and upserts with
`INSERT ... ON CONFLICT DO UPDATE`, so it is idempotent.

```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

Open `http://localhost:3000`, complete the first-run setup wizard, then
**Add a database**. Two options: point Metabase at the synced PostgreSQL
database, or use Metabase's built-in SQLite driver against the `olap.db` file —
the latter requires volume-mounting the file into the container
(`-v /tmp/doin-demo:/data`) and giving the in-container path. If PostgreSQL also
runs in Docker, put both containers on one user-defined network and use the
container name as the host; `localhost` inside the Metabase container is the
container itself.

*Unverified:* the Metabase container and its wizard were not exercised for this
repository.

### 6. Final message to give the user

> The node's live monitoring UI is at **http://localhost:8470/dashboard**
> (same port serves the JSON API — `http://localhost:8470/status` and
> `http://localhost:8470/stats` are the quickest text views). Files: the chain
> and node state are under the `--data-dir` you passed, per-round metrics are
> in the `--stats-file` CSV, and the analytics cube is the `--olap-db` SQLite
> file. If you synced it to PostgreSQL, Metabase is at
> **http://localhost:3000**.
>
> First query to try — verified optimization progress per domain over time:
>
> ```sql
> SELECT d.domain_id,
>        e.optimizer_plugin,
>        count(*)                  AS improvement_rounds,
>        min(r.timestamp_utc)      AS first_improvement,
>        max(r.timestamp_utc)      AS last_improvement,
>        max(r.best_performance)   AS best_performance
> FROM fact_round r
> JOIN dim_domain     d ON d.domain_id     = r.domain_id
> JOIN dim_experiment e ON e.experiment_id = r.experiment_id
> WHERE r.is_improvement = 1
> GROUP BY d.domain_id, e.optimizer_plugin
> ORDER BY best_performance DESC;
> ```
>
> Drop the `GROUP BY` and select `r.round_number, r.timestamp_utc,
> r.best_performance` to plot the convergence curve per domain.
>
> Two caveats worth repeating: `GET /stats/chain-metrics` is broken and always
> returns 500, and the `fact_chain_optimae` table stays empty because nothing
> calls the chain-ingest function at runtime.

That query was executed against a real schema-v3 database and runs cleanly; it
returned zero rows only because the verification run was too short to record an
improvement round.

## Build, test and lint commands

Local:

```bash
pip install -e .[dev]
pytest -q                  # 409 passed in 10.49s
python -m compileall -q src
python -m doin_node.cli --help
python -m doin_node.stats.olap_sync --help
python -m doin_node.storage.migrate --json chain.json --db chain.db
```

CI runs one workflow, `.github/workflows/tier-a.yml` (job `node-contracts`,
Python 3.12), pinning the sibling repos by commit:

```bash
sha256sum pyproject.toml requirements-ci.txt
git -C .ci/doin-core rev-parse HEAD
git -C .ci/doin-plugins rev-parse HEAD
python -m pip install --require-hashes -r requirements-ci.txt
python -m pip install --no-deps -e .ci/doin-core
python -m pip install --no-deps -e .ci/doin-plugins
python -m pip install --no-deps -e .
python -m compileall -q src
pytest -q
```

`requirements-ci.txt` is a hash-locked lockfile compiled from
`requirements-ci.in`, for CI reproducibility only.

**`ruff` and `mypy` are configured but never run.** `pyproject.toml` sets
`[tool.mypy] strict = true` and `[tool.ruff] target-version = "py310"`, and both
are in the `dev` extra, but no CI step invokes them and `pytest-cov` is never
given a `--cov` flag. Do not claim this codebase is lint-clean, type-clean, or
coverage-measured. Unverified: neither tool was run for this document, so the
pre-existing backlog size is unknown.

## Layout

| Path | Purpose |
|---|---|
| `src/doin_node/unified.py` | `UnifiedNode` and `UnifiedNodeConfig`: role dispatch, optimizer/evaluator/gossip/discovery/maintenance loops, block generation, champion migration, candidate leasing, and all HTTP handlers. The orchestrator, and by far the largest module |
| `src/doin_node/cli.py` | `doin-node` console script: config parsing and validation, plugin loading, lifecycle |
| `src/doin_node/node.py` | Earlier orchestrator, kept for reference; the CLI does not use it |
| `src/doin_node/blockchain/` | In-memory chain state, validation, longest-valid-chain selection |
| `src/doin_node/storage/` | Chain persistence: `chaindb.py` (SQLite via aiosqlite) and `migrate.py` (JSON → SQLite) |
| `src/doin_node/network/` | aiohttp transport, flooding and gossipsub, peer discovery, sync, sharding |
| `src/doin_node/stats/` | OLAP schema, SQLite database, experiment tracker, chain metrics, SQLite→PostgreSQL sync |
| `src/doin_node/dashboard/` | `routes.py` and a single large `templates/dashboard.html`; no static assets |
| `src/doin_node/scheduling/` | GPU/compute scheduling and bidding |
| `src/doin_node/validation/` | `OptimaeValidator`, evaluator verification coordination |
| `src/doin_node/benchmarks/` | Scalability, fault-tolerance and attack-resistance harnesses |
| `examples/` | Node configs; `quadratic_single_node.json` is the only self-contained one |
| `scripts/` | Multi-node and cross-machine test drivers, benchmark runner |
| `tests/` | 409 tests; no `conftest.py` |

## Conventions and constraints

- **One process, one JSON config, roles per domain.** A machine's behaviour is
  entirely in its config: each entry in `domains[]` sets `optimize` /
  `evaluate` and names the plugins. Top-level keys map to `UnifiedNodeConfig`.
- **Protocol comes from doin-core.** Do not redefine models, consensus rules or
  message types here; import them. A protocol change belongs upstream.
- **No plugins here.** The node resolves plugins by name through the
  `doin.optimization`, `doin.inference` and `doin.synthetic_data` entry-point
  groups. Adding a domain means shipping a plugin package, not editing this one.
- **Config defaults differ from the example.** In code, `storage_backend`
  defaults to `sqlite` and `network_protocol` to `gossipsub`;
  `examples/quadratic_single_node.json` overrides them to `json` and `flooding`.
  Read the config, not the defaults, when reasoning about a run.
- **Async throughout** — aiohttp plus asyncio loops; tests use
  `asyncio_mode = "auto"`.
- **`cli.py` sets TensorFlow environment variables at import time**
  (`TF_FORCE_GPU_ALLOW_GROWTH`, `TF_GPU_ALLOCATOR`) for ML domains. Set
  `CUDA_VISIBLE_DEVICES=""` when verifying anything that must not touch a GPU.

## Do not touch

- **Running processes and ports.** GPU training workers and live nodes may be
  running. Starting a node SIGKILLs any same-user process on its port — check
  the port first and use `--port`. Never start, stop or restart fleet workers.
- **`doin-data-*/` directories.** These hold live node state and, critically,
  `identity.pem` files — node **private keys**, mode 0600. Never read, copy,
  print, commit or move them, and never use them as fixtures.
- **`predictor_olap.db` and other root-level `*.db` files.** These are live,
  actively written (`-wal`/`-shm` present) and gitignored. Do not open, copy or
  reset them; create a throwaway database instead.
- **`--reset-chain`** deletes the chain database and OLAP files before starting.
  Never pass it against a real data directory.
- **Operator scripts at the repository root** (the per-machine launch script and
  the log-watch script). They hardcode absolute paths, host addresses and an SSH
  target for particular machines, so they will not work anywhere else. Do not
  run them. The launch script is additionally broken as committed: it references
  `examples/predictor_single_node.json`, which does not exist. Both contain
  machine-identifying details that should not be in a public repository — see
  the note below.
- **Campaign configs under `examples/trading/`.** These require external
  packages and absolute paths, and carry peer addresses. They are not runnable
  demos.
- **Secrets in a public repository.** Never write account identifiers, broker
  credentials, private or overlay IP addresses, SSH targets, or machine host
  names into files here — use placeholders such as `<your-host>`. Note that
  `dim_experiment` carries a `hostname` column, so **OLAP exports and
  `/stats/export` responses can contain machine names**; scrub them before
  sharing a cube or pasting output into an issue.
- **Pre-existing exposure.** Several committed files already contain overlay and
  private LAN addresses, an SSH target on a non-standard port, absolute home
  directories and machine codenames — the root operator scripts and many configs
  under `examples/`. This is a public repository. Do not copy those values into
  new files, issues or commit messages, and flag them to the owner rather than
  propagating them.
- **Sibling repositories.** Protocol changes go to `doin-core`, plugin
  implementations to `doin-plugins`, model code to `predictor` / `agent-multi`.
