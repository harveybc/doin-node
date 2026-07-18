"""CLI entry point for launching a unified DON node.

Usage:
    doin-node --config node_config.json
    doin-node --config node_config.json --port 8471
    doin-node --config node_config.json --peers 192.168.1.10:8470,192.168.1.11:8470

Environment variables:
    DON_DATA_DIR:   Override data directory
    DON_PORT:       Override listening port
    DON_PEERS:      Comma-separated bootstrap peers
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import signal

# Ensure TensorFlow uses memory growth so worker subprocesses can share GPU
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import sys
from pathlib import Path
from typing import Any

from doin_core.crypto.identity import PeerIdentity
from doin_core.consensus import IncentiveConfig
from doin_core.models import ResourceLimits
from doin_core.models.fee_market import FeeConfig
from doin_core.plugins.loader import (
    load_inference_plugin,
    load_optimization_plugin,
    load_synthetic_data_plugin,
)

from doin_node.unified import (
    DomainRole,
    UnifiedNode,
    UnifiedNodeConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a unified DON node (optimizer + evaluator + relay)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Override listening port",
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="Override data directory",
    )
    parser.add_argument(
        "--peers",
        help="Comma-separated bootstrap peer addresses (host:port)",
    )
    parser.add_argument(
        "--identity", "-i",
        help="Path to identity key file (generated if missing)",
    )
    parser.add_argument(
        "--stats-file",
        help="Path to experiment stats CSV file (overrides config)",
    )
    parser.add_argument(
        "--olap-db",
        help="Path to OLAP SQLite database (overrides config)",
    )
    parser.add_argument(
        "--reset-chain",
        action="store_true",
        default=False,
        help="Delete existing chain database before starting (fresh genesis)",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


def _parse_param_bounds(
    raw: object, *, domain_id: str
) -> dict[str, tuple[float, float]]:
    """Parse and validate ``param_bounds`` into ``{name: (low, high)}`` tuples.

    Each entry must be a two-element list/tuple of finite, non-boolean numbers
    with ``low <= high``.  The original numeric types are preserved.  Errors
    name the offending domain and parameter and never mutate the input.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"domain '{domain_id}': 'param_bounds' must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"domain '{domain_id}': param_bounds['{name}'] must be a "
                f"[low, high] pair, got {value!r}"
            )
        low, high = value
        for bound in (low, high):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(
                    f"domain '{domain_id}': param_bounds['{name}'] must contain "
                    f"two numbers, got {value!r}"
                )
            if not math.isfinite(bound):
                raise ValueError(
                    f"domain '{domain_id}': param_bounds['{name}'] must be "
                    f"finite, got {value!r}"
                )
        if low > high:
            raise ValueError(
                f"domain '{domain_id}': param_bounds['{name}'] lower bound "
                f"{low} exceeds upper bound {high}"
            )
        bounds[name] = (low, high)
    return bounds


def _parse_resource_limits(raw: object, *, domain_id: str) -> ResourceLimits:
    """Build ``ResourceLimits`` from a JSON section, defaulting when absent."""
    if raw is None:
        return ResourceLimits()
    if not isinstance(raw, dict):
        raise ValueError(
            f"domain '{domain_id}': 'resource_limits' must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(ResourceLimits.model_fields))
    if unknown:
        raise ValueError(
            f"domain '{domain_id}': invalid 'resource_limits' section: "
            f"unknown field(s): {', '.join(unknown)}"
        )
    try:
        return ResourceLimits(**raw)
    except Exception as e:
        raise ValueError(
            f"domain '{domain_id}': invalid 'resource_limits' section: {e}"
        ) from e


def _parse_incentive_config(raw: object, *, domain_id: str) -> IncentiveConfig:
    """Build ``IncentiveConfig`` from a JSON section, defaulting when absent.

    Direction/tolerance semantics are those of the dataclass — this only maps
    the values through, it never reinterprets them.
    """
    if raw is None:
        return IncentiveConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"domain '{domain_id}': 'incentive_config' must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    try:
        return IncentiveConfig(**raw)
    except Exception as e:
        raise ValueError(
            f"domain '{domain_id}': invalid 'incentive_config' section: {e}"
        ) from e


def _parse_fee_config(raw: object) -> FeeConfig:
    """Build ``FeeConfig`` from a JSON section, defaulting when absent."""
    if raw is None:
        return FeeConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"'fee_config' must be a JSON object, got {type(raw).__name__}"
        )
    try:
        return FeeConfig(**raw)
    except Exception as e:
        raise ValueError(f"invalid 'fee_config' section: {e}") from e


def _select_identity_path(
    cli_identity: str | None,
    configured_identity: str,
) -> str | None:
    """Resolve the identity path: CLI override, then JSON, then data-dir default.

    Returns ``None`` when neither an explicit CLI nor a configured path is set,
    signalling the caller to use the existing data-directory default.
    """
    if cli_identity:
        return cli_identity
    if configured_identity:
        return configured_identity
    return None


def load_config(config_path: str, overrides: dict[str, Any]) -> UnifiedNodeConfig:
    """Load and validate node configuration from JSON."""
    path = Path(config_path).resolve()
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = json.load(f)

    # Apply CLI overrides
    if overrides.get("port"):
        raw["port"] = overrides["port"]
    if overrides.get("data_dir"):
        raw["data_dir"] = overrides["data_dir"]
    if overrides.get("peers"):
        raw["bootstrap_peers"] = overrides["peers"]
    if overrides.get("stats_file"):
        raw["experiment_stats_file"] = overrides["stats_file"]
    if overrides.get("olap_db"):
        raw["olap_db_path"] = overrides["olap_db"]

    # Parse domain roles. Plugin config subtrees are deeply copied so nested
    # lists/dicts cannot leak mutations across configuration boundaries (R12).
    domain_roles = []
    for d in raw.get("domains", []):
        domain_id = d["domain_id"]
        domain_roles.append(DomainRole(
            domain_id=domain_id,
            optimize=d.get("optimize", False),
            evaluate=d.get("evaluate", False),
            optimization_plugin=d.get("optimization_plugin", ""),
            optimization_config=copy.deepcopy(d.get("optimization_config") or {}),
            inference_plugin=d.get("inference_plugin", ""),
            synthetic_data_plugin=d.get("synthetic_data_plugin", ""),
            has_synthetic_data=d.get("has_synthetic_data", False),
            synthetic_data_validation=d.get("synthetic_data_validation", True),
            higher_is_better=d.get("higher_is_better", True),
            inference_config=copy.deepcopy(d.get("inference_config") or {}),
            synthetic_data_config=copy.deepcopy(d.get("synthetic_data_config") or {}),
            metric_type=d.get("metric_type", ""),
            param_bounds=_parse_param_bounds(
                d.get("param_bounds"), domain_id=domain_id
            ),
            resource_limits=_parse_resource_limits(
                d.get("resource_limits"), domain_id=domain_id
            ),
            incentive_config=_parse_incentive_config(
                d.get("incentive_config"), domain_id=domain_id
            ),
            target_performance=d.get("target_performance"),
        ))

    # Reference dataclass defaults for absent fields rather than duplicating
    # guessed literals, so loading {} reproduces UnifiedNodeConfig() exactly.
    defaults = UnifiedNodeConfig()
    return UnifiedNodeConfig(
        node_label=raw.get("node_label", defaults.node_label),
        host=raw.get("host", defaults.host),
        port=raw.get("port", defaults.port),
        data_dir=raw.get("data_dir", defaults.data_dir),
        identity_file=raw.get("identity_file", defaults.identity_file),
        bootstrap_peers=raw.get("bootstrap_peers", defaults.bootstrap_peers),
        domains=domain_roles,
        target_block_time=raw.get("target_block_time", defaults.target_block_time),
        initial_threshold=raw.get("initial_threshold", defaults.initial_threshold),
        acceptance_tolerance=raw.get("acceptance_tolerance", defaults.acceptance_tolerance),
        quorum_min_evaluators=raw.get("quorum_min_evaluators", defaults.quorum_min_evaluators),
        quorum_fraction=raw.get("quorum_fraction", defaults.quorum_fraction),
        quorum_tolerance=raw.get("quorum_tolerance", defaults.quorum_tolerance),
        commit_reveal_max_age=raw.get("commit_reveal_max_age", defaults.commit_reveal_max_age),
        finality_confirmation_depth=raw.get("finality_confirmation_depth", defaults.finality_confirmation_depth),
        external_anchor_interval=raw.get("external_anchor_interval", defaults.external_anchor_interval),
        require_deterministic_seed=raw.get("require_deterministic_seed", defaults.require_deterministic_seed),
        eval_poll_interval=raw.get("eval_poll_interval", defaults.eval_poll_interval),
        eval_max_concurrent=raw.get("eval_max_concurrent", defaults.eval_max_concurrent),
        optimizer_loop_interval=raw.get("optimizer_loop_interval", defaults.optimizer_loop_interval),
        shared_claim_timeout=raw.get("shared_claim_timeout", defaults.shared_claim_timeout),
        shared_claim_result_patience=raw.get(
            "shared_claim_result_patience", defaults.shared_claim_result_patience
        ),
        shared_min_peers=raw.get("shared_min_peers", defaults.shared_min_peers),
        shared_initialize_before_peers=raw.get(
            "shared_initialize_before_peers",
            defaults.shared_initialize_before_peers,
        ),
        shared_peer_wait_timeout=raw.get(
            "shared_peer_wait_timeout", defaults.shared_peer_wait_timeout
        ),
        shared_claim_settle_seconds=raw.get(
            "shared_claim_settle_seconds", defaults.shared_claim_settle_seconds
        ),
        shared_claim_confirmation_rounds=raw.get(
            "shared_claim_confirmation_rounds",
            defaults.shared_claim_confirmation_rounds,
        ),
        storage_backend=raw.get("storage_backend", defaults.storage_backend),
        db_path=raw.get("db_path", defaults.db_path),
        snapshot_interval=raw.get("snapshot_interval", defaults.snapshot_interval),
        prune_keep_blocks=raw.get("prune_keep_blocks", defaults.prune_keep_blocks),
        network_protocol=raw.get("network_protocol", defaults.network_protocol),
        gossip_heartbeat_interval=raw.get("gossip_heartbeat_interval", defaults.gossip_heartbeat_interval),
        discovery_enabled=raw.get("discovery_enabled", defaults.discovery_enabled),
        discovery_interval=raw.get("discovery_interval", defaults.discovery_interval),
        dashboard_enabled=raw.get("dashboard_enabled", defaults.dashboard_enabled),
        experiment_stats_file=raw.get("experiment_stats_file", defaults.experiment_stats_file),
        olap_db_path=raw.get("olap_db_path", defaults.olap_db_path),
        reset_chain=raw.get("reset_chain", defaults.reset_chain),
        fee_market_enabled=raw.get("fee_market_enabled", defaults.fee_market_enabled),
        fee_config=_parse_fee_config(raw.get("fee_config")),
    )


def load_identity(identity_path: str | None, data_dir: str) -> PeerIdentity:
    """Load or generate node identity."""
    if identity_path:
        path = Path(identity_path)
    else:
        path = Path(data_dir) / "identity.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    identity = PeerIdentity.load_or_generate(path)
    print(f"  Identity: {identity.peer_id[:16]}...")

    return identity


def setup_plugins(node: UnifiedNode) -> None:
    """Load and register domain plugins for the node.

    Uses setuptools entry points to discover plugins by name.
    """
    _log = logging.getLogger(__name__)
    for domain_id, role in node._domain_roles.items():
        # Optimization plugin — receives an isolated copy of optimization_config.
        if role.optimize and role.optimization_plugin:
            try:
                cls = load_optimization_plugin(role.optimization_plugin)
                plugin = cls()
                opt_cfg = copy.deepcopy(role.optimization_config)
                # Domain metric_type only when the optimizer subtree lacks one.
                if role.metric_type and "metric_type" not in opt_cfg:
                    opt_cfg["metric_type"] = role.metric_type
                # Merge domain-level param_bounds into config so optimizer can find them
                if role.param_bounds and "param_bounds" not in opt_cfg and "hyperparameter_bounds" not in opt_cfg:
                    opt_cfg["param_bounds"] = {k: list(v) for k, v in role.param_bounds.items()}
                plugin.configure(opt_cfg)
                node.register_optimizer_plugin(domain_id, plugin)
                _log.info("Optimizer plugin '%s' loaded for %s", role.optimization_plugin, domain_id)
            except Exception as e:
                _log.error("Could not load optimizer plugin '%s': %s", role.optimization_plugin, e, exc_info=True)

        # Inference plugin — its own subtree, falling back to optimization_config.
        if role.evaluate and role.inference_plugin:
            try:
                cls = load_inference_plugin(role.inference_plugin)
                plugin = cls()
                inf_source = role.inference_config if role.inference_config else role.optimization_config
                inf_cfg = copy.deepcopy(inf_source)
                # Domain metric_type only when the inference subtree lacks one.
                if role.metric_type and "metric_type" not in inf_cfg:
                    inf_cfg["metric_type"] = role.metric_type
                plugin.configure(inf_cfg)
                node.register_evaluator_plugin(domain_id, plugin)
                _log.info("Evaluator plugin '%s' loaded for %s", role.inference_plugin, domain_id)
            except Exception as e:
                _log.error("Could not load evaluator plugin '%s': %s", role.inference_plugin, e, exc_info=True)

        # Synthetic data plugin (MANDATORY for verification trust) — its own
        # subtree, falling back to optimization_config. No metric_type/bounds injection.
        if role.evaluate and role.synthetic_data_plugin:
            try:
                cls = load_synthetic_data_plugin(role.synthetic_data_plugin)
                plugin = cls()
                syn_source = role.synthetic_data_config if role.synthetic_data_config else role.optimization_config
                plugin.configure(copy.deepcopy(syn_source))
                node.register_synthetic_plugin(domain_id, plugin)
                print(f"  Synthetic data plugin '{role.synthetic_data_plugin}' loaded for {domain_id}")
            except Exception as e:
                print(f"  WARNING: Could not load synthetic data plugin '{role.synthetic_data_plugin}': {e}")

        if role.evaluate and not role.has_synthetic_data:
            print(f"  ⚠️  Domain {domain_id} has NO synthetic data — "
                  f"optimae will have ZERO consensus weight!")


async def run_node(node: UnifiedNode) -> None:
    """Start the node and run until interrupted."""
    await node.start()

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def handle_signal() -> None:
        print("\nShutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await stop_event.wait()
    await node.stop()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("  DON Unified Node")
    print("=" * 60)

    # Load config
    overrides = {}
    if args.port:
        overrides["port"] = args.port
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.peers:
        overrides["peers"] = [p.strip() for p in args.peers.split(",")]
    if args.stats_file:
        overrides["stats_file"] = args.stats_file
    if args.olap_db:
        overrides["olap_db"] = args.olap_db

    config = load_config(args.config, overrides)
    print(f"  Config loaded: {args.config}")
    print(f"  Port: {config.port}")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Domains: {len(config.domains)}")

    # Load identity — precedence: CLI --identity, then JSON identity_file,
    # then the existing data-directory default.
    identity_path = _select_identity_path(args.identity, config.identity_file)
    identity = load_identity(identity_path, config.data_dir)

    # Reset chain if requested (via CLI flag or config option)
    if args.reset_chain or config.reset_chain:
        data_path = Path(config.data_dir)
        chain_files = (list(data_path.glob("chain.db*"))
                       + list(data_path.glob("chain.json"))
                       + list(data_path.glob("olap.db*")))
        if chain_files:
            for cf in chain_files:
                cf.unlink()
                print(f"  Deleted: {cf}")
            print(f"  Chain reset complete — {len(chain_files)} file(s) removed")
        else:
            print("  No chain/olap files found to reset")

    # Create node
    node = UnifiedNode(config, identity)

    # Load plugins
    print("\nLoading plugins...")
    setup_plugins(node)

    print(f"\n  Optimizer domains: {node.optimizer_domains}")
    print(f"  Evaluator domains: {node.evaluator_domains}")
    print(f"  Bootstrap peers: {config.bootstrap_peers}")

    # Security summary
    print("\n  Security hardening:")
    print(f"    Commit-reveal: max_age={config.commit_reveal_max_age}s")
    print(f"    Quorum: {config.quorum_min_evaluators} evaluators, "
          f"{config.quorum_fraction:.0%} agreement, "
          f"{config.quorum_tolerance:.0%} tolerance")
    print(f"    Finality: {config.finality_confirmation_depth} confirmations")
    print(f"    External anchoring: every {config.external_anchor_interval} blocks")
    print(f"    Deterministic seed: {config.require_deterministic_seed}")

    print("\n" + "=" * 60)
    print(f"  Starting node on :{config.port}...")
    print("=" * 60 + "\n")

    # Run
    asyncio.run(run_node(node))


if __name__ == "__main__":
    main()
