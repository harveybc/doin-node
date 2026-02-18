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
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from doin_core.crypto.identity import PeerIdentity
from doin_core.plugins.loader import (
    load_inference_plugin,
    load_optimization_plugin,
    load_synthetic_data_plugin,
)

from doin_node.unified import DomainRole, ResourceLimits, UnifiedNode, UnifiedNodeConfig


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
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


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

    # Parse domain roles
    domain_roles = []
    for d in raw.get("domains", []):
        rl = d.get("resource_limits", {})
        domain_roles.append(DomainRole(
            domain_id=d["domain_id"],
            optimize=d.get("optimize", False),
            evaluate=d.get("evaluate", False),
            optimization_plugin=d.get("optimization_plugin", ""),
            optimization_config=d.get("optimization_config", {}),
            inference_plugin=d.get("inference_plugin", ""),
            synthetic_data_plugin=d.get("synthetic_data_plugin", ""),
            has_synthetic_data=d.get("has_synthetic_data", False),
            param_bounds={
                k: tuple(v) for k, v in d.get("param_bounds", {}).items()
            },
            resource_limits=ResourceLimits(**rl) if rl else ResourceLimits(),
            target_performance=d.get("target_performance"),
        ))

    return UnifiedNodeConfig(
        host=raw.get("host", "0.0.0.0"),
        port=raw.get("port", 8470),
        data_dir=raw.get("data_dir", "./don-data"),
        bootstrap_peers=raw.get("bootstrap_peers", []),
        domains=domain_roles,
        target_block_time=raw.get("target_block_time", 600.0),
        initial_threshold=raw.get("initial_threshold", 1.0),
        quorum_min_evaluators=raw.get("quorum_min_evaluators", 3),
        quorum_fraction=raw.get("quorum_fraction", 0.67),
        quorum_tolerance=raw.get("quorum_tolerance", 0.05),
        commit_reveal_max_age=raw.get("commit_reveal_max_age", 600.0),
        finality_confirmation_depth=raw.get("finality_confirmation_depth", 6),
        external_anchor_interval=raw.get("external_anchor_interval", 100),
        require_deterministic_seed=raw.get("require_deterministic_seed", True),
        eval_poll_interval=raw.get("eval_poll_interval", 10.0),
        eval_max_concurrent=raw.get("eval_max_concurrent", 3),
        optimizer_loop_interval=raw.get("optimizer_loop_interval", 30.0),
        experiment_stats_file=raw.get("experiment_stats_file", ""),
        olap_db_path=raw.get("olap_db_path", ""),
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
    for domain_id, role in node._domain_roles.items():
        # Optimization plugin
        if role.optimize and role.optimization_plugin:
            try:
                cls = load_optimization_plugin(role.optimization_plugin)
                plugin = cls()
                plugin.configure(role.optimization_config)
                node.register_optimizer_plugin(domain_id, plugin)
                print(f"  Optimizer plugin '{role.optimization_plugin}' loaded for {domain_id}")
            except Exception as e:
                print(f"  WARNING: Could not load optimizer plugin '{role.optimization_plugin}': {e}")

        # Inference plugin
        if role.evaluate and role.inference_plugin:
            try:
                cls = load_inference_plugin(role.inference_plugin)
                plugin = cls()
                plugin.configure(role.optimization_config)  # Same config
                node.register_evaluator_plugin(domain_id, plugin)
                print(f"  Evaluator plugin '{role.inference_plugin}' loaded for {domain_id}")
            except Exception as e:
                print(f"  WARNING: Could not load evaluator plugin '{role.inference_plugin}': {e}")

        # Synthetic data plugin (MANDATORY for verification trust)
        if role.evaluate and role.synthetic_data_plugin:
            try:
                cls = load_synthetic_data_plugin(role.synthetic_data_plugin)
                plugin = cls()
                plugin.configure(role.optimization_config)
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

    # Load identity
    identity = load_identity(args.identity, config.data_dir)

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
