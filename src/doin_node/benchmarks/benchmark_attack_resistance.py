"""Attack Resistance Benchmark — tests malicious node detection and penalization.

Runs N honest nodes plus a "malicious" node that reports inflated performance.
Measures how the network's reputation system detects and penalizes the attacker
vs honest participants.

Usage:
    python -m doin_node.benchmarks.benchmark_attack_resistance [--honest 3] [--duration 300]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from .common import (
    NodeProcess,
    create_node_cluster,
    poll_status,
    start_cluster,
    stop_cluster,
    wait_for_node,
)


class MaliciousNodeProcess(NodeProcess):
    """A node with a tampered optimizer that reports fake performance.

    We achieve this by creating a modified config that uses a custom
    optimization_config with a huge fake performance boost. The quadratic
    optimizer's performance is -MSE, so we'd need to intercept it.

    For benchmarking, we use a simpler approach: launch a real node but
    with a modified plugin config that has a very wrong target, causing
    its reported performance to be artificially low when verified by
    honest evaluators (the optimizer thinks it's doing great on its
    wrong target, but evaluators use the correct synthetic data).
    """

    def generate_config(self, extra: dict[str, Any] | None = None) -> str:
        """Generate a config with a deliberately wrong optimization target."""
        extra = extra or {}
        # The malicious node optimizes for a DIFFERENT target than what
        # the synthetic data plugin generates. This means:
        # - The optimizer reports good performance (on its fake target)
        # - Evaluators verify against correct synthetic data → low performance
        # - The quorum rejects → reputation penalty
        extra.setdefault("domains", None)

        from .common import SINGLE_CONFIG
        with open(SINGLE_CONFIG) as f:
            config = json.load(f)

        config["port"] = self.port
        config["data_dir"] = self.data_dir
        config["bootstrap_peers"] = self.bootstrap_peers

        # Tamper: give optimizer a wrong target
        for domain in config.get("domains", []):
            domain["optimization_config"] = {
                "n_params": 10,
                "step_size": 0.5,
                "target": [100.0] * 10,  # Way off — real target is in [-5, 5]
            }

        fd, path = tempfile.mkstemp(suffix=".json", prefix=f"bench_malicious_{self.port}_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        self.config_path = path
        return path


async def run_benchmark(
    honest_nodes: int = 3,
    duration: float = 300.0,
    base_port: int = 8490,
) -> dict[str, Any]:
    """Run the attack resistance benchmark."""
    print("=" * 60)
    print("  DOIN Attack Resistance Benchmark")
    print("=" * 60)
    print(f"  Honest nodes: {honest_nodes}, Malicious: 1")
    print(f"  Duration: {duration}s")

    total_nodes = honest_nodes + 1  # +1 malicious

    # Create honest nodes
    nodes = create_node_cluster(honest_nodes, base_port=base_port)

    # Create malicious node
    malicious_port = base_port + honest_nodes
    malicious_dir = tempfile.mkdtemp(prefix="bench_malicious_")
    malicious = MaliciousNodeProcess(
        port=malicious_port,
        data_dir=malicious_dir,
        bootstrap_peers=[f"localhost:{base_port}"],
    )
    malicious.generate_config()

    all_nodes = nodes + [malicious]

    results: dict[str, Any] = {
        "benchmark": "attack_resistance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "honest_nodes": honest_nodes,
            "malicious_nodes": 1,
            "duration": duration,
        },
        "malicious_port": malicious_port,
        "samples": [],
        "final": {},
    }

    try:
        # Start all nodes
        print("\n  Starting nodes...")
        await start_cluster(all_nodes, stagger=1.5)

        # Monitor
        start_time = time.time()
        poll_interval = 10.0

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            sample = {"elapsed": round(elapsed, 1), "nodes": []}

            for node in all_nodes:
                status = await poll_status(node.endpoint)
                is_malicious = node.port == malicious_port
                node_info = {
                    "port": node.port,
                    "malicious": is_malicious,
                    "alive": node.is_alive(),
                    "chain_height": status.get("chain_height", 0) if status else 0,
                    "reputation": status.get("reputation", {}) if status else {},
                    "pending_tasks": status.get("pending_tasks", 0) if status else 0,
                }
                sample["nodes"].append(node_info)

            results["samples"].append(sample)

            # Progress
            alive_count = sum(1 for n in all_nodes if n.is_alive())
            print(f"    [{elapsed:.0f}s] {alive_count}/{total_nodes} alive", end="")

            # Show heights
            heights = [s.get("chain_height", "?") for s in sample["nodes"]]
            print(f", heights={heights}")

            await asyncio.sleep(poll_interval)

        # Final collection
        print("\n  Collecting final state...")
        final_states = []
        for node in all_nodes:
            status = await poll_status(node.endpoint)
            is_malicious = node.port == malicious_port
            final_states.append({
                "port": node.port,
                "malicious": is_malicious,
                "alive": node.is_alive(),
                "status": status,
            })

        results["final"] = {
            "states": final_states,
        }

        # Analyze: did the malicious node get penalized?
        # In the current architecture, the reputation penalty happens when
        # quorum rejects the malicious node's optimae
        print("\n" + "=" * 60)
        print("  Attack Resistance Analysis")
        print("=" * 60)

        for state in final_states:
            label = "MALICIOUS" if state["malicious"] else "honest"
            status = state.get("status") or {}
            height = status.get("chain_height", "?")
            rep = status.get("reputation", {})
            print(f"  [{label}] port={state['port']} height={height} reputation={rep}")

    finally:
        stop_cluster(all_nodes)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DOIN Attack Resistance Benchmark")
    parser.add_argument("--honest", type=int, default=3, help="Number of honest nodes")
    parser.add_argument("--duration", type=float, default=300.0, help="Benchmark duration (s)")
    parser.add_argument("--output", "-o", default="attack_resistance_results.json")
    parser.add_argument("--base-port", type=int, default=8490)
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        honest_nodes=args.honest,
        duration=args.duration,
        base_port=args.base_port,
    ))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
