"""Fault Tolerance Benchmark — measures recovery after node failures.

Runs N nodes, kills K after a warmup period, measures:
- Whether remaining nodes continue producing blocks
- Time for reconnected nodes to sync back
- Blockchain consistency across all nodes

Usage:
    python -m doin_node.benchmarks.benchmark_fault_tolerance [--nodes 4] [--kill 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def run_benchmark(
    num_nodes: int = 4,
    kill_count: int = 2,
    warmup_duration: float = 120.0,
    failure_duration: float = 60.0,
    recovery_duration: float = 120.0,
    base_port: int = 8480,
) -> dict[str, Any]:
    """Run the fault tolerance benchmark.

    Phases:
    1. Warmup: all N nodes run normally
    2. Failure: kill K nodes, remaining continue
    3. Recovery: restart killed nodes, verify sync
    """
    print("=" * 60)
    print("  DOIN Fault Tolerance Benchmark")
    print("=" * 60)
    print(f"  Nodes: {num_nodes}, Kill: {kill_count}")
    print(f"  Warmup: {warmup_duration}s, Failure: {failure_duration}s, Recovery: {recovery_duration}s")

    nodes = create_node_cluster(num_nodes, base_port=base_port)

    results: dict[str, Any] = {
        "benchmark": "fault_tolerance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "num_nodes": num_nodes,
            "kill_count": kill_count,
            "warmup_duration": warmup_duration,
            "failure_duration": failure_duration,
            "recovery_duration": recovery_duration,
        },
        "phases": {},
    }

    try:
        # Phase 1: Warmup
        print(f"\n  Phase 1: Warmup ({warmup_duration}s)...")
        await start_cluster(nodes)
        await asyncio.sleep(warmup_duration)

        warmup_heights = []
        for node in nodes:
            status = await poll_status(node.endpoint)
            h = status.get("chain_height", 0) if status else 0
            warmup_heights.append(h)
        results["phases"]["warmup"] = {"chain_heights": warmup_heights}
        print(f"    Heights after warmup: {warmup_heights}")

        # Phase 2: Kill K nodes
        print(f"\n  Phase 2: Killing {kill_count} nodes...")
        killed_nodes = nodes[-kill_count:]
        surviving_nodes = nodes[:-kill_count]

        for node in killed_nodes:
            node.stop()
            print(f"    Killed node on port {node.port}")

        pre_failure_heights = []
        for node in surviving_nodes:
            status = await poll_status(node.endpoint)
            h = status.get("chain_height", 0) if status else 0
            pre_failure_heights.append(h)

        print(f"    Waiting {failure_duration}s with {len(surviving_nodes)} surviving nodes...")
        await asyncio.sleep(failure_duration)

        post_failure_heights = []
        for node in surviving_nodes:
            status = await poll_status(node.endpoint)
            h = status.get("chain_height", 0) if status else 0
            post_failure_heights.append(h)

        blocks_during_failure = [
            post_failure_heights[i] - pre_failure_heights[i]
            for i in range(len(surviving_nodes))
        ]
        continued = any(b > 0 for b in blocks_during_failure)

        results["phases"]["failure"] = {
            "pre_failure_heights": pre_failure_heights,
            "post_failure_heights": post_failure_heights,
            "blocks_during_failure": blocks_during_failure,
            "network_continued": continued,
        }
        print(f"    Heights after failure period: {post_failure_heights}")
        print(f"    Blocks produced during failure: {blocks_during_failure}")
        print(f"    Network continued: {continued}")

        # Phase 3: Restart killed nodes
        print(f"\n  Phase 3: Restarting killed nodes...")
        recovery_start = time.time()

        for node in killed_nodes:
            node.start()
            print(f"    Restarted node on port {node.port}")

        # Wait for them to come back
        for node in killed_nodes:
            ready = await wait_for_node(node.endpoint, max_wait=30.0)
            if ready:
                print(f"    Node {node.port} is back online")
            else:
                print(f"    WARNING: Node {node.port} did not come back")

        # Wait for sync
        print(f"    Waiting {recovery_duration}s for recovery...")
        await asyncio.sleep(recovery_duration)

        recovery_time = time.time() - recovery_start

        # Check consistency
        final_heights = []
        for node in nodes:
            status = await poll_status(node.endpoint)
            h = status.get("chain_height", 0) if status else -1
            final_heights.append(h)

        # Check if all nodes converged to similar height
        live_heights = [h for h in final_heights if h >= 0]
        height_spread = max(live_heights) - min(live_heights) if live_heights else -1
        consistent = height_spread <= 2  # Allow 2 block difference

        results["phases"]["recovery"] = {
            "recovery_time": round(recovery_time, 2),
            "final_heights": final_heights,
            "height_spread": height_spread,
            "consistent": consistent,
        }
        print(f"    Final heights: {final_heights}")
        print(f"    Height spread: {height_spread}")
        print(f"    Consistent: {consistent}")

    finally:
        stop_cluster(nodes)

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    f_phase = results["phases"].get("failure", {})
    r_phase = results["phases"].get("recovery", {})
    print(f"  Network continued during failure: {f_phase.get('network_continued', '?')}")
    print(f"  Recovery consistent: {r_phase.get('consistent', '?')}")
    print(f"  Height spread after recovery: {r_phase.get('height_spread', '?')}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DOIN Fault Tolerance Benchmark")
    parser.add_argument("--nodes", type=int, default=4, help="Total nodes")
    parser.add_argument("--kill", type=int, default=2, help="Nodes to kill")
    parser.add_argument("--warmup", type=float, default=120.0, help="Warmup duration (s)")
    parser.add_argument("--failure-time", type=float, default=60.0, help="Failure period (s)")
    parser.add_argument("--recovery-time", type=float, default=120.0, help="Recovery period (s)")
    parser.add_argument("--output", "-o", default="fault_tolerance_results.json")
    parser.add_argument("--base-port", type=int, default=8480)
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        num_nodes=args.nodes,
        kill_count=args.kill,
        warmup_duration=args.warmup,
        failure_duration=args.failure_time,
        recovery_duration=args.recovery_time,
        base_port=args.base_port,
    ))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
