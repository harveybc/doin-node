"""Scalability Benchmark — measures time-to-reach-performance-threshold vs N nodes.

Runs clusters of 1, 2, 3, ... N nodes on localhost, measures how quickly
each cluster reaches a target optimization performance threshold.

Usage:
    python -m doin_node.benchmarks.benchmark_scalability [--max-nodes 5] [--duration 300]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

from .common import create_node_cluster, poll_status, start_cluster, stop_cluster


async def run_scalability_trial(
    num_nodes: int,
    duration: float,
    performance_threshold: float,
    base_port: int = 8470,
) -> dict[str, Any]:
    """Run one scalability trial with N nodes.

    Returns:
        Dict with trial results including time_to_threshold, final metrics, etc.
    """
    print(f"\n  Trial: {num_nodes} node(s), max {duration}s")

    nodes = create_node_cluster(num_nodes, base_port=base_port)

    result: dict[str, Any] = {
        "num_nodes": num_nodes,
        "duration_limit": duration,
        "performance_threshold": performance_threshold,
        "time_to_threshold": None,
        "threshold_reached": False,
        "final_chain_heights": [],
        "samples": [],
    }

    try:
        await start_cluster(nodes)
        start_time = time.time()
        poll_interval = 5.0

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            sample = {"elapsed": round(elapsed, 1), "nodes": []}

            for node in nodes:
                status = await poll_status(node.endpoint)
                node_data = {
                    "port": node.port,
                    "alive": node.is_alive(),
                    "status": status,
                }
                sample["nodes"].append(node_data)

                # Check if any node reports threshold reached
                if status and not result["threshold_reached"]:
                    chain_height = status.get("chain_height", 0)
                    # Use chain height as a proxy for progress
                    if chain_height >= 3:  # At least 3 blocks = meaningful work
                        result["threshold_reached"] = True
                        result["time_to_threshold"] = round(elapsed, 2)
                        print(f"    Threshold reached at {elapsed:.1f}s (height={chain_height})")

            result["samples"].append(sample)
            await asyncio.sleep(poll_interval)

        # Collect final heights
        for node in nodes:
            status = await poll_status(node.endpoint)
            height = status.get("chain_height", 0) if status else 0
            result["final_chain_heights"].append(height)

    finally:
        stop_cluster(nodes)

    if not result["threshold_reached"]:
        print(f"    Threshold NOT reached within {duration}s")

    return result


async def run_benchmark(
    max_nodes: int = 5,
    duration: float = 300.0,
    performance_threshold: float = -10.0,
    base_port: int = 8470,
) -> dict[str, Any]:
    """Run the full scalability benchmark."""
    print("=" * 60)
    print("  DOIN Scalability Benchmark")
    print("=" * 60)

    results = {
        "benchmark": "scalability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "max_nodes": max_nodes,
            "duration_per_trial": duration,
            "performance_threshold": performance_threshold,
        },
        "trials": [],
    }

    for n in range(1, max_nodes + 1):
        trial = await run_scalability_trial(
            num_nodes=n,
            duration=duration,
            performance_threshold=performance_threshold,
            base_port=base_port,
        )
        results["trials"].append(trial)
        # Wait between trials
        await asyncio.sleep(3.0)

    # Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    for trial in results["trials"]:
        n = trial["num_nodes"]
        t = trial["time_to_threshold"]
        reached = trial["threshold_reached"]
        heights = trial["final_chain_heights"]
        status = f"{t:.1f}s" if reached else "NOT REACHED"
        print(f"  {n} node(s): threshold={status}, final_heights={heights}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DOIN Scalability Benchmark")
    parser.add_argument("--max-nodes", type=int, default=3, help="Maximum number of nodes to test")
    parser.add_argument("--duration", type=float, default=180.0, help="Duration per trial (seconds)")
    parser.add_argument("--threshold", type=float, default=-10.0, help="Performance threshold")
    parser.add_argument("--output", "-o", default="scalability_results.json", help="Output JSON file")
    parser.add_argument("--base-port", type=int, default=8470, help="Base port for nodes")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        max_nodes=args.max_nodes,
        duration=args.duration,
        performance_threshold=args.threshold,
        base_port=args.base_port,
    ))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
