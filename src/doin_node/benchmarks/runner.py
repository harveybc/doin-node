"""Benchmark Suite Runner — runs all three benchmarks sequentially.

Usage:
    python -m doin_node.benchmarks.runner [--quick]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from .benchmark_scalability import run_benchmark as run_scalability
from .benchmark_fault_tolerance import run_benchmark as run_fault_tolerance
from .benchmark_attack_resistance import run_benchmark as run_attack_resistance


async def run_all(quick: bool = False) -> dict[str, Any]:
    """Run all benchmarks and collect results."""
    print("\n" + "=" * 60)
    print("  DOIN Network Benchmark Suite")
    print("=" * 60)
    print(f"  Mode: {'quick' if quick else 'full'}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results: dict[str, Any] = {
        "suite": "doin_benchmarks",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "quick" if quick else "full",
        "benchmarks": {},
    }

    # Scalability
    if quick:
        scale_result = await run_scalability(max_nodes=2, duration=60.0, base_port=8470)
    else:
        scale_result = await run_scalability(max_nodes=5, duration=300.0, base_port=8470)
    all_results["benchmarks"]["scalability"] = scale_result

    await asyncio.sleep(5.0)

    # Fault tolerance
    if quick:
        ft_result = await run_fault_tolerance(
            num_nodes=3, kill_count=1, warmup_duration=60.0,
            failure_duration=30.0, recovery_duration=60.0, base_port=8480,
        )
    else:
        ft_result = await run_fault_tolerance(
            num_nodes=5, kill_count=2, warmup_duration=120.0,
            failure_duration=60.0, recovery_duration=120.0, base_port=8480,
        )
    all_results["benchmarks"]["fault_tolerance"] = ft_result

    await asyncio.sleep(5.0)

    # Attack resistance
    if quick:
        ar_result = await run_attack_resistance(honest_nodes=2, duration=120.0, base_port=8490)
    else:
        ar_result = await run_attack_resistance(honest_nodes=3, duration=300.0, base_port=8490)
    all_results["benchmarks"]["attack_resistance"] = ar_result

    print("\n" + "=" * 60)
    print("  All Benchmarks Complete")
    print("=" * 60)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="DOIN Benchmark Suite Runner")
    parser.add_argument("--quick", action="store_true", help="Quick mode (shorter durations)")
    parser.add_argument("--output", "-o", default="benchmark_results.json")
    args = parser.parse_args()

    results = asyncio.run(run_all(quick=args.quick))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {args.output}")


if __name__ == "__main__":
    main()
