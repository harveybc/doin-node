"""Validate and display a modular experiment handoff without starting a node."""

from __future__ import annotations

import argparse
import json

from doin_node.cli import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Node JSON config to validate")
    args = parser.parse_args()
    config = load_config(args.config, {})
    handoffs = []
    for role in config.domains:
        experiment = role.optimization_config.get("modular_experiment")
        if experiment is not None:
            handoffs.append({
                "domain_id": role.domain_id,
                "regime": experiment["regime"],
                "branch_names": [b["name"] for b in experiment["architecture"]["branches"]],
                "metric": experiment["objective"]["metric"],
                "split": experiment["objective"]["split"],
                "optimizer_evaluator_match": (
                    role.inference_config.get("modular_experiment") == experiment
                ),
                "status": "config_validated_no_training",
            })
    if not handoffs:
        parser.error("config has no modular_experiment")
    print(json.dumps(handoffs, sort_keys=True))


if __name__ == "__main__":
    main()
