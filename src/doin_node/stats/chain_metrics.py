"""Helpers for enriching OPTIMAE_ACCEPTED on-chain payloads with experiment metrics.

These functions build, extract, and aggregate experiment context that rides
alongside the existing transaction payload — enabling L3 meta-optimizer
training, cross-node benchmarks, and reproducibility verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Keys added by build_onchain_metrics — used for detection / extraction
_METRIC_KEYS = frozenset({
    "experiment_id",
    "round_number",
    "time_to_this_result_seconds",
    "optimization_config_hash",
    "data_hash",
    "reported_performance",
    "previous_best_performance",
})


def _hash_dict(d: dict) -> str:
    """Deterministic SHA-256 hash of a dict."""
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_onchain_metrics(
    experiment_id: str,
    round_number: int,
    time_to_this_result_seconds: float,
    optimization_config: dict,
    data_hash: str | None = None,
    previous_best_performance: float | None = None,
    reported_performance: float = 0.0,
) -> dict[str, Any]:
    """Build the extra fields for OPTIMAE_ACCEPTED payload."""
    return {
        "experiment_id": experiment_id,
        "round_number": round_number,
        "time_to_this_result_seconds": round(time_to_this_result_seconds, 3),
        "optimization_config_hash": _hash_dict(optimization_config),
        "data_hash": data_hash or "",
        "reported_performance": reported_performance,
        "previous_best_performance": previous_best_performance,
    }


def extract_onchain_metrics(tx_payload: dict) -> dict | None:
    """Extract experiment metrics from an OPTIMAE_ACCEPTED transaction payload.

    Returns None if no experiment metrics present (old-format transactions).
    """
    if not tx_payload or "experiment_id" not in tx_payload:
        return None
    return {k: tx_payload.get(k) for k in _METRIC_KEYS}


def collect_chain_metrics(
    chain_blocks: list,
    domain_id: str | None = None,
) -> list[dict]:
    """Walk all blocks, extract experiment metrics from OPTIMAE_ACCEPTED txs.

    Returns a list of dicts suitable for OLAP ingestion or L3 meta-optimizer
    training. Optionally filter by *domain_id*.

    Each block is expected to be a ``Block`` model instance (or dict with
    ``header`` and ``transactions``).
    """
    results: list[dict] = []
    for block in chain_blocks:
        # Support both model objects and dicts
        if hasattr(block, "header"):
            header = block.header
            txs = block.transactions
            b_height = header.index
            b_ts = header.timestamp if hasattr(header, "timestamp") else ""
        else:
            header = block.get("header", {})
            txs = block.get("transactions", [])
            b_height = header.get("index", 0) if isinstance(header, dict) else getattr(header, "index", 0)
            b_ts = header.get("timestamp", "") if isinstance(header, dict) else getattr(header, "timestamp", "")

        for tx in txs:
            # Get tx_type
            if hasattr(tx, "tx_type"):
                tx_type = tx.tx_type
                tx_domain = tx.domain_id
                tx_peer = tx.peer_id
                payload = tx.payload
            else:
                tx_type = tx.get("tx_type", "")
                tx_domain = tx.get("domain_id", "")
                tx_peer = tx.get("peer_id", "")
                payload = tx.get("payload", {})

            # Filter for OPTIMAE_ACCEPTED
            tx_type_str = tx_type.value if hasattr(tx_type, "value") else str(tx_type)
            if tx_type_str != "optimae_accepted":
                continue

            if domain_id and tx_domain != domain_id:
                continue

            metrics = extract_onchain_metrics(payload)
            row = {
                "experiment_id": metrics.get("experiment_id") if metrics else None,
                "round_number": metrics.get("round_number") if metrics else None,
                "domain_id": tx_domain,
                "optimizer_id": tx_peer,
                "parameters": payload.get("parameters"),
                "verified_performance": payload.get("verified_performance"),
                "reported_performance": metrics.get("reported_performance") if metrics else payload.get("reported_performance"),
                "time_to_this_result_seconds": metrics.get("time_to_this_result_seconds") if metrics else None,
                "optimization_config_hash": metrics.get("optimization_config_hash") if metrics else None,
                "data_hash": metrics.get("data_hash") if metrics else None,
                "previous_best_performance": metrics.get("previous_best_performance") if metrics else None,
                "effective_increment": payload.get("effective_increment"),
                "reward_fraction": payload.get("reward_fraction"),
                "quorum_agree_fraction": payload.get("quorum_agree_fraction"),
                "optimae_id": payload.get("optimae_id"),
                "block_height": b_height,
                "block_timestamp": b_ts,
            }
            results.append(row)

    return results
