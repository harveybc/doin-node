"""Tests for doin_node.stats.chain_metrics and OLAP chain ingestion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from doin_node.stats.chain_metrics import (
    _hash_dict,
    build_onchain_metrics,
    collect_chain_metrics,
    extract_onchain_metrics,
)


# ── build_onchain_metrics ────────────────────────────────────────


class TestBuildOnchainMetrics:
    def test_basic_fields(self):
        m = build_onchain_metrics(
            experiment_id="exp1",
            round_number=5,
            time_to_this_result_seconds=12.3456,
            optimization_config={"lr": 0.01},
            data_hash="abc123",
            previous_best_performance=0.8,
            reported_performance=0.9,
        )
        assert m["experiment_id"] == "exp1"
        assert m["round_number"] == 5
        assert m["time_to_this_result_seconds"] == 12.346  # rounded to 3
        assert isinstance(m["optimization_config_hash"], str)
        assert len(m["optimization_config_hash"]) == 64  # SHA-256 hex
        assert m["data_hash"] == "abc123"
        assert m["previous_best_performance"] == 0.8
        assert m["reported_performance"] == 0.9

    def test_defaults(self):
        m = build_onchain_metrics(
            experiment_id="e",
            round_number=0,
            time_to_this_result_seconds=0.0,
            optimization_config={},
        )
        assert m["data_hash"] == ""
        assert m["previous_best_performance"] is None
        assert m["reported_performance"] == 0.0


# ── _hash_dict ───────────────────────────────────────────────────


class TestHashDict:
    def test_deterministic(self):
        d = {"b": 2, "a": 1}
        assert _hash_dict(d) == _hash_dict(d)
        assert _hash_dict({"a": 1, "b": 2}) == _hash_dict(d)

    def test_different_dicts(self):
        assert _hash_dict({"a": 1}) != _hash_dict({"a": 2})

    def test_empty(self):
        h = _hash_dict({})
        assert isinstance(h, str) and len(h) == 64


# ── extract_onchain_metrics ──────────────────────────────────────


class TestExtractOnchainMetrics:
    def test_old_format_returns_none(self):
        old_payload = {
            "optimae_id": "opt1",
            "verified_performance": 0.9,
            "effective_increment": 0.1,
        }
        assert extract_onchain_metrics(old_payload) is None

    def test_new_format_extracts(self):
        payload = {
            "optimae_id": "opt1",
            "verified_performance": 0.9,
            "experiment_id": "exp1",
            "round_number": 3,
            "time_to_this_result_seconds": 5.0,
            "optimization_config_hash": "abc",
            "data_hash": "",
            "reported_performance": 0.85,
            "previous_best_performance": 0.8,
        }
        m = extract_onchain_metrics(payload)
        assert m is not None
        assert m["experiment_id"] == "exp1"
        assert m["round_number"] == 3

    def test_empty_payload(self):
        assert extract_onchain_metrics({}) is None
        assert extract_onchain_metrics(None) is None  # type: ignore[arg-type]


# ── collect_chain_metrics ────────────────────────────────────────


def _make_block(index, timestamp, transactions):
    """Create a mock block object."""
    header = SimpleNamespace(index=index, timestamp=timestamp)
    return SimpleNamespace(header=header, transactions=transactions)


def _make_tx(tx_type, domain_id, peer_id, payload):
    tx_type_obj = SimpleNamespace(value=tx_type)
    return SimpleNamespace(
        tx_type=tx_type_obj,
        domain_id=domain_id,
        peer_id=peer_id,
        payload=payload,
    )


class TestCollectChainMetrics:
    def _sample_blocks(self):
        metrics = build_onchain_metrics(
            experiment_id="exp1",
            round_number=1,
            time_to_this_result_seconds=10.0,
            optimization_config={"lr": 0.01},
            reported_performance=0.9,
        )
        tx1 = _make_tx("optimae_accepted", "dom1", "peer1", {
            "optimae_id": "opt1",
            "verified_performance": 0.88,
            "effective_increment": 0.05,
            "reward_fraction": 0.5,
            "quorum_agree_fraction": 0.8,
            **metrics,
        })
        tx2 = _make_tx("optimae_accepted", "dom2", "peer2", {
            "optimae_id": "opt2",
            "verified_performance": 0.7,
            "effective_increment": 0.02,
        })
        tx3 = _make_tx("coin_minted", "dom1", "peer1", {"amount": 1.0})
        block = _make_block(1, "2025-01-01T00:00:00Z", [tx1, tx2, tx3])
        return [block]

    def test_collects_all(self):
        blocks = self._sample_blocks()
        results = collect_chain_metrics(blocks)
        assert len(results) == 2  # two optimae_accepted, coin_minted filtered out

    def test_filters_by_domain(self):
        blocks = self._sample_blocks()
        results = collect_chain_metrics(blocks, domain_id="dom1")
        assert len(results) == 1
        assert results[0]["domain_id"] == "dom1"
        assert results[0]["experiment_id"] == "exp1"

    def test_old_format_has_none_metrics(self):
        blocks = self._sample_blocks()
        results = collect_chain_metrics(blocks, domain_id="dom2")
        assert len(results) == 1
        assert results[0]["experiment_id"] is None

    def test_block_height_and_timestamp(self):
        blocks = self._sample_blocks()
        results = collect_chain_metrics(blocks)
        assert results[0]["block_height"] == 1
        assert results[0]["block_timestamp"] == "2025-01-01T00:00:00Z"


# ── OLAP ingestion ──────────────────────────────────────────────


class TestOLAPChainIngestion:
    def _make_olap(self):
        from doin_node.stats.olap_db import OLAPDatabase
        tmp = tempfile.mktemp(suffix=".db")
        return OLAPDatabase(tmp), tmp

    def _sample_blocks(self):
        metrics = build_onchain_metrics(
            experiment_id="exp1",
            round_number=1,
            time_to_this_result_seconds=10.0,
            optimization_config={"lr": 0.01},
            reported_performance=0.9,
        )
        tx = _make_tx("optimae_accepted", "dom1", "peer1", {
            "optimae_id": "opt1",
            "verified_performance": 0.88,
            "effective_increment": 0.05,
            "reward_fraction": 0.5,
            "quorum_agree_fraction": 0.8,
            **metrics,
        })
        return [_make_block(1, "2025-01-01T00:00:00Z", [tx])]

    def test_ingest_creates_records(self):
        db, path = self._make_olap()
        blocks = self._sample_blocks()
        count = db.ingest_from_chain(blocks)
        assert count == 1
        row = db._conn.execute(
            "SELECT * FROM fact_chain_optimae WHERE optimae_id='opt1'"
        ).fetchone()
        assert row is not None
        assert row["domain_id"] == "dom1"
        assert row["experiment_id"] == "exp1"
        db.close()

    def test_idempotent(self):
        db, path = self._make_olap()
        blocks = self._sample_blocks()
        c1 = db.ingest_from_chain(blocks)
        c2 = db.ingest_from_chain(blocks)
        assert c1 == 1
        assert c2 == 0  # no new records
        total = db._conn.execute("SELECT COUNT(*) FROM fact_chain_optimae").fetchone()[0]
        assert total == 1
        db.close()


# ── Integration: full flow ───────────────────────────────────────


class TestIntegrationFlow:
    def test_build_collect_ingest(self):
        """Full flow: build metrics → put in payload → collect from chain → ingest into OLAP."""
        from doin_node.stats.olap_db import OLAPDatabase

        # 1. Build metrics
        metrics = build_onchain_metrics(
            experiment_id="exp_full",
            round_number=3,
            time_to_this_result_seconds=42.123,
            optimization_config={"algo": "cma-es", "pop": 50},
            data_hash="datahash123",
            previous_best_performance=0.75,
            reported_performance=0.85,
        )

        # 2. Create payload (as unified.py would)
        payload = {
            "optimae_id": "opt_full",
            "verified_performance": 0.83,
            "effective_increment": 0.08,
            "reward_fraction": 0.6,
            "quorum_agree_fraction": 0.9,
            "incentive_reason": "honest",
            **metrics,
        }

        # 3. Create mock block
        tx = _make_tx("optimae_accepted", "test_domain", "node_abc", payload)
        blocks = [_make_block(5, "2025-06-15T12:00:00Z", [tx])]

        # 4. Collect
        collected = collect_chain_metrics(blocks)
        assert len(collected) == 1
        assert collected[0]["experiment_id"] == "exp_full"
        assert collected[0]["optimization_config_hash"] == metrics["optimization_config_hash"]

        # 5. Ingest into OLAP
        tmp = tempfile.mktemp(suffix=".db")
        db = OLAPDatabase(tmp)
        count = db.ingest_from_chain(blocks)
        assert count == 1

        row = db._conn.execute(
            "SELECT * FROM fact_chain_optimae WHERE optimae_id='opt_full'"
        ).fetchone()
        assert row["experiment_id"] == "exp_full"
        assert row["round_number"] == 3
        assert row["verified_performance"] == 0.83
        assert row["block_height"] == 5
        assert row["data_hash"] == "datahash123"
        db.close()
