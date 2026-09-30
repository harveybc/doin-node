"""Stage-1 shadow archive: disposable file bytes, not a lake upload.

One fixture block is written, read back, and checked. A flipped byte is
refused. A second put of the same bytes does not create a second record.
A missing or unreadable file is a refusal, not a successful validation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone

import pytest

from doin_core.archive.body import ArchiveRefusal, build_archive
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.archive.file_adapter import LAKE_WRITE_FACT, FileArchiveAdapter
from doin_node.archive.warehouse import DisposableWarehouse, project_metrics
from doin_node.stats.chain_metrics import collect_chain_metrics
from doin_node.stats.olap_db import OLAPDatabase

TS = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _fixture():
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="fixture-domain",
        peer_id="peer-fixture",
        payload={
            "optimae_id": "opt-win",
            "parameters": {"w": 1},
            "verified_performance": 0.9,
            "reported_performance": 0.9,
        },
        timestamp=TS,
    )
    header = BlockHeader(
        index=3,
        previous_hash="ab" * 32,
        timestamp=TS,
        merkle_root=compute_merkle_root([tx.id]),
        generator_id="node-fixture",
        weighted_performance_sum=1.5,
        threshold=0.5,
    )
    block = Block(header=header, transactions=[tx])
    lose = {
        "record_id": "rec-lose-1",
        "candidate_id": "cand-lose",
        "attempt": 1,
        "won": False,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.2,
        "parameters": {"w": 0.2},
        "metrics": {"split": "val"},
    }
    win = {
        "record_id": "rec-win-1",
        "candidate_id": "cand-win",
        "attempt": 1,
        "won": True,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.9,
        "parameters": {"w": 1},
        "metrics": {"split": "val"},
    }
    retry = {
        "record_id": "rec-lose-2",
        "candidate_id": "cand-lose",
        "attempt": 2,
        "won": False,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.25,
        "parameters": {"w": 0.25},
        "metrics": {"split": "val"},
    }
    envelope = build_archive(block=block, candidates=[lose, win], retries=[retry])
    return block, envelope


def test_lake_registration_is_not_the_byte_store() -> None:
    assert FileArchiveAdapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    assert "does not copy resource bytes" in LAKE_WRITE_FACT
    assert "write_metrics" in LAKE_WRITE_FACT


def test_disposable_warehouse_matches_olap_call_shape() -> None:
    assert DisposableWarehouse.LABEL == "DISPOSABLE"
    for name in ("create_experiment", "record_round"):
        real = inspect.signature(getattr(OLAPDatabase, name))
        double = inspect.signature(getattr(DisposableWarehouse, name))
        assert list(real.parameters) == list(double.parameters)
    with pytest.raises(ArchiveRefusal, match="POSTGRES_REFUSED"):
        DisposableWarehouse("postgresql://example/doin")


def test_round_trip_tamper_retry_and_missing(tmp_path) -> None:
    _block, envelope = _fixture()
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(envelope)
    assert receipt.verified is True
    assert receipt.already_stored is False
    assert receipt.body_digest == envelope.body_digest
    assert receipt.manifest_digest == envelope.manifest_digest
    body, manifest = adapter.read_verified(receipt)
    assert body == envelope.body
    assert manifest == envelope.manifest
    assert hashlib.sha256(body).hexdigest() == receipt.body_digest

    again = adapter.put(envelope)
    assert again.already_stored is True
    assert again.body_digest == receipt.body_digest
    assert adapter.record_count() == 1
    assert len(list((tmp_path / "objects").glob(receipt.body_digest))) == 1

    body_path = tmp_path / "objects" / receipt.body_digest
    original = body_path.read_bytes()
    body_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        adapter.read_verified(receipt)
    body_path.write_bytes(original)

    body_path.unlink()
    with pytest.raises(ArchiveRefusal, match="MISSING"):
        adapter.read_body(receipt.body_digest)


def test_unreadable_file_refuses(tmp_path) -> None:
    _block, envelope = _fixture()
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(envelope)
    body_path = tmp_path / "objects" / receipt.body_digest
    os.chmod(body_path, 0)
    try:
        with pytest.raises(ArchiveRefusal, match="UNREADABLE"):
            adapter.read_body(receipt.body_digest)
    finally:
        os.chmod(body_path, 0o644)


def test_projection_keeps_non_winners_retries_and_unanchored(tmp_path) -> None:
    block, envelope = _fixture()
    accepted_only = collect_chain_metrics([block])
    assert len(accepted_only) == 1
    assert accepted_only[0]["optimae_id"] == "opt-win"

    warehouse = DisposableWarehouse(":memory:")
    assert warehouse.describe()["label"] == "DISPOSABLE"
    assert warehouse.describe()["postgres"] is False
    experiment_id = warehouse.create_experiment(
        domain_id="fixture-domain",
        node_id="node-fixture",
        hostname="",
    )
    inserted = project_metrics(warehouse, envelope, experiment_id=experiment_id)
    assert inserted == 3
    rows = warehouse.get_rounds(experiment_id)
    assert len(rows) == 3
    metrics = [json.loads(row["metrics"]) for row in rows]
    assert {item["candidate_id"] for item in metrics} == {"cand-lose", "cand-win"}
    assert any(item["won"] is False for item in metrics)
    assert any(item["kind"] == "retry" and item["attempt"] == 2 for item in metrics)
    assert {item["anchor"] for item in metrics} == {"ANCHORED"}
    assert all(item["chain_verified"] is False for item in metrics)
    assert project_metrics(warehouse, envelope, experiment_id=experiment_id) == 0
    assert len(warehouse.get_rounds(experiment_id)) == 3

    unanchored = build_archive(
        block=None,
        candidates=[
            {
                "record_id": "rec-open-1",
                "candidate_id": "cand-open",
                "attempt": 1,
                "won": False,
                "domain_id": "fixture-domain",
                "peer_id": "peer-fixture",
                "performance": 0.1,
                "parameters": {},
                "metrics": {},
            }
        ],
        retries=[],
    )
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(unanchored)
    assert receipt.anchor == "UNANCHORED"
    assert receipt.header_hash is None
    reloaded, manifest = adapter.read_verified(receipt)
    assert json.loads(reloaded)["schema"] == "doin.unanchored_records.v1"
    assert json.loads(manifest)["header_hash"] is None
    other = warehouse.create_experiment(
        domain_id="fixture-domain",
        node_id="node-fixture",
        hostname="",
        experiment_id="exp-unanchored",
    )
    assert project_metrics(warehouse, unanchored, experiment_id=other) == 1
    projected = json.loads(warehouse.get_rounds(other)[0]["metrics"])
    assert projected["anchor"] == "UNANCHORED"
    assert projected["chain_verified"] is False
    assert projected["header_hash"] is None
