"""Tests for doin_node.storage.chaindb.ChainDB."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.crypto.hashing import compute_merkle_root
from doin_node.storage.chaindb import ChainDB


# ── Helpers ──────────────────────────────────────────────────────

def make_block(index: int, prev_hash: str, generator: str = "node-1") -> Block:
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="test",
        peer_id=generator,
        payload={"i": index},
    )
    merkle = compute_merkle_root([tx.id])
    header = BlockHeader(
        index=index,
        previous_hash=prev_hash,
        merkle_root=merkle,
        generator_id=generator,
        weighted_performance_sum=1.0,
        threshold=1.0,
    )
    return Block(header=header, transactions=[tx])


@pytest.fixture
def db(tmp_path):
    """Yield an opened ChainDB, close after test."""
    db_path = tmp_path / "test_chain.db"
    cdb = ChainDB(db_path)
    cdb.open()
    yield cdb
    cdb.close()


@pytest.fixture
def seeded_db(db):
    """DB with genesis + 3 blocks."""
    genesis = db.initialize("gen")
    prev = genesis.hash
    for i in range(1, 4):
        b = make_block(i, prev)
        db.append_block(b)
        prev = b.hash
    return db


# ── Open / close / tables ───────────────────────────────────────

class TestOpenClose:
    def test_open_creates_file(self, tmp_path):
        p = tmp_path / "sub" / "chain.db"
        cdb = ChainDB(p)
        cdb.open()
        assert p.exists()
        cdb.close()

    def test_double_close_safe(self, db):
        db.close()
        db.close()  # should not raise

    def test_tables_exist(self, db):
        tables = {
            r[0]
            for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"blocks", "transactions", "state_snapshots", "peers", "metadata"} <= tables


# ── Genesis ──────────────────────────────────────────────────────

class TestGenesis:
    def test_initialize(self, db):
        genesis = db.initialize("gen")
        assert genesis.header.index == 0
        assert db.height == 1

    def test_double_initialize_raises(self, db):
        db.initialize()
        with pytest.raises(ValueError, match="already initialized"):
            db.initialize()

    def test_genesis_tip(self, db):
        genesis = db.initialize()
        assert db.tip_hash == genesis.hash


# ── Append / validate ────────────────────────────────────────────

class TestAppendBlock:
    def test_append_valid(self, db):
        genesis = db.initialize()
        b1 = make_block(1, genesis.hash)
        db.append_block(b1)
        assert db.height == 2

    def test_wrong_index_raises(self, db):
        db.initialize()
        bad = make_block(5, "x" * 64)
        with pytest.raises(ValueError, match="Expected block index"):
            db.append_block(bad)

    def test_wrong_prev_hash_raises(self, db):
        db.initialize()
        bad = make_block(1, "0" * 64)
        with pytest.raises(ValueError, match="Previous hash mismatch"):
            db.append_block(bad)

    def test_bad_merkle_raises(self, db):
        genesis = db.initialize()
        b = make_block(1, genesis.hash)
        b.header.merkle_root = "bad" * 21 + "ba"
        # rehash so block hash is consistent with (bad) header
        b.hash = b.header.compute_hash()
        with pytest.raises(ValueError, match="Merkle root mismatch"):
            db.append_block(b)

    def test_bad_block_hash_raises(self, db):
        genesis = db.initialize()
        b = make_block(1, genesis.hash)
        b.hash = "0" * 64
        with pytest.raises(ValueError, match="Block hash mismatch"):
            db.append_block(b)

    def test_append_blocks_stops_on_error(self, db):
        genesis = db.initialize()
        b1 = make_block(1, genesis.hash)
        bad = make_block(5, "x" * 64)
        b3 = make_block(2, b1.hash)
        count = db.append_blocks([b1, bad, b3])
        assert count == 1
        assert db.height == 2


# ── Block queries ────────────────────────────────────────────────

class TestBlockQueries:
    def test_get_block_by_index(self, seeded_db):
        b = seeded_db.get_block(0)
        assert b is not None
        assert b.header.index == 0

    def test_get_block_none(self, seeded_db):
        assert seeded_db.get_block(999) is None

    def test_get_block_by_hash(self, seeded_db):
        b1 = seeded_db.get_block(1)
        found = seeded_db.get_block_by_hash(b1.hash)
        assert found is not None
        assert found.header.index == 1

    def test_get_block_by_hash_none(self, seeded_db):
        assert seeded_db.get_block_by_hash("nonexistent") is None

    def test_get_blocks_range(self, seeded_db):
        blocks = seeded_db.get_blocks_range(1, 3)
        assert len(blocks) == 3
        assert [b.header.index for b in blocks] == [1, 2, 3]

    def test_get_tip(self, seeded_db):
        tip = seeded_db.get_tip()
        assert tip is not None
        assert tip.header.index == 3

    def test_get_tip_empty(self, db):
        assert db.get_tip() is None

    def test_height_empty(self, db):
        assert db.height == 0

    def test_tip_hash_empty(self, db):
        assert db.tip_hash == ""


# ── Transaction queries ──────────────────────────────────────────

class TestTransactionQueries:
    def test_get_transactions_by_block(self, seeded_db):
        txs = seeded_db.get_transactions(1)
        assert len(txs) == 1
        assert txs[0].peer_id == "node-1"

    def test_get_transactions_empty_block(self, seeded_db):
        # genesis has no transactions
        txs = seeded_db.get_transactions(0)
        assert txs == []

    def test_get_transactions_by_peer(self, seeded_db):
        txs = seeded_db.get_transactions_by_peer("node-1")
        assert len(txs) == 3

    def test_get_transactions_by_peer_unknown(self, seeded_db):
        assert seeded_db.get_transactions_by_peer("nobody") == []

    def test_get_transactions_by_type(self, seeded_db):
        txs = seeded_db.get_transactions_by_type("optimae_accepted")
        assert len(txs) == 3

    def test_get_transactions_by_type_unknown(self, seeded_db):
        assert seeded_db.get_transactions_by_type("nonexistent") == []

    def test_transaction_limit(self, seeded_db):
        txs = seeded_db.get_transactions_by_peer("node-1", limit=2)
        assert len(txs) == 2


# ── State snapshots ──────────────────────────────────────────────

class TestSnapshots:
    def test_save_and_get_latest(self, seeded_db):
        seeded_db.save_snapshot(2, "hash2", {"a": 1.0}, {"a": 0.5}, {"d": 1})
        snap = seeded_db.get_latest_snapshot()
        assert snap is not None
        assert snap["block_index"] == 2
        assert snap["balances"] == {"a": 1.0}

    def test_multiple_snapshots(self, seeded_db):
        seeded_db.save_snapshot(1, "h1", {}, {}, {})
        seeded_db.save_snapshot(3, "h3", {"x": 9.0}, {}, {})
        snap = seeded_db.get_latest_snapshot()
        assert snap["block_index"] == 3

    def test_get_snapshot_at(self, seeded_db):
        seeded_db.save_snapshot(1, "h1", {"a": 1.0}, {}, {})
        seeded_db.save_snapshot(3, "h3", {"a": 3.0}, {}, {})
        snap = seeded_db.get_snapshot_at(2)
        assert snap["block_index"] == 1

    def test_no_snapshot(self, seeded_db):
        assert seeded_db.get_latest_snapshot() is None

    def test_snapshot_replace(self, seeded_db):
        seeded_db.save_snapshot(1, "h1", {"a": 1.0}, {}, {})
        seeded_db.save_snapshot(1, "h1", {"a": 2.0}, {}, {})
        snap = seeded_db.get_latest_snapshot()
        assert snap["balances"] == {"a": 2.0}


# ── Pruning ──────────────────────────────────────────────────────

class TestPruning:
    def test_prune_transactions(self, seeded_db):
        pruned = seeded_db.prune_transactions_before(2)
        assert pruned == 1  # block 1 had 1 tx
        assert seeded_db.get_transactions(1) == []
        assert len(seeded_db.get_transactions(2)) == 1

    def test_prune_none(self, seeded_db):
        assert seeded_db.prune_transactions_before(0) == 0

    def test_blocks_remain_after_prune(self, seeded_db):
        seeded_db.prune_transactions_before(3)
        assert seeded_db.get_block(1) is not None
        assert seeded_db.height == 4


# ── Peer storage ─────────────────────────────────────────────────

class TestPeerStorage:
    def test_save_and_get_peers(self, db):
        db.save_peer("p1", "127.0.0.1", 8000, time.time(), 0.5, ["d1"], ["evaluator"])
        peers = db.get_peers()
        assert len(peers) == 1
        assert peers[0]["peer_id"] == "p1"
        assert peers[0]["domains"] == ["d1"]

    def test_update_peer(self, db):
        db.save_peer("p1", "127.0.0.1", 8000, 1.0)
        db.save_peer("p1", "127.0.0.1", 9000, 2.0)
        peers = db.get_peers()
        assert len(peers) == 1
        assert peers[0]["port"] == 9000

    def test_multiple_peers_ordered(self, db):
        db.save_peer("old", "1.1.1.1", 80, 1.0)
        db.save_peer("new", "2.2.2.2", 80, 9999.0)
        peers = db.get_peers()
        assert peers[0]["peer_id"] == "new"

    def test_peers_limit(self, db):
        for i in range(5):
            db.save_peer(f"p{i}", "1.1.1.1", 8000 + i, float(i))
        assert len(db.get_peers(limit=3)) == 3


# ── Statistics ───────────────────────────────────────────────────

class TestStats:
    def test_stats_empty(self, db):
        stats = db.get_stats()
        assert stats["height"] == 0
        assert stats["total_transactions"] == 0

    def test_stats_seeded(self, seeded_db):
        seeded_db.save_peer("p1", "1.1.1.1", 80, 1.0)
        seeded_db.save_snapshot(1, "h", {}, {}, {})
        stats = seeded_db.get_stats()
        assert stats["height"] == 4
        assert stats["total_transactions"] == 3
        assert stats["state_snapshots"] == 1
        assert stats["known_peers"] == 1
        assert stats["db_size_bytes"] > 0
