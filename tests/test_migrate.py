"""Tests for JSON → SQLite chain migration."""

import json
import tempfile
from pathlib import Path

import pytest

from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.crypto.hashing import compute_merkle_root

from doin_node.storage.chaindb import ChainDB
from doin_node.storage.migrate import migrate_json_to_sqlite


def make_chain(n: int) -> list[Block]:
    """Create a valid chain of n blocks (including genesis)."""
    genesis = Block.genesis("test")
    blocks = [genesis]
    for i in range(1, n):
        prev = blocks[-1]
        tx = Transaction(
            tx_type=TransactionType.OPTIMAE_ACCEPTED,
            domain_id="test",
            peer_id="node-1",
            payload={"block": i},
        )
        merkle = compute_merkle_root([tx.id])
        header = BlockHeader(
            index=i,
            previous_hash=prev.hash,
            merkle_root=merkle,
            generator_id="node-1",
            weighted_performance_sum=1.0,
            threshold=1.0,
        )
        blocks.append(Block(header=header, transactions=[tx]))
    return blocks


def save_chain_json(blocks: list[Block], path: Path) -> None:
    data = [json.loads(b.model_dump_json()) for b in blocks]
    path.write_text(json.dumps(data))


class TestMigration:

    def test_migrate_empty_chain(self, tmp_path):
        json_path = tmp_path / "chain.json"
        json_path.write_text("[]")
        db_path = tmp_path / "chain.db"
        count = migrate_json_to_sqlite(json_path, db_path)
        assert count == 0

    def test_migrate_genesis_only(self, tmp_path):
        blocks = make_chain(1)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)

        db_path = tmp_path / "chain.db"
        count = migrate_json_to_sqlite(json_path, db_path)
        assert count == 1

        db = ChainDB(db_path)
        db.open()
        assert db.height == 1
        db.close()

    def test_migrate_multiple_blocks(self, tmp_path):
        blocks = make_chain(10)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)

        db_path = tmp_path / "chain.db"
        count = migrate_json_to_sqlite(json_path, db_path)
        assert count == 10

        db = ChainDB(db_path)
        db.open()
        assert db.height == 10
        # Verify tip matches
        tip = db.get_tip()
        assert tip is not None
        assert tip.hash == blocks[-1].hash
        db.close()

    def test_migrate_preserves_transactions(self, tmp_path):
        blocks = make_chain(5)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)

        db_path = tmp_path / "chain.db"
        migrate_json_to_sqlite(json_path, db_path)

        db = ChainDB(db_path)
        db.open()
        for i in range(1, 5):
            txs = db.get_transactions(i)
            assert len(txs) == 1
            assert txs[0].payload["block"] == i
        db.close()

    def test_migrate_block_by_hash(self, tmp_path):
        blocks = make_chain(3)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)

        db_path = tmp_path / "chain.db"
        migrate_json_to_sqlite(json_path, db_path)

        db = ChainDB(db_path)
        db.open()
        for block in blocks:
            found = db.get_block_by_hash(block.hash)
            assert found is not None
            assert found.header.index == block.header.index
        db.close()

    def test_json_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            migrate_json_to_sqlite(tmp_path / "nope.json", tmp_path / "chain.db")

    def test_existing_db_with_data_raises(self, tmp_path):
        blocks = make_chain(3)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)
        db_path = tmp_path / "chain.db"

        # First migration
        migrate_json_to_sqlite(json_path, db_path)

        # Second migration should fail
        with pytest.raises(ValueError, match="already contains"):
            migrate_json_to_sqlite(json_path, db_path)

    def test_migrate_large_chain(self, tmp_path):
        blocks = make_chain(100)
        json_path = tmp_path / "chain.json"
        save_chain_json(blocks, json_path)

        db_path = tmp_path / "chain.db"
        count = migrate_json_to_sqlite(json_path, db_path, batch_size=25)
        assert count == 100

        db = ChainDB(db_path)
        db.open()
        assert db.height == 100
        stats = db.get_stats()
        assert stats["total_transactions"] == 99  # genesis has 0 txs
        db.close()
