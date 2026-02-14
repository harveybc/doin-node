"""ChainDB — SQLite-backed blockchain storage.

Replaces the JSON chain file with a proper database that supports:
  - O(1) block lookup by index or hash
  - Efficient range queries for sync
  - State snapshots at finality checkpoints
  - Pruning of old transaction bodies
  - Concurrent read access (WAL mode)
  - Crash recovery (SQLite transactions)

Schema:
  blocks:     header fields + hash, indexed by index and hash
  transactions: full tx data, indexed by id and block_index
  state_snapshots: periodic snapshots of balances + reputation
  metadata:   chain state (height, tip hash, etc.)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction
from doin_core.crypto.hashing import compute_merkle_root

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class ChainDB:
    """SQLite-backed blockchain storage.

    Thread-safe for reads (WAL mode). Writes must be serialized
    (single writer, enforced by SQLite).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Open the database and create tables if needed."""
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # Autocommit by default
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._create_tables()
        logger.info("ChainDB opened: %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS blocks (
                block_index INTEGER PRIMARY KEY,
                hash TEXT NOT NULL UNIQUE,
                previous_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                merkle_root TEXT NOT NULL,
                generator_id TEXT NOT NULL,
                weighted_performance_sum REAL NOT NULL,
                threshold REAL NOT NULL,
                tx_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash);

            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                block_index INTEGER NOT NULL,
                tx_index INTEGER NOT NULL,
                tx_type TEXT NOT NULL,
                domain_id TEXT NOT NULL,
                peer_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (block_index) REFERENCES blocks(block_index)
            );

            CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_index);
            CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(tx_type);
            CREATE INDEX IF NOT EXISTS idx_tx_peer ON transactions(peer_id);
            CREATE INDEX IF NOT EXISTS idx_tx_domain ON transactions(domain_id);

            CREATE TABLE IF NOT EXISTS state_snapshots (
                block_index INTEGER PRIMARY KEY,
                block_hash TEXT NOT NULL,
                balances TEXT NOT NULL,
                reputation TEXT NOT NULL,
                domain_stats TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS peers (
                peer_id TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                port INTEGER NOT NULL,
                last_seen REAL NOT NULL,
                reputation REAL NOT NULL DEFAULT 0.0,
                domains TEXT NOT NULL DEFAULT '[]',
                roles TEXT NOT NULL DEFAULT '[]'
            );
        """)

        # Set schema version
        self._conn.execute(
            "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

    # ── Block operations ─────────────────────────────────────────

    @property
    def height(self) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT MAX(block_index) as h FROM blocks"
        ).fetchone()
        h = row["h"]
        return (h + 1) if h is not None else 0

    @property
    def tip_hash(self) -> str:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT hash FROM blocks ORDER BY block_index DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else ""

    def get_tip(self) -> Block | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM blocks ORDER BY block_index DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def get_block(self, index: int) -> Block | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM blocks WHERE block_index = ?", (index,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def get_block_by_hash(self, block_hash: str) -> Block | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM blocks WHERE hash = ?", (block_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def get_blocks_range(self, from_index: int, to_index: int) -> list[Block]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM blocks WHERE block_index >= ? AND block_index <= ? ORDER BY block_index",
            (from_index, to_index),
        ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def append_block(self, block: Block) -> None:
        """Validate and append a block atomically (block + all transactions)."""
        assert self._conn is not None

        # Validate
        self._validate_block(block)

        # Atomic write
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """INSERT INTO blocks
                   (block_index, hash, previous_hash, timestamp, merkle_root,
                    generator_id, weighted_performance_sum, threshold, tx_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    block.header.index,
                    block.hash,
                    block.header.previous_hash,
                    block.header.timestamp.isoformat(),
                    block.header.merkle_root,
                    block.header.generator_id,
                    block.header.weighted_performance_sum,
                    block.header.threshold,
                    len(block.transactions),
                ),
            )

            for i, tx in enumerate(block.transactions):
                self._conn.execute(
                    """INSERT INTO transactions
                       (tx_id, block_index, tx_index, tx_type, domain_id,
                        peer_id, payload, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tx.id,
                        block.header.index,
                        i,
                        tx.tx_type.value,
                        tx.domain_id,
                        tx.peer_id,
                        json.dumps(tx.payload),
                        tx.timestamp.isoformat(),
                    ),
                )

            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def append_blocks(self, blocks: list[Block]) -> int:
        """Validate and append multiple blocks. Returns count appended."""
        appended = 0
        for block in blocks:
            try:
                self.append_block(block)
                appended += 1
            except Exception as e:
                logger.warning("Block #%d failed: %s", block.header.index, e)
                break
        return appended

    def _validate_block(self, block: Block) -> None:
        """Validate block before appending."""
        expected_index = self.height
        if block.header.index != expected_index:
            raise ValueError(
                f"Expected block index {expected_index}, got {block.header.index}"
            )

        if expected_index > 0:
            tip = self.tip_hash
            if block.header.previous_hash != tip:
                raise ValueError(
                    f"Previous hash mismatch: expected {tip[:16]}, "
                    f"got {block.header.previous_hash[:16]}"
                )

        # Verify merkle root
        tx_hashes = [tx.id for tx in block.transactions]
        expected_merkle = compute_merkle_root(tx_hashes)
        if block.header.merkle_root != expected_merkle:
            raise ValueError("Merkle root mismatch")

        # Verify block hash
        if block.hash != block.header.compute_hash():
            raise ValueError("Block hash mismatch")

    # ── Transaction queries ──────────────────────────────────────

    def get_transactions(self, block_index: int) -> list[Transaction]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM transactions WHERE block_index = ? ORDER BY tx_index",
            (block_index,),
        ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    def get_transactions_by_peer(
        self, peer_id: str, limit: int = 100
    ) -> list[Transaction]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM transactions WHERE peer_id = ? ORDER BY block_index DESC LIMIT ?",
            (peer_id, limit),
        ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    def get_transactions_by_type(
        self, tx_type: str, limit: int = 100
    ) -> list[Transaction]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM transactions WHERE tx_type = ? ORDER BY block_index DESC LIMIT ?",
            (tx_type, limit),
        ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    # ── State snapshots ──────────────────────────────────────────

    def save_snapshot(
        self,
        block_index: int,
        block_hash: str,
        balances: dict[str, float],
        reputation: dict[str, float],
        domain_stats: dict[str, Any],
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """INSERT OR REPLACE INTO state_snapshots
               (block_index, block_hash, balances, reputation, domain_stats)
               VALUES (?, ?, ?, ?, ?)""",
            (
                block_index,
                block_hash,
                json.dumps(balances),
                json.dumps(reputation),
                json.dumps(domain_stats),
            ),
        )

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM state_snapshots ORDER BY block_index DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "block_index": row["block_index"],
            "block_hash": row["block_hash"],
            "balances": json.loads(row["balances"]),
            "reputation": json.loads(row["reputation"]),
            "domain_stats": json.loads(row["domain_stats"]),
        }

    def get_snapshot_at(self, block_index: int) -> dict[str, Any] | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM state_snapshots WHERE block_index <= ? ORDER BY block_index DESC LIMIT 1",
            (block_index,),
        ).fetchone()
        if row is None:
            return None
        return {
            "block_index": row["block_index"],
            "block_hash": row["block_hash"],
            "balances": json.loads(row["balances"]),
            "reputation": json.loads(row["reputation"]),
            "domain_stats": json.loads(row["domain_stats"]),
        }

    # ── Pruning ──────────────────────────────────────────────────

    def prune_transactions_before(self, block_index: int) -> int:
        """Remove transaction bodies before a given block height.

        Keeps block headers intact for chain validation.
        Only prune blocks that have a state snapshot after them.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "DELETE FROM transactions WHERE block_index < ?", (block_index,)
        )
        pruned = cursor.rowcount
        if pruned:
            self._conn.execute("PRAGMA incremental_vacuum")
            logger.info("Pruned %d transactions before block %d", pruned, block_index)
        return pruned

    # ── Peer storage ─────────────────────────────────────────────

    def save_peer(
        self, peer_id: str, address: str, port: int,
        last_seen: float, reputation: float = 0.0,
        domains: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """INSERT OR REPLACE INTO peers
               (peer_id, address, port, last_seen, reputation, domains, roles)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                peer_id, address, port, last_seen, reputation,
                json.dumps(domains or []),
                json.dumps(roles or []),
            ),
        )

    def get_peers(self, limit: int = 100) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM peers ORDER BY last_seen DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "peer_id": r["peer_id"],
                "address": r["address"],
                "port": r["port"],
                "last_seen": r["last_seen"],
                "reputation": r["reputation"],
                "domains": json.loads(r["domains"]),
                "roles": json.loads(r["roles"]),
            }
            for r in rows
        ]

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        assert self._conn is not None
        tx_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM transactions"
        ).fetchone()["c"]
        snapshot_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM state_snapshots"
        ).fetchone()["c"]
        peer_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM peers"
        ).fetchone()["c"]
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "height": self.height,
            "tip_hash": self.tip_hash,
            "total_transactions": tx_count,
            "state_snapshots": snapshot_count,
            "known_peers": peer_count,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }

    # ── Genesis ──────────────────────────────────────────────────

    def initialize(self, generator_id: str = "genesis") -> Block:
        """Create and store the genesis block."""
        if self.height > 0:
            raise ValueError("Chain already initialized")
        genesis = Block.genesis(generator_id)
        self.append_block(genesis)
        return genesis

    # ── Conversion helpers ───────────────────────────────────────

    def _row_to_block(self, row: sqlite3.Row) -> Block:
        """Convert a database row to a Block with its transactions."""
        txs = self.get_transactions(row["block_index"])
        header = BlockHeader(
            index=row["block_index"],
            previous_hash=row["previous_hash"],
            timestamp=row["timestamp"],
            merkle_root=row["merkle_root"],
            generator_id=row["generator_id"],
            weighted_performance_sum=row["weighted_performance_sum"],
            threshold=row["threshold"],
        )
        return Block(header=header, transactions=txs, hash=row["hash"])

    def _row_to_transaction(self, row: sqlite3.Row) -> Transaction:
        from doin_core.models.transaction import TransactionType
        return Transaction(
            id=row["tx_id"],
            tx_type=TransactionType(row["tx_type"]),
            domain_id=row["domain_id"],
            peer_id=row["peer_id"],
            payload=json.loads(row["payload"]),
            timestamp=row["timestamp"],
        )
