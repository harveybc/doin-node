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
from doin_core.models.transaction import (
    TX_ID_PATTERN,
    Transaction,
    TransactionIntegrityError,
    compute_transaction_id,
)
from doin_core.crypto.hashing import compute_merkle_root

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Metadata keys (findings 202-203). Identity keys are stamped once and
# never silently rewritten; height/tip are maintained transactionally on
# every append/rollback; pruning keys record the checkpoint commitment
# that a pruned chain's suffix verification depends on.
META_CHAIN_ID = "chain_id"
META_GENESIS_HASH = "genesis_hash"
META_HEIGHT = "height"
META_TIP_HASH = "tip_hash"
META_PRUNED_BEFORE = "pruned_before_index"
META_CHECKPOINT_INDEX = "checkpoint_block_index"
META_CHECKPOINT_HASH = "checkpoint_block_hash"


class ChainIdentityError(Exception):
    """Typed refusal: the database attests a different chain identity."""

    def __init__(self, field: str, stored: str, requested: str) -> None:
        super().__init__(
            f"chain identity conflict on {field}: stored {stored!r}, "
            f"requested {requested!r}"
        )
        self.field = field
        self.stored = stored
        self.requested = requested


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

    # ── Metadata ─────────────────────────────────────────────────

    def get_metadata(self, key: str) -> str | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )

    def get_chain_identity(self) -> tuple[str, str] | None:
        """Return (chain_id, genesis_hash) as attested by this database.

        None when the database carries no identity attestation (legacy or
        freshly created chain).
        """
        chain_id = self.get_metadata(META_CHAIN_ID)
        genesis_hash = self.get_metadata(META_GENESIS_HASH)
        if not chain_id or not genesis_hash:
            return None
        return (chain_id, genesis_hash)

    def set_chain_identity(self, chain_id: str, genesis_hash: str) -> None:
        """Stamp the chain identity. Refuses to overwrite a different one.

        Raises ChainIdentityError (typed) on conflict — identity is never
        silently rewritten.
        """
        if not chain_id or not genesis_hash:
            raise ValueError("chain_id and genesis_hash must be non-empty")
        stored = self.get_chain_identity()
        if stored is not None:
            if stored[0] != chain_id:
                raise ChainIdentityError(META_CHAIN_ID, stored[0], chain_id)
            if stored[1] != genesis_hash:
                raise ChainIdentityError(
                    META_GENESIS_HASH, stored[1], genesis_hash
                )
            return
        self.set_metadata(META_CHAIN_ID, chain_id)
        self.set_metadata(META_GENESIS_HASH, genesis_hash)
        logger.info(
            "Chain identity stamped: chain_id=%s genesis=%s…",
            chain_id, genesis_hash[:16],
        )

    def get_pruning_checkpoint(self) -> dict[str, Any] | None:
        """Return the pruning/checkpoint provenance, or None if never pruned."""
        pruned_before = self.get_metadata(META_PRUNED_BEFORE)
        if pruned_before is None:
            return None
        checkpoint_index = self.get_metadata(META_CHECKPOINT_INDEX)
        checkpoint_hash = self.get_metadata(META_CHECKPOINT_HASH)
        return {
            "pruned_before_index": int(pruned_before),
            "checkpoint_block_index": (
                int(checkpoint_index) if checkpoint_index is not None else None
            ),
            "checkpoint_block_hash": checkpoint_hash,
        }

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

            # Maintain the metadata height/tip claim transactionally so the
            # verifier can check it against the verified rows (check 10).
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (META_HEIGHT, str(block.header.index + 1)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (META_TIP_HASH, block.hash),
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

    def rollback_to(self, index: int) -> int:
        """Remove all blocks with block_index > *index*.

        Used during chain reorganisation when we discover a longer
        valid chain from a peer.  Returns the number of blocks removed.
        """
        assert self._conn is not None
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "DELETE FROM transactions WHERE block_index > ?", (index,)
            )
            cur = self._conn.execute(
                "DELETE FROM blocks WHERE block_index > ?", (index,)
            )
            # Also remove any state snapshots that are now invalid
            self._conn.execute(
                "DELETE FROM state_snapshots WHERE block_index > ?", (index,)
            )
            # Refresh the metadata height/tip claim from the surviving rows
            tip_row = self._conn.execute(
                "SELECT block_index, hash FROM blocks "
                "ORDER BY block_index DESC LIMIT 1"
            ).fetchone()
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (
                    META_HEIGHT,
                    str(tip_row["block_index"] + 1) if tip_row else "0",
                ),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (META_TIP_HASH, tip_row["hash"] if tip_row else ""),
            )
            self._conn.execute("COMMIT")
            return cur.rowcount
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

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

        # Independently recompute every transaction's content hash before
        # Merkle calculation. Supplied IDs (and Pydantic construction) are
        # never trusted here (finding 201).
        recomputed_hashes: list[str] = []
        seen_ids: set[str] = set()
        for i, tx in enumerate(block.transactions):
            expected_id = compute_transaction_id(
                tx_type=tx.tx_type.value,
                domain_id=tx.domain_id,
                peer_id=tx.peer_id,
                payload=tx.payload,
                timestamp=tx.timestamp.isoformat(),
            )
            if not TX_ID_PATTERN.fullmatch(tx.id) or tx.id != expected_id:
                raise TransactionIntegrityError(
                    "transaction ID does not match canonical content hash",
                    block_index=block.header.index,
                    tx_index=i,
                    tx_id=tx.id,
                )
            if expected_id in seen_ids:
                raise TransactionIntegrityError(
                    "duplicate transaction ID within block",
                    block_index=block.header.index,
                    tx_index=i,
                    tx_id=expected_id,
                )
            seen_ids.add(expected_id)
            recomputed_hashes.append(expected_id)

        # Verify merkle root from the independently recomputed hashes
        expected_merkle = compute_merkle_root(recomputed_hashes)
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

    def has_transaction(self, tx_id: str) -> bool:
        """Return whether a transaction is already present in the chain."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT 1 FROM transactions WHERE tx_id = ? LIMIT 1",
            (tx_id,),
        ).fetchone()
        return row is not None

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

        Keeps block headers intact for chain validation. Records the
        pruning/checkpoint provenance atomically with the deletion
        (findings 202-203): a pruned chain can only ever verify as a
        typed ``verified_suffix_from_checkpoint``, and only when this
        commitment and the retained suffix verify. Without this
        metadata, missing bodies are corruption, not pruning.
        """
        assert self._conn is not None
        if block_index <= 0:
            return 0  # nothing below genesis to prune — historical no-op
        checkpoint_index = block_index - 1
        checkpoint_row = self._conn.execute(
            "SELECT hash FROM blocks WHERE block_index = ?",
            (checkpoint_index,),
        ).fetchone()
        if checkpoint_row is None:
            raise ValueError(
                f"cannot prune before block {block_index}: checkpoint block "
                f"{checkpoint_index} is not stored"
            )

        prior = self.get_metadata(META_PRUNED_BEFORE)
        effective_before = max(int(prior), block_index) if prior else block_index

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "DELETE FROM transactions WHERE block_index < ?", (block_index,)
            )
            pruned = cursor.rowcount
            for key, value in (
                (META_PRUNED_BEFORE, str(effective_before)),
                (META_CHECKPOINT_INDEX, str(effective_before - 1)),
            ):
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, value),
                )
            commit_row = self._conn.execute(
                "SELECT hash FROM blocks WHERE block_index = ?",
                (effective_before - 1,),
            ).fetchone()
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (META_CHECKPOINT_HASH, commit_row["hash"]),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
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

        # Verify the stored row's content against its stored ID from the raw
        # fields, before (and independent of) Pydantic construction. A
        # mismatch is a typed integrity failure with coordinates and no
        # payload dump (finding 201).
        stored_id = row["tx_id"]
        payload = json.loads(row["payload"])
        expected_id = compute_transaction_id(
            tx_type=row["tx_type"],
            domain_id=row["domain_id"],
            peer_id=row["peer_id"],
            payload=payload,
            timestamp=row["timestamp"],
        )
        if not TX_ID_PATTERN.fullmatch(stored_id) or stored_id != expected_id:
            raise TransactionIntegrityError(
                "stored transaction content does not match its ID",
                block_index=row["block_index"],
                tx_index=row["tx_index"],
                tx_id=stored_id,
            )
        return Transaction(
            id=stored_id,
            tx_type=TransactionType(row["tx_type"]),
            domain_id=row["domain_id"],
            peer_id=row["peer_id"],
            payload=payload,
            timestamp=row["timestamp"],
        )
