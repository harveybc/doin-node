"""Chain — in-memory blockchain with persistence support.

Manages the ordered sequence of blocks, provides validation,
and handles chain selection (longest valid chain wins).
"""

from __future__ import annotations

import json
import logging
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


class ChainError(Exception):
    """Raised when a chain operation fails."""


class ChainIntegrityError(ChainError):
    """A transaction inside a block fails content-hash binding.

    Typed integrity failure carrying block/transaction coordinates.
    Never includes the transaction payload (finding 201).
    """

    def __init__(
        self,
        message: str,
        *,
        block_index: int,
        tx_index: int,
        tx_id: str,
    ) -> None:
        super().__init__(
            f"{message} (block={block_index}, tx_index={tx_index}, "
            f"tx_id={tx_id[:16]}…)"
        )
        self.block_index = block_index
        self.tx_index = tx_index
        self.tx_id = tx_id


class Chain:
    """Manages the DON blockchain.

    Stores the ordered sequence of blocks, validates new blocks,
    and persists to disk as a JSON file.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._blocks: list[Block] = []
        self._data_dir = Path(data_dir) if data_dir else None
        self._block_index: dict[str, int] = {}  # hash -> position

    @property
    def blocks(self) -> list[Block]:
        """All blocks in the chain (read-only view)."""
        return list(self._blocks)

    @property
    def height(self) -> int:
        """Current chain height (number of blocks)."""
        return len(self._blocks)

    @property
    def tip(self) -> Block | None:
        """The latest block in the chain."""
        return self._blocks[-1] if self._blocks else None

    def get_block(self, index: int) -> Block | None:
        """Get a block by its index."""
        if 0 <= index < len(self._blocks):
            return self._blocks[index]
        return None

    def get_block_by_hash(self, block_hash: str) -> Block | None:
        """Get a block by its hash."""
        idx = self._block_index.get(block_hash)
        if idx is not None:
            return self._blocks[idx]
        return None

    def get_blocks_range(self, from_index: int, to_index: int) -> list[Block]:
        """Get blocks in an index range (inclusive).

        Args:
            from_index: Start block index.
            to_index: End block index (inclusive).

        Returns:
            List of blocks in the requested range.
        """
        from_index = max(0, from_index)
        to_index = min(to_index, len(self._blocks) - 1)
        if from_index > to_index:
            return []
        return self._blocks[from_index:to_index + 1]

    def has_transaction(self, tx_id: str) -> bool:
        """Return whether a transaction is already present in the chain."""
        return any(
            tx.id == tx_id
            for block in self._blocks
            for tx in block.transactions
        )

    def find_common_ancestor(self, their_tip_hash: str) -> int:
        """Find the highest block index that exists in both chains.

        If the tip hash matches any block in our chain, that's the
        common point. Otherwise returns -1.
        """
        idx = self._block_index.get(their_tip_hash)
        return idx if idx is not None else -1

    def validate_and_append_blocks(self, blocks: list[Block]) -> int:
        """Validate and append a sequence of blocks.

        Blocks must be in order and chain from our current tip.

        Args:
            blocks: Ordered list of blocks to append.

        Returns:
            Number of blocks successfully appended.

        Raises:
            ChainError: If a block fails validation.
        """
        appended = 0
        for block in blocks:
            try:
                self._validate_block(block)
                self._blocks.append(block)
                self._block_index[block.hash] = len(self._blocks) - 1
                appended += 1
            except ChainError as e:
                logger.warning("Block #%d failed validation: %s", block.header.index, e)
                break
        if appended:
            logger.info("Appended %d blocks (height now %d)", appended, self.height)
        return appended

    def initialize(self, generator_id: str = "genesis") -> Block:
        """Create and store the genesis block.

        Returns:
            The genesis block.

        Raises:
            ChainError: If chain is already initialized.
        """
        if self._blocks:
            raise ChainError("Chain already initialized")

        genesis = Block.genesis(generator_id)
        self._blocks.append(genesis)
        self._block_index[genesis.hash] = 0
        logger.info("Chain initialized with genesis block %s", genesis.hash[:12])
        return genesis

    def append_block(self, block: Block) -> None:
        """Validate and append a block to the chain.

        Args:
            block: The block to append.

        Raises:
            ChainError: If validation fails.
        """
        self._validate_block(block)
        self._blocks.append(block)
        self._block_index[block.hash] = len(self._blocks) - 1
        logger.info(
            "Block #%d appended (hash=%s, txs=%d)",
            block.header.index,
            block.hash[:12],
            len(block.transactions),
        )

    def _validate_block(self, block: Block) -> None:
        """Validate a block before appending.

        Checks:
        - Chain is not empty (genesis must exist)
        - Block index is sequential
        - Previous hash matches tip
        - Merkle root is correct
        - Block hash matches header
        """
        if not self._blocks:
            raise ChainError("Chain not initialized — append genesis first")

        tip = self._blocks[-1]
        expected_index = tip.header.index + 1

        if block.header.index != expected_index:
            raise ChainError(
                f"Expected block index {expected_index}, got {block.header.index}"
            )

        if block.header.previous_hash != tip.hash:
            raise ChainError(
                f"Previous hash mismatch: expected {tip.hash[:12]}, "
                f"got {block.header.previous_hash[:12]}"
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
                raise ChainIntegrityError(
                    "transaction ID does not match canonical content hash",
                    block_index=block.header.index,
                    tx_index=i,
                    tx_id=tx.id,
                )
            if expected_id in seen_ids:
                raise ChainIntegrityError(
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
            raise ChainError("Merkle root mismatch")

        # Verify block hash
        expected_hash = block.header.compute_hash()
        if block.hash != expected_hash:
            raise ChainError("Block hash mismatch")

    def save(self, path: str | Path | None = None) -> None:
        """Persist the chain to a JSON file."""
        save_path = Path(path) if path else self._default_path()
        if save_path is None:
            raise ChainError("No save path specified and no data_dir configured")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = [json.loads(block.model_dump_json()) for block in self._blocks]
        save_path.write_text(json.dumps(data, indent=2))
        logger.info("Chain saved to %s (%d blocks)", save_path, len(self._blocks))

    def load(self, path: str | Path | None = None) -> None:
        """Load the chain from a JSON file."""
        load_path = Path(path) if path else self._default_path()
        if load_path is None or not load_path.exists():
            raise ChainError(f"Chain file not found: {load_path}")

        data = json.loads(load_path.read_text())
        blocks: list[Block] = []
        for i, raw in enumerate(data):
            try:
                blocks.append(Block.model_validate(raw))
            except TransactionIntegrityError as exc:
                # Typed refusal with coordinates, no payload dump.
                raise ChainIntegrityError(
                    "stored transaction content does not match its ID",
                    block_index=i,
                    tx_index=exc.tx_index if exc.tx_index is not None else -1,
                    tx_id=exc.tx_id or "",
                ) from None
        self._blocks = blocks
        self._block_index = {b.hash: i for i, b in enumerate(self._blocks)}
        logger.info("Chain loaded from %s (%d blocks)", load_path, len(self._blocks))

    def _default_path(self) -> Path | None:
        """Default chain file path."""
        if self._data_dir:
            return self._data_dir / "chain.json"
        return None
