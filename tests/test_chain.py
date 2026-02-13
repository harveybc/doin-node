"""Tests for blockchain chain management."""

import json
import tempfile
from pathlib import Path

import pytest

from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.crypto.hashing import compute_merkle_root
from doin_node.blockchain.chain import Chain, ChainError


def _make_block(previous: Block, index: int, generator: str = "node-1") -> Block:
    """Create a valid block linked to the previous one."""
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="d1",
        peer_id="optimizer-1",
        payload={"test": True},
    )
    merkle_root = compute_merkle_root([tx.id])

    header = BlockHeader(
        index=index,
        previous_hash=previous.hash,
        merkle_root=merkle_root,
        generator_id=generator,
        weighted_performance_sum=1.0,
        threshold=1.0,
    )
    return Block(header=header, transactions=[tx])


class TestChain:
    def test_initialize(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        assert chain.height == 1
        assert chain.tip == genesis

    def test_double_initialize_fails(self) -> None:
        chain = Chain()
        chain.initialize()
        with pytest.raises(ChainError, match="already initialized"):
            chain.initialize()

    def test_append_valid_block(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        block = _make_block(genesis, 1)
        chain.append_block(block)
        assert chain.height == 2

    def test_append_wrong_index_fails(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        block = _make_block(genesis, 5)  # Wrong index
        with pytest.raises(ChainError, match="Expected block index"):
            chain.append_block(block)

    def test_append_wrong_previous_hash_fails(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        block = _make_block(genesis, 1)
        # Tamper with previous hash
        block.header.previous_hash = "f" * 64
        block.hash = block.header.compute_hash()
        with pytest.raises(ChainError, match="Previous hash mismatch"):
            chain.append_block(block)

    def test_get_block_by_index(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        assert chain.get_block(0) == genesis
        assert chain.get_block(1) is None

    def test_get_block_by_hash(self) -> None:
        chain = Chain()
        genesis = chain.initialize()
        assert chain.get_block_by_hash(genesis.hash) == genesis
        assert chain.get_block_by_hash("nonexistent") is None

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            genesis = chain.initialize()
            block1 = _make_block(genesis, 1)
            chain.append_block(block1)
            chain.save()

            chain2 = Chain(data_dir=tmpdir)
            chain2.load()
            assert chain2.height == 2
            assert chain2.tip is not None
            assert chain2.tip.hash == block1.hash
