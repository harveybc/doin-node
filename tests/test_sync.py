"""Tests for block sync protocol — chain synchronization between peers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.protocol.messages import ChainStatus

from doin_node.blockchain.chain import Chain, ChainError
from doin_node.network.sync import SyncManager, SyncState, MAX_BLOCKS_PER_REQUEST


# ── Helpers ──────────────────────────────────────────────────────────

def make_block(index: int, previous_hash: str, generator: str = "node-1") -> Block:
    """Create a valid block for testing."""
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="test",
        peer_id=generator,
        payload={"block": index},
    )
    merkle_root = compute_merkle_root([tx.id])
    header = BlockHeader(
        index=index,
        previous_hash=previous_hash,
        merkle_root=merkle_root,
        generator_id=generator,
        weighted_performance_sum=1.0,
        threshold=1.0,
    )
    return Block(header=header, transactions=[tx])


def make_chain(length: int, generator: str = "node-1") -> Chain:
    """Build a chain of the given length (including genesis)."""
    chain = Chain()
    chain.initialize(generator)
    for i in range(1, length):
        block = make_block(i, chain.tip.hash, generator)
        chain.append_block(block)
    return chain


# ── Chain methods ────────────────────────────────────────────────────

class TestChainBlocksRange:

    def test_get_full_range(self):
        chain = make_chain(5)
        blocks = chain.get_blocks_range(0, 4)
        assert len(blocks) == 5
        assert blocks[0].header.index == 0
        assert blocks[4].header.index == 4

    def test_get_partial_range(self):
        chain = make_chain(5)
        blocks = chain.get_blocks_range(2, 3)
        assert len(blocks) == 2
        assert blocks[0].header.index == 2
        assert blocks[1].header.index == 3

    def test_empty_range(self):
        chain = make_chain(5)
        blocks = chain.get_blocks_range(10, 15)
        assert len(blocks) == 0

    def test_clamped_range(self):
        chain = make_chain(3)
        blocks = chain.get_blocks_range(0, 100)
        assert len(blocks) == 3


class TestChainValidateAndAppend:

    def test_append_valid_blocks(self):
        chain = make_chain(3)
        # Build 2 more blocks that chain from the tip
        new_blocks = []
        prev_hash = chain.tip.hash
        for i in range(3, 5):
            b = make_block(i, prev_hash)
            new_blocks.append(b)
            prev_hash = b.hash

        appended = chain.validate_and_append_blocks(new_blocks)
        assert appended == 2
        assert chain.height == 5

    def test_append_invalid_block_stops(self):
        chain = make_chain(3)
        # First block is valid, second has wrong previous_hash
        b1 = make_block(3, chain.tip.hash)
        b2 = make_block(4, "wrong_hash")

        appended = chain.validate_and_append_blocks([b1, b2])
        assert appended == 1
        assert chain.height == 4

    def test_append_empty_list(self):
        chain = make_chain(3)
        appended = chain.validate_and_append_blocks([])
        assert appended == 0


class TestChainFindCommonAncestor:

    def test_tip_hash_found(self):
        chain = make_chain(5)
        block_2 = chain.get_block(2)
        idx = chain.find_common_ancestor(block_2.hash)
        assert idx == 2

    def test_unknown_hash(self):
        chain = make_chain(5)
        idx = chain.find_common_ancestor("nonexistent")
        assert idx == -1


# ── SyncManager ──────────────────────────────────────────────────────

class TestSyncManager:

    def test_get_our_status(self):
        sm = SyncManager(our_height=10, our_tip_hash="abc", finalized_height=5)
        status = sm.get_our_status()
        assert status.chain_height == 10
        assert status.tip_hash == "abc"
        assert status.finalized_height == 5

    def test_update_peer_status(self):
        sm = SyncManager(our_height=5)
        status = ChainStatus(chain_height=10, tip_hash="xyz", tip_index=9)
        state = sm.update_peer_status("peer1:8470", status)
        assert state.their_height == 10
        assert state.their_tip_hash == "xyz"

    def test_needs_sync_when_behind(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer1:8470", ChainStatus(
            chain_height=10, tip_hash="x", tip_index=9,
        ))
        assert sm.needs_sync("peer1:8470")

    def test_no_sync_when_equal(self):
        sm = SyncManager(our_height=10)
        sm.update_peer_status("peer1:8470", ChainStatus(
            chain_height=10, tip_hash="x", tip_index=9,
        ))
        assert not sm.needs_sync("peer1:8470")

    def test_no_sync_while_syncing(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer1:8470", ChainStatus(
            chain_height=10, tip_hash="x", tip_index=9,
        ))
        sm.mark_syncing("peer1:8470")
        assert not sm.needs_sync("peer1:8470")

    def test_peers_ahead_sorted(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("a:1", ChainStatus(chain_height=7, tip_hash="a", tip_index=6))
        sm.update_peer_status("b:2", ChainStatus(chain_height=15, tip_hash="b", tip_index=14))
        sm.update_peer_status("c:3", ChainStatus(chain_height=10, tip_hash="c", tip_index=9))

        ahead = sm.peers_ahead()
        assert len(ahead) == 3
        assert ahead[0].their_height == 15
        assert ahead[1].their_height == 10
        assert ahead[2].their_height == 7

    def test_compute_blocks_needed(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer:1", ChainStatus(
            chain_height=100, tip_hash="x", tip_index=99,
        ))
        needed = sm.compute_blocks_needed("peer:1")
        assert needed is not None
        from_idx, to_idx = needed
        assert from_idx == 5
        assert to_idx == 5 + MAX_BLOCKS_PER_REQUEST - 1

    def test_compute_blocks_needed_small_gap(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer:1", ChainStatus(
            chain_height=8, tip_hash="x", tip_index=7,
        ))
        needed = sm.compute_blocks_needed("peer:1")
        assert needed is not None
        from_idx, to_idx = needed
        assert from_idx == 5
        assert to_idx == 7  # Only need 3 blocks

    def test_record_sync_success(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer:1", ChainStatus(
            chain_height=10, tip_hash="x", tip_index=9,
        ))
        sm.mark_syncing("peer:1")
        assert sm.is_syncing
        sm.record_sync_success("peer:1", 10)
        assert not sm.is_syncing
        assert sm.peers["peer:1"].sync_failures == 0

    def test_record_sync_failure(self):
        sm = SyncManager(our_height=5)
        sm.update_peer_status("peer:1", ChainStatus(
            chain_height=10, tip_hash="x", tip_index=9,
        ))
        sm.mark_syncing("peer:1")
        sm.record_sync_failure("peer:1")
        assert not sm.is_syncing
        assert sm.peers["peer:1"].sync_failures == 1


# ── Integration: Chain + SyncManager ─────────────────────────────────

class TestChainSyncIntegration:
    """Simulate two chains and sync from one to the other."""

    def test_sync_chain_behind(self):
        """Node A has 10 blocks, Node B has 3 → B syncs from A."""
        chain_a = make_chain(10)
        chain_b = make_chain(3)

        # B discovers A is at height 10
        sm = SyncManager(
            our_height=chain_b.height,
            our_tip_hash=chain_b.tip.hash,
        )
        sm.update_peer_status("a:8470", ChainStatus(
            chain_height=chain_a.height,
            tip_hash=chain_a.tip.hash,
            tip_index=chain_a.tip.header.index,
        ))

        assert sm.needs_sync("a:8470")
        needed = sm.compute_blocks_needed("a:8470")
        assert needed is not None

        from_idx, to_idx = needed
        # "Fetch" blocks from chain A
        blocks = chain_a.get_blocks_range(from_idx, to_idx)
        assert len(blocks) > 0

        # But these blocks chain from A's genesis, not B's!
        # In a real scenario, both chains start from the same genesis.
        # Let's test with a shared genesis:

    def test_sync_shared_genesis(self):
        """Both nodes share genesis. A is ahead. B syncs."""
        # Build chain A with 10 blocks
        chain_a = make_chain(10)

        # Build chain B from same genesis (copy blocks 0-2)
        chain_b = Chain()
        chain_b._blocks = list(chain_a.blocks[:3])
        chain_b._block_index = {b.hash: i for i, b in enumerate(chain_b._blocks)}

        assert chain_b.height == 3
        assert chain_b.tip.hash == chain_a.get_block(2).hash  # Same block 2

        # Fetch missing blocks from A
        blocks_to_sync = chain_a.get_blocks_range(3, 9)
        assert len(blocks_to_sync) == 7

        # Validate and append
        appended = chain_b.validate_and_append_blocks(blocks_to_sync)
        assert appended == 7
        assert chain_b.height == 10
        assert chain_b.tip.hash == chain_a.tip.hash  # Same tip!

    def test_sync_already_caught_up(self):
        """Chains are equal → no sync needed."""
        chain_a = make_chain(5)
        sm = SyncManager(our_height=5, our_tip_hash=chain_a.tip.hash)
        sm.update_peer_status("a:8470", ChainStatus(
            chain_height=5, tip_hash=chain_a.tip.hash, tip_index=4,
        ))
        assert not sm.needs_sync("a:8470")

    def test_sync_we_are_ahead(self):
        """We're ahead of the peer → no sync, but peer can sync from us."""
        sm = SyncManager(our_height=10, our_tip_hash="xxx")
        sm.update_peer_status("a:8470", ChainStatus(
            chain_height=5, tip_hash="yyy", tip_index=4,
        ))
        assert not sm.needs_sync("a:8470")
        assert len(sm.peers_ahead()) == 0
