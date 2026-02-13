"""Block sync protocol — synchronize chains between peers.

Handles:
  1. Initial sync: on connect, exchange chain status → fetch missing blocks
  2. Catch-up sync: when a block announcement reveals we're behind
  3. Fork resolution: when chains diverge, use fork choice rule

The sync is pull-based: a node that discovers it's behind requests
blocks from the peer that's ahead. Blocks are fetched via HTTP
(not flooding) since they can be large.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from doin_core.models.block import Block
from doin_core.protocol.messages import (
    BlockRequest,
    BlockResponse,
    ChainStatus,
)

logger = logging.getLogger(__name__)

# Max blocks to request in a single batch
MAX_BLOCKS_PER_REQUEST = 50

# Timeout for sync HTTP requests
SYNC_TIMEOUT = ClientTimeout(total=30)


@dataclass
class SyncState:
    """Tracks sync state with a specific peer."""

    peer_endpoint: str
    their_height: int = 0
    their_tip_hash: str = ""
    their_finalized_height: int = 0
    syncing: bool = False
    last_sync_height: int = 0
    sync_failures: int = 0

    @property
    def is_ahead(self) -> bool:
        """Check if the peer is ahead of our last sync height."""
        return self.their_height > self.last_sync_height


@dataclass
class SyncManager:
    """Manages block synchronization with peers.

    The sync flow:
      1. Exchange chain status (height + tip hash)
      2. If peer is ahead, request blocks from our height to theirs
      3. Validate and append received blocks
      4. If peer's tip is on a different fork, find common ancestor
         and decide via fork choice rule

    Sync is cooperative: both nodes serve blocks on request.
    """

    our_height: int = 0
    our_tip_hash: str = ""
    finalized_height: int = 0
    peers: dict[str, SyncState] = field(default_factory=dict)
    _active_syncs: set[str] = field(default_factory=set)

    def get_our_status(self) -> ChainStatus:
        """Get our chain status for exchange with peers."""
        return ChainStatus(
            chain_height=self.our_height,
            tip_hash=self.our_tip_hash,
            tip_index=max(0, self.our_height - 1),
            finalized_height=self.finalized_height,
        )

    def update_our_state(
        self,
        height: int,
        tip_hash: str,
        finalized_height: int = 0,
    ) -> None:
        """Update our chain state after local changes."""
        self.our_height = height
        self.our_tip_hash = tip_hash
        self.finalized_height = finalized_height

    def update_peer_status(self, endpoint: str, status: ChainStatus) -> SyncState:
        """Record a peer's chain status.

        Args:
            endpoint: Peer endpoint (host:port).
            status: Their chain status.

        Returns:
            Updated SyncState for this peer.
        """
        if endpoint not in self.peers:
            self.peers[endpoint] = SyncState(peer_endpoint=endpoint)

        state = self.peers[endpoint]
        state.their_height = status.chain_height
        state.their_tip_hash = status.tip_hash
        state.their_finalized_height = status.finalized_height
        state.last_sync_height = self.our_height
        return state

    def needs_sync(self, endpoint: str) -> bool:
        """Check if we need to sync with a peer."""
        state = self.peers.get(endpoint)
        if state is None:
            return False
        if state.syncing:
            return False
        return state.their_height > self.our_height

    def peers_ahead(self) -> list[SyncState]:
        """Get peers that are ahead of us, sorted by height (tallest first)."""
        ahead = [s for s in self.peers.values() if s.their_height > self.our_height]
        ahead.sort(key=lambda s: s.their_height, reverse=True)
        return ahead

    def compute_blocks_needed(self, endpoint: str) -> tuple[int, int] | None:
        """Compute the range of blocks we need from a peer.

        Returns:
            (from_index, to_index) inclusive, or None if no sync needed.
        """
        state = self.peers.get(endpoint)
        if state is None or state.their_height <= self.our_height:
            return None

        from_index = self.our_height  # We have 0..height-1, need height..
        to_index = min(
            state.their_height - 1,
            from_index + MAX_BLOCKS_PER_REQUEST - 1,
        )
        return (from_index, to_index)

    def mark_syncing(self, endpoint: str, syncing: bool = True) -> None:
        """Mark a peer as currently being synced with."""
        state = self.peers.get(endpoint)
        if state:
            state.syncing = syncing
            if syncing:
                self._active_syncs.add(endpoint)
            else:
                self._active_syncs.discard(endpoint)

    def record_sync_failure(self, endpoint: str) -> None:
        """Record a sync failure for a peer."""
        state = self.peers.get(endpoint)
        if state:
            state.sync_failures += 1
            state.syncing = False
            self._active_syncs.discard(endpoint)

    def record_sync_success(self, endpoint: str, new_height: int) -> None:
        """Record successful sync with a peer."""
        state = self.peers.get(endpoint)
        if state:
            state.sync_failures = 0
            state.syncing = False
            state.last_sync_height = new_height
            self._active_syncs.discard(endpoint)

    @property
    def is_syncing(self) -> bool:
        """Check if any sync is in progress."""
        return len(self._active_syncs) > 0


async def fetch_chain_status(
    session: ClientSession,
    endpoint: str,
) -> ChainStatus | None:
    """Fetch chain status from a peer via HTTP.

    Args:
        session: aiohttp client session.
        endpoint: Peer endpoint (host:port).

    Returns:
        ChainStatus or None on failure.
    """
    url = f"http://{endpoint}/chain/status"
    try:
        async with session.get(url, timeout=SYNC_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return ChainStatus.model_validate(data)
    except Exception:
        logger.debug("Failed to fetch chain status from %s", endpoint, exc_info=True)
        return None


async def fetch_blocks(
    session: ClientSession,
    endpoint: str,
    from_index: int,
    to_index: int,
) -> list[Block]:
    """Fetch blocks from a peer via HTTP.

    Args:
        session: aiohttp client session.
        endpoint: Peer endpoint (host:port).
        from_index: Start block index.
        to_index: End block index (inclusive).

    Returns:
        List of validated Block objects. Empty on failure.
    """
    url = f"http://{endpoint}/chain/blocks"
    params = {"from": str(from_index), "to": str(to_index)}
    try:
        async with session.get(url, params=params, timeout=SYNC_TIMEOUT) as resp:
            if resp.status != 200:
                logger.warning("Block fetch from %s returned %d", endpoint, resp.status)
                return []
            data = await resp.json()
            response = BlockResponse.model_validate(data)
            return [Block.model_validate(b) for b in response.blocks]
    except Exception:
        logger.debug("Failed to fetch blocks from %s", endpoint, exc_info=True)
        return []
