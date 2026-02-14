"""Peer Discovery — find and connect to peers in the DOIN network.

Three discovery mechanisms:
  1. Bootstrap nodes: hard-coded seed nodes that new nodes contact first
  2. Peer Exchange (PEX): connected peers share their known peers
  3. Random walk: periodically query random peers for their neighbors

This provides decentralized peer discovery without requiring a centralized
registry or DHT. Simple, robust, and sufficient for networks up to ~100K nodes.

For even larger networks, a full Kademlia DHT can be layered on top.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT = ClientTimeout(total=5)
PEX_INTERVAL = 60.0          # Seconds between peer exchange rounds
MAX_KNOWN_PEERS = 1000        # Max peers to track
MIN_CONNECTIONS = 4            # Minimum active connections to maintain
TARGET_CONNECTIONS = 8         # Target number of connections
MAX_CONNECTIONS = 50           # Hard cap on connections
PEER_TTL = 3600.0             # Seconds before a peer is considered stale
BOOTSTRAP_RETRY_INTERVAL = 30.0


@dataclass
class DiscoveredPeer:
    """A discovered peer with metadata."""

    peer_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    last_connected: float = 0.0
    connection_failures: int = 0
    source: str = "unknown"  # "bootstrap", "pex", "walk", "config"
    domains: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    chain_height: int = 0

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > PEER_TTL

    @property
    def is_connectable(self) -> bool:
        """Can we try connecting to this peer?"""
        if self.connection_failures > 10:
            return False
        if self.connection_failures > 0:
            # Exponential backoff: wait 2^failures seconds
            backoff = min(2 ** self.connection_failures, 300)
            return (time.time() - self.last_connected) > backoff
        return True


class PeerDiscovery:
    """Manages peer discovery and connection management.

    Maintains a pool of known peers, periodically discovers new ones,
    and ensures the node always has enough connections.
    """

    def __init__(
        self,
        our_peer_id: str,
        our_port: int,
        bootstrap_nodes: list[str] | None = None,
    ) -> None:
        self.our_peer_id = our_peer_id
        self.our_port = our_port
        self._bootstrap = bootstrap_nodes or []
        self._known_peers: dict[str, DiscoveredPeer] = {}
        self._connected: set[str] = set()  # Currently connected peer endpoints
        self._running = False

    @property
    def known_count(self) -> int:
        return len(self._known_peers)

    @property
    def connected_count(self) -> int:
        return len(self._connected)

    def add_peer(self, peer: DiscoveredPeer) -> None:
        """Add or update a discovered peer."""
        if peer.peer_id == self.our_peer_id:
            return
        if len(self._known_peers) >= MAX_KNOWN_PEERS:
            self._evict_stale()
        existing = self._known_peers.get(peer.endpoint)
        if existing:
            existing.last_seen = max(existing.last_seen, peer.last_seen)
            if peer.chain_height > existing.chain_height:
                existing.chain_height = peer.chain_height
            if peer.domains:
                existing.domains = peer.domains
            if peer.roles:
                existing.roles = peer.roles
        else:
            self._known_peers[peer.endpoint] = peer

    def mark_connected(self, endpoint: str) -> None:
        self._connected.add(endpoint)
        peer = self._known_peers.get(endpoint)
        if peer:
            peer.last_connected = time.time()
            peer.connection_failures = 0

    def mark_disconnected(self, endpoint: str) -> None:
        self._connected.discard(endpoint)

    def mark_failed(self, endpoint: str) -> None:
        peer = self._known_peers.get(endpoint)
        if peer:
            peer.connection_failures += 1

    def needs_more_peers(self) -> bool:
        return len(self._connected) < MIN_CONNECTIONS

    def get_connectable_peers(self, limit: int = 10) -> list[DiscoveredPeer]:
        """Get peers we can try connecting to."""
        candidates = [
            p for p in self._known_peers.values()
            if p.endpoint not in self._connected
            and p.is_connectable
            and not p.is_stale
        ]
        # Sort by: fewer failures first, then most recently seen
        candidates.sort(
            key=lambda p: (p.connection_failures, -p.last_seen),
        )
        return candidates[:limit]

    def get_peers_for_domain(self, domain_id: str) -> list[DiscoveredPeer]:
        """Get peers that participate in a specific domain."""
        return [
            p for p in self._known_peers.values()
            if domain_id in p.domains and not p.is_stale
        ]

    def get_random_peers(self, n: int = 5) -> list[DiscoveredPeer]:
        """Get N random known peers (for PEX sharing)."""
        active = [p for p in self._known_peers.values() if not p.is_stale]
        return random.sample(active, min(n, len(active)))

    def our_peer_info(self, domains: list[str] | None = None, roles: list[str] | None = None) -> dict[str, Any]:
        """Get our peer info for sharing with others."""
        return {
            "peer_id": self.our_peer_id,
            "port": self.our_port,
            "domains": domains or [],
            "roles": roles or [],
        }

    # ── Discovery methods ────────────────────────────────────────

    async def discover_from_bootstrap(self, session: ClientSession) -> int:
        """Contact bootstrap nodes and get their peer lists."""
        discovered = 0
        for addr in self._bootstrap:
            try:
                peers = await self._fetch_peers(session, addr)
                for p in peers:
                    self.add_peer(p)
                    discovered += 1
                # Add bootstrap node itself
                host, port_str = addr.rsplit(":", 1)
                self.add_peer(DiscoveredPeer(
                    peer_id=f"bootstrap-{addr}",
                    address=host,
                    port=int(port_str),
                    source="bootstrap",
                ))
                discovered += 1
            except Exception:
                logger.debug("Bootstrap %s unreachable", addr)
        return discovered

    async def peer_exchange(self, session: ClientSession) -> int:
        """Exchange peer lists with connected peers (PEX)."""
        discovered = 0
        for endpoint in list(self._connected):
            try:
                peers = await self._fetch_peers(session, endpoint)
                for p in peers:
                    if p.endpoint not in self._known_peers:
                        p.source = "pex"
                        self.add_peer(p)
                        discovered += 1
            except Exception:
                logger.debug("PEX failed with %s", endpoint)
        return discovered

    async def random_walk(self, session: ClientSession) -> int:
        """Contact random known peers to discover new ones."""
        discovered = 0
        targets = self.get_random_peers(3)
        for peer in targets:
            try:
                peers = await self._fetch_peers(session, peer.endpoint)
                for p in peers:
                    if p.endpoint not in self._known_peers:
                        p.source = "walk"
                        self.add_peer(p)
                        discovered += 1
            except Exception:
                pass
        return discovered

    async def _fetch_peers(
        self, session: ClientSession, endpoint: str,
    ) -> list[DiscoveredPeer]:
        """Fetch peer list from a node's /peers endpoint."""
        url = f"http://{endpoint}/peers"
        async with session.get(url, timeout=DISCOVERY_TIMEOUT) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            peers = []
            for p in data.get("peers", []):
                peers.append(DiscoveredPeer(
                    peer_id=p.get("peer_id", ""),
                    address=p.get("address", ""),
                    port=p.get("port", 8470),
                    domains=p.get("domains", []),
                    roles=p.get("roles", []),
                    chain_height=p.get("chain_height", 0),
                ))
            return peers

    # ── Maintenance ──────────────────────────────────────────────

    def _evict_stale(self) -> None:
        """Remove stale peers to make room for new ones."""
        stale = [
            ep for ep, p in self._known_peers.items()
            if p.is_stale and ep not in self._connected
        ]
        for ep in stale[:len(stale) // 2 + 1]:
            del self._known_peers[ep]

    def cleanup(self) -> int:
        """Remove stale and permanently failed peers."""
        to_remove = []
        for ep, peer in self._known_peers.items():
            if peer.connection_failures > 20 and ep not in self._connected:
                to_remove.append(ep)
            elif peer.is_stale and ep not in self._connected:
                to_remove.append(ep)
        for ep in to_remove:
            del self._known_peers[ep]
        return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        return {
            "known_peers": len(self._known_peers),
            "connected": len(self._connected),
            "bootstrap_nodes": len(self._bootstrap),
            "by_source": {
                source: sum(1 for p in self._known_peers.values() if p.source == source)
                for source in {"bootstrap", "pex", "walk", "config", "unknown"}
            },
        }
