"""Peer management — tracking neighbors and their state."""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field


class PeerState(str, Enum):
    """Connection state of a peer."""

    DISCOVERED = "discovered"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


@dataclass
class Peer:
    """Represents a known peer in the network."""

    peer_id: str
    address: str
    port: int
    state: PeerState = PeerState.DISCOVERED
    roles: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    latency_ms: float | None = None

    @property
    def endpoint(self) -> str:
        """Full endpoint address."""
        return f"{self.address}:{self.port}"

    def mark_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = time.time()

    def is_stale(self, timeout_seconds: float = 300.0) -> bool:
        """Check if peer hasn't been seen recently."""
        return (time.time() - self.last_seen) > timeout_seconds
