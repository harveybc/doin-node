"""Domain sharding for DOIN — nodes only track and process domains they participate in."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShardConfig:
    """Configuration for domain sharding."""
    subscribed_domains: list[str] = field(default_factory=list)
    relay_all: bool = False
    max_relay_domains: int = 50


class DomainShard:
    """Manages which domains a node participates in."""

    def __init__(self, config: ShardConfig | None = None) -> None:
        self.config = config or ShardConfig()
        self._peer_domains: dict[str, set[str]] = {}

    # --- domain membership ---

    def should_process(self, domain_id: str) -> bool:
        """Return True if this node should process data for *domain_id*."""
        if self.config.relay_all:
            return True
        return domain_id in self.config.subscribed_domains

    def should_relay(self, message: dict[str, Any]) -> bool:
        """Decide whether to relay/forward a message.

        Rules:
        - Blocks always relay (they contain multi-domain txs).
        - If relay_all, relay everything.
        - Otherwise relay only if the message's domain_id is subscribed.
        """
        if message.get("type") == "block":
            return True
        if self.config.relay_all:
            return True
        domain_id = message.get("domain_id")
        if domain_id is None:
            return False
        return domain_id in self.config.subscribed_domains

    def subscribe(self, domain_id: str) -> bool:
        """Subscribe to a domain. Returns False if at cap (non-relay_all)."""
        if domain_id in self.config.subscribed_domains:
            return True
        if not self.config.relay_all and len(self.config.subscribed_domains) >= self.config.max_relay_domains:
            return False
        self.config.subscribed_domains.append(domain_id)
        return True

    def unsubscribe(self, domain_id: str) -> None:
        try:
            self.config.subscribed_domains.remove(domain_id)
        except ValueError:
            pass

    # --- peer tracking ---

    def register_peer_domains(self, peer_id: str, domains: list[str]) -> None:
        """Track which domains a peer participates in."""
        self._peer_domains[peer_id] = set(domains)

    def get_peers_for_domain(self, domain_id: str) -> list[str]:
        """Return peer ids that share *domain_id*."""
        return [p for p, ds in self._peer_domains.items() if domain_id in ds]

    # --- stats ---

    def get_shard_stats(self) -> dict[str, Any]:
        all_domains: set[str] = set()
        for ds in self._peer_domains.values():
            all_domains.update(ds)
        return {
            "subscribed_domains": len(self.config.subscribed_domains),
            "relay_all": self.config.relay_all,
            "tracked_peers": len(self._peer_domains),
            "known_domains": len(all_domains),
            "max_relay_domains": self.config.max_relay_domains,
        }

    # --- tx filtering ---

    def filter_block_txs(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        """From a block, return only txs whose domain this node processes."""
        return [tx for tx in block.get("transactions", []) if self.should_process(tx.get("domain_id", ""))]


class ShardRouter:
    """Routes messages to the correct DomainShard (one shard instance per node)."""

    def __init__(self, shard: DomainShard) -> None:
        self.shard = shard

    def route(self, message: dict[str, Any]) -> str:
        """Return routing decision: 'process', 'relay', or 'drop'."""
        domain_id = message.get("domain_id", "")
        msg_type = message.get("type", "")

        if msg_type == "block":
            return "process"  # always accept blocks (filter txs inside)

        if self.shard.should_process(domain_id):
            return "process"

        if self.shard.should_relay(message):
            return "relay"

        return "drop"

    def select_peers(self, domain_id: str) -> list[str]:
        """Shard-aware peer selection — prefer peers in the same domain."""
        return self.shard.get_peers_for_domain(domain_id)
