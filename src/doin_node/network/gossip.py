"""GossipSub — scalable message propagation replacing O(N²) flooding.

Instead of sending every message to every peer, GossipSub maintains:
  1. A mesh of D peers per topic (active forwarding)
  2. Additional peers for gossip metadata (lazy pull)

Messages propagate in O(log N) hops through the mesh.
Nodes not in the mesh hear about messages via gossip (IHAVE/IWANT).

This is modeled after libp2p's GossipSub v1.1 specification,
adapted for DOIN's message types.

Topics map to message types:
  - "optimae"   → commits, reveals, announcements
  - "blocks"    → block announcements
  - "tasks"     → task lifecycle
  - "discovery" → peer discovery, chain status
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from doin_core.protocol.messages import Message, MessageType

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

D_TARGET = 6           # Target mesh size per topic
D_LOW = 4              # Below this, graft new peers
D_HIGH = 12            # Above this, prune excess peers
D_LAZY = 6             # Peers to send gossip metadata to
GOSSIP_HISTORY = 5     # Number of heartbeat windows to remember
HEARTBEAT_INTERVAL = 1.0  # Seconds between gossip heartbeats
MAX_IHAVE_LENGTH = 100    # Max message IDs per IHAVE
CACHE_SIZE = 10_000       # Message dedup cache size
FANOUT_TTL = 60.0         # Seconds to keep fanout peers for a topic


# Message type → topic mapping
TOPIC_MAP: dict[MessageType, str] = {
    MessageType.OPTIMAE_COMMIT: "optimae",
    MessageType.OPTIMAE_REVEAL: "optimae",
    MessageType.OPTIMAE_ANNOUNCEMENT: "optimae",
    MessageType.BLOCK_ANNOUNCEMENT: "blocks",
    MessageType.CHAIN_STATUS: "blocks",
    MessageType.BLOCK_REQUEST: "blocks",
    MessageType.BLOCK_RESPONSE: "blocks",
    MessageType.TASK_CREATED: "tasks",
    MessageType.TASK_CLAIMED: "tasks",
    MessageType.TASK_COMPLETED: "tasks",
    MessageType.PEER_DISCOVERY: "discovery",
}


@dataclass
class PeerScore:
    """Tracks a peer's behavior for mesh management."""

    peer_id: str
    messages_delivered: int = 0
    messages_dropped: int = 0
    invalid_messages: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """Simple scoring: delivered - (dropped * 2) - (invalid * 10)."""
        return (
            self.messages_delivered
            - self.messages_dropped * 2
            - self.invalid_messages * 10
        )


@dataclass
class TopicState:
    """State for a single gossip topic."""

    mesh: set[str] = field(default_factory=set)        # Active forwarding peers
    fanout: set[str] = field(default_factory=set)      # Peers for topics we publish but don't subscribe
    fanout_last_pub: float = 0.0                       # Last publish timestamp for fanout cleanup
    recent_messages: list[list[str]] = field(default_factory=list)  # Message IDs per heartbeat window


class GossipSub:
    """GossipSub protocol for scalable P2P message propagation.

    Manages per-topic meshes and handles message forwarding,
    gossip metadata exchange, and mesh maintenance.
    """

    def __init__(
        self,
        peer_id: str,
        d_target: int = D_TARGET,
        d_low: int = D_LOW,
        d_high: int = D_HIGH,
    ) -> None:
        self.peer_id = peer_id
        self.d_target = d_target
        self.d_low = d_low
        self.d_high = d_high

        # Topic state
        self._topics: dict[str, TopicState] = {}
        self._subscriptions: set[str] = set()

        # Peer tracking
        self._known_peers: set[str] = set()
        self._peer_scores: dict[str, PeerScore] = {}
        self._peer_topics: dict[str, set[str]] = {}  # peer → topics they subscribe to

        # Message cache (dedup + gossip)
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._message_cache: dict[str, Message] = {}  # msg_id → message (for IWANT)

        # Handlers
        self._handlers: dict[MessageType, list[Callable[..., Coroutine[Any, Any, None]]]] = {}

        # Send callback (set by transport)
        self._send_fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]] | None = None

    def set_send_fn(
        self,
        fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]],
    ) -> None:
        """Set the function used to send messages to peers."""
        self._send_fn = fn

    # ── Subscription management ──────────────────────────────────

    def subscribe(self, topic: str) -> None:
        """Subscribe to a topic and join its mesh."""
        self._subscriptions.add(topic)
        if topic not in self._topics:
            self._topics[topic] = TopicState()

    def unsubscribe(self, topic: str) -> None:
        self._subscriptions.discard(topic)

    def subscribe_all(self) -> None:
        """Subscribe to all DOIN topics."""
        for topic in {"optimae", "blocks", "tasks", "discovery"}:
            self.subscribe(topic)

    # ── Message handlers ─────────────────────────────────────────

    def on_message(
        self,
        msg_type: MessageType,
        handler: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    # ── Peer management ──────────────────────────────────────────

    def add_peer(self, peer_id: str, topics: set[str] | None = None) -> None:
        """Register a known peer and add to all subscribed topic meshes."""
        self._known_peers.add(peer_id)
        if peer_id not in self._peer_scores:
            self._peer_scores[peer_id] = PeerScore(peer_id=peer_id)
        if topics:
            self._peer_topics[peer_id] = topics
        # Add to mesh for all subscribed topics so publish() can reach them
        for topic in self._subscriptions:
            state = self._topics.get(topic)
            if state and peer_id != self.peer_id:
                state.mesh.add(peer_id)

    def remove_peer(self, peer_id: str) -> None:
        self._known_peers.discard(peer_id)
        self._peer_scores.pop(peer_id, None)
        self._peer_topics.pop(peer_id, None)
        # Remove from all meshes
        for state in self._topics.values():
            state.mesh.discard(peer_id)
            state.fanout.discard(peer_id)

    # ── Publishing ───────────────────────────────────────────────

    async def publish(self, message: Message) -> int:
        """Publish a message to the appropriate topic.

        Returns the number of peers the message was sent to.
        """
        topic = TOPIC_MAP.get(message.msg_type, "discovery")
        msg_id = self._message_id(message)

        # Mark as seen
        self._mark_seen(msg_id, message)

        # Get forwarding peers
        state = self._topics.get(topic)
        if state is None:
            self.subscribe(topic)
            state = self._topics[topic]

        # If we're in the mesh for this topic, use mesh peers
        # Otherwise use fanout peers
        if topic in self._subscriptions and state.mesh:
            targets = state.mesh
        elif state.fanout:
            targets = state.fanout
            state.fanout_last_pub = time.time()
        else:
            # No mesh or fanout — pick random peers
            candidates = self._peers_for_topic(topic)
            targets = set(random.sample(
                list(candidates),
                min(self.d_target, len(candidates)),
            )) if candidates else set()
            state.fanout = targets
            state.fanout_last_pub = time.time()

        # Send to all targets
        sent = 0
        for peer_id in targets:
            if peer_id == self.peer_id:
                continue
            if await self._send_message(peer_id, message):
                sent += 1

        # Track for gossip
        if state.recent_messages:
            state.recent_messages[-1].append(msg_id)
        else:
            state.recent_messages.append([msg_id])

        return sent

    async def handle_incoming(
        self, message: Message, from_peer: str,
    ) -> bool:
        """Handle an incoming message from a peer.

        Returns True if the message was new (not seen before).
        """
        msg_id = self._message_id(message)

        # Update peer score
        score = self._peer_scores.get(from_peer)
        if score:
            score.messages_delivered += 1
            score.last_seen = time.time()

        # Dedup
        if msg_id in self._seen:
            return False

        self._mark_seen(msg_id, message)

        # Dispatch to handlers
        handlers = self._handlers.get(message.msg_type, [])
        for handler in handlers:
            try:
                await handler(message, from_peer)
            except Exception:
                logger.exception("Handler error for %s", message.msg_type.value)

        # Forward to mesh peers (excluding sender)
        topic = TOPIC_MAP.get(message.msg_type, "discovery")
        state = self._topics.get(topic)
        if state:
            for peer_id in list(state.mesh):
                if peer_id != from_peer and peer_id != self.peer_id:
                    await self._send_message(peer_id, message)

        return True

    # ── Gossip heartbeat ─────────────────────────────────────────

    async def heartbeat(self) -> None:
        """Periodic mesh maintenance.

        Called every HEARTBEAT_INTERVAL seconds:
        1. Graft peers if mesh is below D_LOW
        2. Prune peers if mesh is above D_HIGH
        3. Send IHAVE gossip to non-mesh peers
        4. Rotate message history windows
        """
        for topic, state in self._topics.items():
            if topic not in self._subscriptions:
                continue

            candidates = self._peers_for_topic(topic)

            # Graft: mesh too small → add peers
            if len(state.mesh) < self.d_low:
                needed = self.d_target - len(state.mesh)
                available = candidates - state.mesh - {self.peer_id}
                to_add = self._select_best_peers(available, needed)
                state.mesh.update(to_add)
                for peer_id in to_add:
                    await self._send_control(peer_id, "GRAFT", topic)

            # Prune: mesh too large → remove low-score peers
            if len(state.mesh) > self.d_high:
                excess = len(state.mesh) - self.d_target
                to_remove = self._select_worst_peers(state.mesh, excess)
                state.mesh -= to_remove
                for peer_id in to_remove:
                    await self._send_control(peer_id, "PRUNE", topic)

            # Gossip: send IHAVE to non-mesh peers
            non_mesh = candidates - state.mesh - {self.peer_id}
            gossip_targets = random.sample(
                list(non_mesh),
                min(D_LAZY, len(non_mesh)),
            ) if non_mesh else []

            if gossip_targets and state.recent_messages:
                # Collect recent message IDs
                recent_ids = []
                for window in state.recent_messages[-GOSSIP_HISTORY:]:
                    recent_ids.extend(window)
                recent_ids = recent_ids[-MAX_IHAVE_LENGTH:]

                if recent_ids:
                    for peer_id in gossip_targets:
                        await self._send_control(
                            peer_id, "IHAVE", topic,
                            {"message_ids": recent_ids},
                        )

            # Rotate history windows
            state.recent_messages.append([])
            if len(state.recent_messages) > GOSSIP_HISTORY + 1:
                state.recent_messages.pop(0)

        # Clean fanout for topics we don't subscribe to
        now = time.time()
        for topic, state in self._topics.items():
            if topic not in self._subscriptions and state.fanout:
                if now - state.fanout_last_pub > FANOUT_TTL:
                    state.fanout.clear()

        # Evict old seen cache entries
        while len(self._seen) > CACHE_SIZE:
            self._seen.popitem(last=False)

    async def handle_control(
        self, from_peer: str, control_type: str, topic: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Handle a gossip control message (GRAFT/PRUNE/IHAVE/IWANT)."""
        state = self._topics.get(topic)
        if state is None:
            return

        if control_type == "GRAFT":
            # Peer wants to join our mesh
            if len(state.mesh) < self.d_high:
                state.mesh.add(from_peer)
            else:
                # Too many peers — send PRUNE back
                await self._send_control(from_peer, "PRUNE", topic)

        elif control_type == "PRUNE":
            # Peer is removing us from their mesh
            state.mesh.discard(from_peer)

        elif control_type == "IHAVE":
            # Peer has messages we might not have
            if data and "message_ids" in data:
                wanted = [
                    mid for mid in data["message_ids"]
                    if mid not in self._seen
                ]
                if wanted:
                    await self._send_control(
                        from_peer, "IWANT", topic,
                        {"message_ids": wanted},
                    )

        elif control_type == "IWANT":
            # Peer wants messages from us
            if data and "message_ids" in data:
                for mid in data["message_ids"]:
                    msg = self._message_cache.get(mid)
                    if msg:
                        await self._send_message(from_peer, msg)

    # ── Helpers ──────────────────────────────────────────────────

    def _peers_for_topic(self, topic: str) -> set[str]:
        """Get all known peers that subscribe to a topic."""
        # For now, assume all peers subscribe to all topics
        # In production, use SUBSCRIBE messages to track
        return set(self._known_peers)

    def _select_best_peers(self, candidates: set[str], n: int) -> set[str]:
        """Select the N best-scoring peers from candidates."""
        scored = [
            (pid, self._peer_scores.get(pid, PeerScore(peer_id=pid)).score)
            for pid in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {pid for pid, _ in scored[:n]}

    def _select_worst_peers(self, candidates: set[str], n: int) -> set[str]:
        """Select the N worst-scoring peers from candidates."""
        scored = [
            (pid, self._peer_scores.get(pid, PeerScore(peer_id=pid)).score)
            for pid in candidates
        ]
        scored.sort(key=lambda x: x[1])
        return {pid for pid, _ in scored[:n]}

    def _mark_seen(self, msg_id: str, message: Message) -> None:
        self._seen[msg_id] = time.time()
        self._message_cache[msg_id] = message
        # Evict old cache entries
        while len(self._message_cache) > CACHE_SIZE:
            oldest = next(iter(self._message_cache))
            del self._message_cache[oldest]

    async def _send_message(self, peer_id: str, message: Message) -> bool:
        if self._send_fn is None:
            return False
        try:
            payload = json.loads(message.model_dump_json())
            return await self._send_fn(peer_id, {"type": "message", "data": payload})
        except Exception:
            logger.debug("Failed to send to %s", peer_id[:12], exc_info=True)
            return False

    async def _send_control(
        self, peer_id: str, control_type: str, topic: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        if self._send_fn is None:
            return False
        try:
            payload = {
                "type": "control",
                "control": control_type,
                "topic": topic,
                "from": self.peer_id,
            }
            if data:
                payload["data"] = data
            return await self._send_fn(peer_id, payload)
        except Exception:
            return False

    @staticmethod
    def _message_id(message: Message) -> str:
        payload = json.dumps({
            "msg_type": message.msg_type.value,
            "sender_id": message.sender_id,
            "timestamp": message.timestamp.isoformat(),
            "payload": message.payload,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_mesh_stats(self) -> dict[str, Any]:
        return {
            topic: {
                "mesh_size": len(state.mesh),
                "fanout_size": len(state.fanout),
                "recent_messages": sum(len(w) for w in state.recent_messages),
            }
            for topic, state in self._topics.items()
        }
