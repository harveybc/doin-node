"""Controlled flooding protocol for message propagation.

Messages are propagated to neighbors with a TTL that decrements on each hop.
Duplicate messages (same ID) are dropped to prevent infinite loops.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from doin_core.protocol.messages import Message, MessageType

logger = logging.getLogger(__name__)

# Maximum number of seen message IDs to cache (LRU eviction)
MAX_SEEN_CACHE = 10_000


@dataclass
class FloodingConfig:
    """Configuration for the controlled flooding protocol."""

    default_ttl: int = 7
    max_ttl: int = 15
    fanout: int | None = None  # None = flood to all neighbors
    dedup_window_seconds: float = 300.0


class FloodingProtocol:
    """Implements controlled flooding for P2P message propagation.

    Each message gets a unique ID based on its content hash. When a node
    receives a message, it checks the dedup cache — if seen before, drop it.
    Otherwise, decrement TTL and forward to neighbors (up to fanout limit).
    """

    def __init__(self, config: FloodingConfig | None = None) -> None:
        self.config = config or FloodingConfig()
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._handlers: dict[MessageType, list[Callable[..., Coroutine[Any, Any, None]]]] = {}

    def on_message(
        self,
        msg_type: MessageType,
        handler: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Register a handler for a specific message type.

        Args:
            msg_type: The message type to handle.
            handler: Async callable that receives (message, sender_peer_id).
        """
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    def should_propagate(self, message: Message) -> bool:
        """Check if a message should be propagated.

        Returns False if:
        - Message has been seen before (dedup)
        - TTL has reached 0
        - TTL exceeds max allowed
        """
        msg_id = self._message_id(message)

        # Check TTL
        if message.ttl <= 0:
            logger.debug("Dropping message %s: TTL expired", msg_id[:12])
            return False

        if message.ttl > self.config.max_ttl:
            logger.warning("Dropping message %s: TTL %d exceeds max", msg_id[:12], message.ttl)
            return False

        # Check dedup cache
        if msg_id in self._seen:
            logger.debug("Dropping message %s: already seen", msg_id[:12])
            return False

        return True

    def mark_seen(self, message: Message) -> None:
        """Mark a message as seen in the dedup cache."""
        msg_id = self._message_id(message)
        self._seen[msg_id] = time.time()

        # Evict old entries
        while len(self._seen) > MAX_SEEN_CACHE:
            self._seen.popitem(last=False)

    def prepare_forward(self, message: Message) -> Message:
        """Prepare a message for forwarding (decrement TTL)."""
        return Message(
            msg_type=message.msg_type,
            sender_id=message.sender_id,
            timestamp=message.timestamp,
            ttl=message.ttl - 1,
            payload=message.payload,
        )

    async def handle_incoming(self, message: Message, from_peer: str) -> bool:
        """Process an incoming message.

        Args:
            message: The received message.
            from_peer: Peer ID of the sender.

        Returns:
            True if the message should be forwarded, False if dropped.
        """
        if not self.should_propagate(message):
            return False

        self.mark_seen(message)

        # Dispatch to registered handlers
        handlers = self._handlers.get(message.msg_type, [])
        for handler in handlers:
            try:
                await handler(message, from_peer)
            except Exception:
                logger.exception(
                    "Handler error for %s from %s",
                    message.msg_type.value,
                    from_peer[:12],
                )

        return True

    def cleanup_stale(self) -> int:
        """Remove expired entries from the dedup cache.

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - self.config.dedup_window_seconds
        removed = 0
        while self._seen:
            msg_id, timestamp = next(iter(self._seen.items()))
            if timestamp < cutoff:
                self._seen.popitem(last=False)
                removed += 1
            else:
                break
        return removed

    @staticmethod
    def _message_id(message: Message) -> str:
        """Compute a unique ID for dedup purposes."""
        payload = json.dumps(
            {
                "msg_type": message.msg_type.value,
                "sender_id": message.sender_id,
                "timestamp": message.timestamp.isoformat(),
                "payload": message.payload,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
