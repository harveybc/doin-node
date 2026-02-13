"""Tests for the controlled flooding protocol."""

import asyncio
from datetime import datetime, timezone

import pytest

from doin_core.protocol.messages import Message, MessageType
from doin_node.network.flooding import FloodingConfig, FloodingProtocol


def _make_message(
    sender: str = "peer-1",
    ttl: int = 7,
    msg_type: MessageType = MessageType.OPTIMAE_ANNOUNCEMENT,
) -> Message:
    return Message(
        msg_type=msg_type,
        sender_id=sender,
        ttl=ttl,
        payload={"test": True},
    )


class TestFloodingProtocol:
    def test_new_message_propagates(self) -> None:
        fp = FloodingProtocol()
        msg = _make_message()
        assert fp.should_propagate(msg)

    def test_duplicate_message_dropped(self) -> None:
        fp = FloodingProtocol()
        msg = _make_message()
        fp.mark_seen(msg)
        assert not fp.should_propagate(msg)

    def test_ttl_zero_dropped(self) -> None:
        fp = FloodingProtocol()
        msg = _make_message(ttl=0)
        assert not fp.should_propagate(msg)

    def test_ttl_exceeds_max_dropped(self) -> None:
        fp = FloodingProtocol(FloodingConfig(max_ttl=10))
        msg = _make_message(ttl=20)
        assert not fp.should_propagate(msg)

    def test_forward_decrements_ttl(self) -> None:
        fp = FloodingProtocol()
        msg = _make_message(ttl=5)
        forwarded = fp.prepare_forward(msg)
        assert forwarded.ttl == 4

    @pytest.mark.asyncio
    async def test_handle_incoming_calls_handler(self) -> None:
        fp = FloodingProtocol()
        received: list[Message] = []

        async def handler(msg: Message, sender: str) -> None:
            received.append(msg)

        fp.on_message(MessageType.OPTIMAE_ANNOUNCEMENT, handler)
        msg = _make_message()
        result = await fp.handle_incoming(msg, "peer-2")

        assert result is True
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handle_incoming_duplicate_skips_handler(self) -> None:
        fp = FloodingProtocol()
        received: list[Message] = []

        async def handler(msg: Message, sender: str) -> None:
            received.append(msg)

        fp.on_message(MessageType.OPTIMAE_ANNOUNCEMENT, handler)
        msg = _make_message()

        await fp.handle_incoming(msg, "peer-2")
        result = await fp.handle_incoming(msg, "peer-3")

        assert result is False
        assert len(received) == 1  # Only first time
