"""Tests for doin_node.network.gossip.GossipSub."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from doin_core.protocol.messages import Message, MessageType
from doin_node.network.gossip import GossipSub, TopicState


# ── Helpers ──────────────────────────────────────────────────────

def make_message(
    msg_type: MessageType = MessageType.BLOCK_ANNOUNCEMENT,
    sender: str = "sender-1",
    payload: dict | None = None,
) -> Message:
    return Message(msg_type=msg_type, sender_id=sender, payload=payload or {})


@pytest.fixture
def gs():
    """GossipSub with small mesh params for easier testing."""
    g = GossipSub("local", d_target=3, d_low=2, d_high=5)
    g.set_send_fn(AsyncMock(return_value=True))
    g.subscribe_all()
    return g


@pytest.fixture
def populated_gs(gs):
    """GossipSub with 6 peers added."""
    for i in range(6):
        gs.add_peer(f"peer-{i}", {"blocks", "optimae", "tasks", "discovery"})
    return gs


# ── Subscriptions ────────────────────────────────────────────────

class TestSubscriptions:
    def test_subscribe(self):
        g = GossipSub("me")
        g.subscribe("blocks")
        assert "blocks" in g._subscriptions

    def test_unsubscribe(self):
        g = GossipSub("me")
        g.subscribe("blocks")
        g.unsubscribe("blocks")
        assert "blocks" not in g._subscriptions

    def test_unsubscribe_unknown(self):
        g = GossipSub("me")
        g.unsubscribe("nope")  # no error

    def test_subscribe_all(self):
        g = GossipSub("me")
        g.subscribe_all()
        assert g._subscriptions == {"optimae", "blocks", "tasks", "discovery"}


# ── Peer management ─────────────────────────────────────────────

class TestPeers:
    def test_add_peer(self, gs):
        gs.add_peer("p1")
        assert "p1" in gs._known_peers

    def test_add_peer_with_topics(self, gs):
        gs.add_peer("p1", {"blocks"})
        assert gs._peer_topics["p1"] == {"blocks"}

    def test_remove_peer(self, populated_gs):
        # Put peer in mesh first
        populated_gs._topics["blocks"].mesh.add("peer-0")
        populated_gs.remove_peer("peer-0")
        assert "peer-0" not in populated_gs._known_peers
        assert "peer-0" not in populated_gs._topics["blocks"].mesh

    def test_remove_unknown_peer(self, gs):
        gs.remove_peer("nope")  # no error


# ── Publishing ───────────────────────────────────────────────────

class TestPublish:
    @pytest.mark.asyncio
    async def test_publish_to_mesh(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh = {"peer-0", "peer-1"}
        msg = make_message()
        sent = await gs.publish(msg)
        assert sent == 2

    @pytest.mark.asyncio
    async def test_publish_creates_topic_if_needed(self):
        g = GossipSub("me")
        g.set_send_fn(AsyncMock(return_value=True))
        g.add_peer("p1")
        msg = make_message()
        await g.publish(msg)
        assert "blocks" in g._topics

    @pytest.mark.asyncio
    async def test_publish_skips_self(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh = {"local", "peer-0"}
        msg = make_message()
        sent = await gs.publish(msg)
        assert sent == 1  # only peer-0

    @pytest.mark.asyncio
    async def test_publish_uses_fanout_when_unsubscribed(self, populated_gs):
        gs = populated_gs
        gs.unsubscribe("blocks")
        gs._topics["blocks"].fanout = {"peer-0", "peer-1"}
        msg = make_message()
        sent = await gs.publish(msg)
        assert sent == 2


# ── Incoming messages ────────────────────────────────────────────

class TestHandleIncoming:
    @pytest.mark.asyncio
    async def test_new_message_accepted(self, populated_gs):
        msg = make_message()
        result = await populated_gs.handle_incoming(msg, "peer-0")
        assert result is True

    @pytest.mark.asyncio
    async def test_duplicate_rejected(self, populated_gs):
        msg = make_message()
        await populated_gs.handle_incoming(msg, "peer-0")
        result = await populated_gs.handle_incoming(msg, "peer-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_handler_dispatched(self, populated_gs):
        handler = AsyncMock()
        populated_gs.on_message(MessageType.BLOCK_ANNOUNCEMENT, handler)
        msg = make_message()
        await populated_gs.handle_incoming(msg, "peer-0")
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forwards_to_mesh(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh = {"peer-0", "peer-1", "peer-2"}
        msg = make_message()
        await gs.handle_incoming(msg, "peer-0")
        send_fn = gs._send_fn
        # Should forward to peer-1 and peer-2 (not peer-0 sender, not "local")
        targets = {call.args[0] for call in send_fn.call_args_list}
        assert "peer-0" not in targets
        assert "peer-1" in targets
        assert "peer-2" in targets

    @pytest.mark.asyncio
    async def test_updates_peer_score(self, populated_gs):
        gs = populated_gs
        msg = make_message()
        await gs.handle_incoming(msg, "peer-0")
        assert gs._peer_scores["peer-0"].messages_delivered >= 1


# ── Heartbeat ────────────────────────────────────────────────────

class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_graft_when_mesh_small(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh.clear()
        await gs.heartbeat()
        assert len(gs._topics["blocks"].mesh) >= gs.d_low

    @pytest.mark.asyncio
    async def test_prune_when_mesh_large(self, populated_gs):
        gs = populated_gs
        # Add 10 peers to mesh (> d_high=5)
        for i in range(10):
            pid = f"xpeer-{i}"
            gs.add_peer(pid)
            gs._topics["blocks"].mesh.add(pid)
        await gs.heartbeat()
        assert len(gs._topics["blocks"].mesh) <= gs.d_high

    @pytest.mark.asyncio
    async def test_sends_graft_control(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh.clear()
        await gs.heartbeat()
        send_fn = gs._send_fn
        graft_calls = [
            c for c in send_fn.call_args_list
            if c.args[1].get("control") == "GRAFT"
        ]
        assert len(graft_calls) > 0

    @pytest.mark.asyncio
    async def test_rotates_history(self, populated_gs):
        gs = populated_gs
        state = gs._topics["blocks"]
        initial = len(state.recent_messages)
        await gs.heartbeat()
        assert len(state.recent_messages) == initial + 1


# ── Control messages ─────────────────────────────────────────────

class TestControlMessages:
    @pytest.mark.asyncio
    async def test_graft_adds_to_mesh(self, populated_gs):
        gs = populated_gs
        await gs.handle_control("peer-0", "GRAFT", "blocks")
        assert "peer-0" in gs._topics["blocks"].mesh

    @pytest.mark.asyncio
    async def test_graft_rejected_when_full(self, populated_gs):
        gs = populated_gs
        # Fill mesh to d_high
        for i in range(gs.d_high):
            gs._topics["blocks"].mesh.add(f"fill-{i}")
        await gs.handle_control("peer-0", "GRAFT", "blocks")
        assert "peer-0" not in gs._topics["blocks"].mesh
        # Should have sent PRUNE back
        send_fn = gs._send_fn
        assert any(
            c.args[1].get("control") == "PRUNE"
            for c in send_fn.call_args_list
        )

    @pytest.mark.asyncio
    async def test_prune_removes_from_mesh(self, populated_gs):
        gs = populated_gs
        gs._topics["blocks"].mesh.add("peer-0")
        await gs.handle_control("peer-0", "PRUNE", "blocks")
        assert "peer-0" not in gs._topics["blocks"].mesh

    @pytest.mark.asyncio
    async def test_ihave_triggers_iwant(self, populated_gs):
        gs = populated_gs
        await gs.handle_control("peer-0", "IHAVE", "blocks", {"message_ids": ["abc123"]})
        send_fn = gs._send_fn
        iwant_calls = [
            c for c in send_fn.call_args_list
            if c.args[1].get("control") == "IWANT"
        ]
        assert len(iwant_calls) == 1
        assert "abc123" in iwant_calls[0].args[1]["data"]["message_ids"]

    @pytest.mark.asyncio
    async def test_ihave_no_iwant_for_seen(self, populated_gs):
        gs = populated_gs
        msg = make_message()
        msg_id = GossipSub._message_id(msg)
        gs._mark_seen(msg_id, msg)
        await gs.handle_control("peer-0", "IHAVE", "blocks", {"message_ids": [msg_id]})
        send_fn = gs._send_fn
        iwant_calls = [
            c for c in send_fn.call_args_list
            if c.args[1].get("control") == "IWANT"
        ]
        assert len(iwant_calls) == 0

    @pytest.mark.asyncio
    async def test_iwant_sends_cached_message(self, populated_gs):
        gs = populated_gs
        msg = make_message()
        msg_id = GossipSub._message_id(msg)
        gs._mark_seen(msg_id, msg)
        gs._send_fn.reset_mock()
        await gs.handle_control("peer-0", "IWANT", "blocks", {"message_ids": [msg_id]})
        assert gs._send_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_unknown_topic_ignored(self, populated_gs):
        await populated_gs.handle_control("peer-0", "GRAFT", "nonexistent")
        # no error


# ── Message ID ───────────────────────────────────────────────────

class TestMessageId:
    def test_deterministic(self):
        m1 = make_message()
        m2 = Message(
            msg_type=m1.msg_type,
            sender_id=m1.sender_id,
            timestamp=m1.timestamp,
            payload=m1.payload,
        )
        assert GossipSub._message_id(m1) == GossipSub._message_id(m2)

    def test_different_payload_different_id(self):
        m1 = make_message(payload={"a": 1})
        m2 = make_message(payload={"a": 2})
        # timestamps differ too, but payloads certainly differ
        assert GossipSub._message_id(m1) != GossipSub._message_id(m2)


# ── Mesh stats ───────────────────────────────────────────────────

class TestMeshStats:
    def test_stats_structure(self, populated_gs):
        stats = populated_gs.get_mesh_stats()
        assert "blocks" in stats
        assert "mesh_size" in stats["blocks"]
        assert "fanout_size" in stats["blocks"]
        assert "recent_messages" in stats["blocks"]
