"""Tests for doin_node.network.discovery.PeerDiscovery."""

from __future__ import annotations

import time

import pytest

from doin_node.network.discovery import (
    DiscoveredPeer,
    PeerDiscovery,
    MAX_KNOWN_PEERS,
    MIN_CONNECTIONS,
    PEER_TTL,
)


# ── Helpers ──────────────────────────────────────────────────────

def make_peer(
    pid: str = "p1",
    address: str = "10.0.0.1",
    port: int = 8470,
    **kwargs,
) -> DiscoveredPeer:
    return DiscoveredPeer(peer_id=pid, address=address, port=port, **kwargs)


@pytest.fixture
def disc():
    return PeerDiscovery("local-node", 8470)


# ── Add peers ────────────────────────────────────────────────────

class TestAddPeers:
    def test_add_peer(self, disc):
        disc.add_peer(make_peer("p1"))
        assert disc.known_count == 1

    def test_dedup_by_endpoint(self, disc):
        disc.add_peer(make_peer("p1", "1.1.1.1", 80))
        disc.add_peer(make_peer("p1-dup", "1.1.1.1", 80))
        assert disc.known_count == 1

    def test_dedup_updates_last_seen(self, disc):
        disc.add_peer(make_peer("p1", last_seen=100.0))
        disc.add_peer(make_peer("p1", last_seen=200.0))
        peer = disc._known_peers["10.0.0.1:8470"]
        assert peer.last_seen == 200.0

    def test_dedup_updates_chain_height(self, disc):
        disc.add_peer(make_peer("p1", chain_height=5))
        disc.add_peer(make_peer("p1", chain_height=10))
        assert disc._known_peers["10.0.0.1:8470"].chain_height == 10

    def test_skip_self(self, disc):
        disc.add_peer(make_peer("local-node"))
        assert disc.known_count == 0

    def test_max_capacity_evicts(self, disc):
        # Fill to max
        for i in range(MAX_KNOWN_PEERS):
            disc.add_peer(make_peer(f"p{i}", "10.0.0.1", 8000 + i))
        assert disc.known_count == MAX_KNOWN_PEERS
        # Adding one more triggers eviction (if stale peers exist)
        # Make some stale
        for ep in list(disc._known_peers)[:10]:
            disc._known_peers[ep].last_seen = time.time() - PEER_TTL - 1
        disc.add_peer(make_peer("new", "99.99.99.99", 1234))
        assert disc.known_count <= MAX_KNOWN_PEERS


# ── Connection tracking ──────────────────────────────────────────

class TestConnectionTracking:
    def test_mark_connected(self, disc):
        disc.add_peer(make_peer("p1"))
        disc.mark_connected("10.0.0.1:8470")
        assert disc.connected_count == 1
        peer = disc._known_peers["10.0.0.1:8470"]
        assert peer.connection_failures == 0
        assert peer.last_connected > 0

    def test_mark_disconnected(self, disc):
        disc.mark_connected("10.0.0.1:8470")
        disc.mark_disconnected("10.0.0.1:8470")
        assert disc.connected_count == 0

    def test_mark_disconnected_unknown(self, disc):
        disc.mark_disconnected("nope")  # no error

    def test_mark_failed(self, disc):
        disc.add_peer(make_peer("p1"))
        disc.mark_failed("10.0.0.1:8470")
        assert disc._known_peers["10.0.0.1:8470"].connection_failures == 1

    def test_mark_failed_unknown(self, disc):
        disc.mark_failed("nope")  # no error

    def test_connected_resets_failures(self, disc):
        disc.add_peer(make_peer("p1"))
        disc.mark_failed("10.0.0.1:8470")
        disc.mark_failed("10.0.0.1:8470")
        disc.mark_connected("10.0.0.1:8470")
        assert disc._known_peers["10.0.0.1:8470"].connection_failures == 0


# ── needs_more_peers ─────────────────────────────────────────────

class TestNeedsMorePeers:
    def test_needs_when_empty(self, disc):
        assert disc.needs_more_peers() is True

    def test_no_need_when_enough(self, disc):
        for i in range(MIN_CONNECTIONS):
            disc.mark_connected(f"10.0.0.{i}:8470")
        assert disc.needs_more_peers() is False


# ── get_connectable_peers ────────────────────────────────────────

class TestConnectable:
    def test_excludes_connected(self, disc):
        disc.add_peer(make_peer("p1"))
        disc.mark_connected("10.0.0.1:8470")
        assert len(disc.get_connectable_peers()) == 0

    def test_excludes_stale(self, disc):
        disc.add_peer(make_peer("p1", last_seen=time.time() - PEER_TTL - 1))
        assert len(disc.get_connectable_peers()) == 0

    def test_excludes_too_many_failures(self, disc):
        disc.add_peer(make_peer("p1"))
        disc._known_peers["10.0.0.1:8470"].connection_failures = 11
        assert len(disc.get_connectable_peers()) == 0

    def test_respects_backoff(self, disc):
        disc.add_peer(make_peer("p1"))
        p = disc._known_peers["10.0.0.1:8470"]
        p.connection_failures = 3
        p.last_connected = time.time()  # just now
        # backoff = 2^3 = 8 seconds, so not connectable yet
        assert len(disc.get_connectable_peers()) == 0

    def test_connectable_after_backoff(self, disc):
        disc.add_peer(make_peer("p1"))
        p = disc._known_peers["10.0.0.1:8470"]
        p.connection_failures = 1
        p.last_connected = time.time() - 10  # 10s ago, backoff=2s
        assert len(disc.get_connectable_peers()) == 1

    def test_sorted_by_failures_then_freshness(self, disc):
        disc.add_peer(make_peer("good", "1.1.1.1", 80, last_seen=time.time()))
        disc.add_peer(make_peer("bad", "2.2.2.2", 80, last_seen=time.time()))
        disc._known_peers["2.2.2.2:80"].connection_failures = 2
        disc._known_peers["2.2.2.2:80"].last_connected = time.time() - 300
        peers = disc.get_connectable_peers()
        assert peers[0].peer_id == "good"

    def test_limit(self, disc):
        for i in range(20):
            disc.add_peer(make_peer(f"p{i}", "10.0.0.1", 8000 + i))
        assert len(disc.get_connectable_peers(limit=5)) == 5


# ── get_peers_for_domain ─────────────────────────────────────────

class TestPeersForDomain:
    def test_filter_by_domain(self, disc):
        disc.add_peer(make_peer("p1", domains=["ml-img"]))
        disc.add_peer(make_peer("p2", "2.2.2.2", 80, domains=["ml-txt"]))
        result = disc.get_peers_for_domain("ml-img")
        assert len(result) == 1
        assert result[0].peer_id == "p1"

    def test_excludes_stale(self, disc):
        disc.add_peer(make_peer("p1", domains=["d1"], last_seen=time.time() - PEER_TTL - 1))
        assert disc.get_peers_for_domain("d1") == []


# ── Stale peer cleanup ──────────────────────────────────────────

class TestCleanup:
    def test_removes_stale(self, disc):
        disc.add_peer(make_peer("stale", last_seen=time.time() - PEER_TTL - 1))
        removed = disc.cleanup()
        assert removed == 1
        assert disc.known_count == 0

    def test_removes_high_failure(self, disc):
        disc.add_peer(make_peer("broken"))
        disc._known_peers["10.0.0.1:8470"].connection_failures = 21
        removed = disc.cleanup()
        assert removed == 1

    def test_keeps_connected(self, disc):
        disc.add_peer(make_peer("connected", last_seen=time.time() - PEER_TTL - 1))
        disc.mark_connected("10.0.0.1:8470")
        removed = disc.cleanup()
        assert removed == 0

    def test_keeps_healthy(self, disc):
        disc.add_peer(make_peer("healthy"))
        assert disc.cleanup() == 0


# ── Exponential backoff ──────────────────────────────────────────

class TestBackoff:
    def test_backoff_increases(self):
        p = make_peer("p1")
        p.connection_failures = 1
        p.last_connected = time.time()
        assert not p.is_connectable  # backoff=2s

        p.connection_failures = 5
        p.last_connected = time.time() - 20  # 20s ago
        assert not p.is_connectable  # backoff=32s

    def test_backoff_caps_at_300(self):
        p = make_peer("p1")
        p.connection_failures = 10
        p.last_connected = time.time() - 301
        assert p.is_connectable  # max backoff=300, waited 301

    def test_no_backoff_zero_failures(self):
        p = make_peer("p1")
        assert p.is_connectable


# ── Stats ────────────────────────────────────────────────────────

class TestStats:
    def test_stats_structure(self, disc):
        disc.add_peer(make_peer("p1", source="pex"))
        disc.mark_connected("10.0.0.1:8470")
        stats = disc.get_stats()
        assert stats["known_peers"] == 1
        assert stats["connected"] == 1
        assert stats["bootstrap_nodes"] == 0
        assert "pex" in stats["by_source"]
        assert stats["by_source"]["pex"] == 1
