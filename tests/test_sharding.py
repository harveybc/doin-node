"""Tests for domain sharding."""

import pytest
from doin_node.network.sharding import DomainShard, ShardConfig, ShardRouter


@pytest.fixture
def shard():
    cfg = ShardConfig(subscribed_domains=["ml", "vision"])
    return DomainShard(cfg)


@pytest.fixture
def full_shard():
    return DomainShard(ShardConfig(relay_all=True))


# --- should_process ---

def test_process_subscribed(shard):
    assert shard.should_process("ml") is True

def test_process_unsubscribed(shard):
    assert shard.should_process("audio") is False

def test_process_relay_all(full_shard):
    assert full_shard.should_process("anything") is True


# --- should_relay ---

def test_relay_subscribed(shard):
    assert shard.should_relay({"domain_id": "ml", "type": "tx"}) is True

def test_relay_unsubscribed(shard):
    assert shard.should_relay({"domain_id": "audio", "type": "tx"}) is False

def test_relay_block_always(shard):
    assert shard.should_relay({"domain_id": "audio", "type": "block"}) is True

def test_relay_all_mode(full_shard):
    assert full_shard.should_relay({"domain_id": "x", "type": "tx"}) is True

def test_relay_missing_domain(shard):
    assert shard.should_relay({"type": "tx"}) is False


# --- peer domains ---

def test_register_and_get_peers(shard):
    shard.register_peer_domains("p1", ["ml", "nlp"])
    shard.register_peer_domains("p2", ["vision"])
    assert shard.get_peers_for_domain("ml") == ["p1"]
    assert shard.get_peers_for_domain("vision") == ["p2"]

def test_get_peers_empty(shard):
    assert shard.get_peers_for_domain("unknown") == []

def test_register_overwrites(shard):
    shard.register_peer_domains("p1", ["ml"])
    shard.register_peer_domains("p1", ["vision"])
    assert shard.get_peers_for_domain("ml") == []
    assert shard.get_peers_for_domain("vision") == ["p1"]


# --- subscribe / unsubscribe ---

def test_subscribe_new(shard):
    assert shard.subscribe("audio") is True
    assert shard.should_process("audio") is True

def test_subscribe_duplicate(shard):
    assert shard.subscribe("ml") is True
    assert shard.config.subscribed_domains.count("ml") == 1

def test_unsubscribe(shard):
    shard.unsubscribe("ml")
    assert shard.should_process("ml") is False

def test_unsubscribe_missing(shard):
    shard.unsubscribe("nope")  # no error


# --- max_relay_domains cap ---

def test_max_relay_cap():
    cfg = ShardConfig(max_relay_domains=3)
    s = DomainShard(cfg)
    for i in range(5):
        s.subscribe(f"d{i}")
    assert len(s.config.subscribed_domains) == 3


# --- shard stats ---

def test_shard_stats(shard):
    shard.register_peer_domains("p1", ["ml", "nlp"])
    stats = shard.get_shard_stats()
    assert stats["subscribed_domains"] == 2
    assert stats["tracked_peers"] == 1
    assert stats["known_domains"] == 2
    assert stats["relay_all"] is False


# --- block tx filtering ---

def test_filter_block_txs(shard):
    block = {"transactions": [
        {"domain_id": "ml", "data": 1},
        {"domain_id": "audio", "data": 2},
        {"domain_id": "vision", "data": 3},
    ]}
    result = shard.filter_block_txs(block)
    assert len(result) == 2
    assert all(tx["domain_id"] in ("ml", "vision") for tx in result)


# --- ShardRouter ---

def test_router_process(shard):
    r = ShardRouter(shard)
    assert r.route({"domain_id": "ml", "type": "tx"}) == "process"

def test_router_drop(shard):
    r = ShardRouter(shard)
    assert r.route({"domain_id": "audio", "type": "tx"}) == "drop"

def test_router_block(shard):
    r = ShardRouter(shard)
    assert r.route({"domain_id": "audio", "type": "block"}) == "process"

def test_router_select_peers(shard):
    shard.register_peer_domains("p1", ["ml"])
    r = ShardRouter(shard)
    assert r.select_peers("ml") == ["p1"]
