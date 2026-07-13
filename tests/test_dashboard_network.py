"""Focused tests for the decentralized consolidated dashboard."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from doin_node.dashboard.routes import (
    _PACKAGE_VERSIONS,
    _blockchain_metrics_payload,
    _build_network_overview,
    _deduplicate_monitor_members,
    _endpoint_http_url,
    _local_monitor_snapshot,
    _peer_endpoint_groups,
)


_DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "src/doin_node/dashboard/templates/dashboard.html"
)


class _FakeTransport:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    async def get_json(self, url, *, timeout_seconds):
        self.calls.append((url, timeout_seconds))
        result = self.payloads[url]
        if isinstance(result, Exception):
            raise result
        return dict(result)


def _node(*, peers=None, transport=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            node_label="omega",
            port=8470,
            domains=[SimpleNamespace(domain_id="trading-domain", higher_is_better=True)],
        ),
        identity=SimpleNamespace(peer_id="local-peer"),
        peer_id="local-peer",
        _start_time=time.time() - 120,
        chaindb=SimpleNamespace(height=8),
        _domain_roles={"trading-domain": object()},
        _current_candidate={
            "domain_id": "trading-domain",
            "stage": 2,
            "candidate_num": 3,
            "total_candidates": 20,
            "timestamp": "2026-07-13T12:00:00Z",
            "candidate_params": {"large": "not exposed"},
        },
        _domain_best={"trading-domain": ({"secret": "redacted"}, 0.04)},
        _optimizer_plugins={
            "trading-domain": SimpleNamespace(
                get_runtime_statistics=lambda: {
                    "candidate_evaluations_total": 202,
                    "candidate_history_source": "/private/history.csv",
                }
            )
        },
        _alerts=[{
            "severity": "warning",
            "category": "local-test",
            "message": "local alert",
            "timestamp": "2026-07-13T12:00:00Z",
        }],
        _alerts_unseen=1,
        _own_addresses={"127.0.0.1", "192.168.1.10", "100.64.0.10", "localhost"},
        _peers=peers or {},
        discovery=None,
        transport=transport,
    )


def test_local_monitor_is_compact_and_human_labeled() -> None:
    snapshot = _local_monitor_snapshot(_node())

    assert snapshot["node_label"] == "omega"
    assert snapshot["chain_height"] == 8
    assert snapshot["candidate"]["candidate_num"] == 3
    assert snapshot["optimization_history"] == {
        "candidate_evaluations_total": 202,
        "domains": {"trading-domain": {"candidate_evaluations_total": 202}},
    }
    assert "candidate_params" not in snapshot["candidate"]
    assert snapshot["best_performance"] == {"trading-domain": 0.04}
    assert snapshot["alerts_count"] == 1
    assert snapshot["known_endpoints"] == [
        "100.64.0.10:8470", "127.0.0.1:8470", "192.168.1.10:8470",
    ]
    assert snapshot["versions"] == _PACKAGE_VERSIONS
    assert "predictor" not in snapshot["versions"]


def test_peer_groups_deduplicate_alternate_routes() -> None:
    peers = {
        "192.168.1.2:8470": SimpleNamespace(peer_id="peer-a"),
        "100.64.0.2:8470": SimpleNamespace(peer_id="peer-a"),
        "192.168.1.3:8471": SimpleNamespace(peer_id="peer-b"),
    }

    node = _node(peers=peers)
    node.discovery = SimpleNamespace(_known_peers={
        "100.64.0.3:8471": SimpleNamespace(peer_id="peer-b"),
        "192.168.1.4:8470": SimpleNamespace(peer_id="peer-c"),
    })

    assert _peer_endpoint_groups(node) == [
        ("peer-a", ["192.168.1.2:8470", "100.64.0.2:8470"]),
        ("peer-b", ["192.168.1.3:8471", "100.64.0.3:8471"]),
        ("peer-c", ["192.168.1.4:8470"]),
    ]


def test_endpoint_url_supports_ipv4_and_ipv6() -> None:
    assert _endpoint_http_url("192.168.1.2:8470", "/api/monitor") == (
        "http://192.168.1.2:8470/api/monitor"
    )
    assert _endpoint_http_url("fd00::2:8470", "/dashboard") == (
        "http://[fd00::2]:8470/dashboard"
    )


def test_route_snapshots_are_merged_by_resolved_peer_identity() -> None:
    local = {
        "peer_id": "local-peer", "status": "online", "endpoint": "local",
        "known_endpoints": [], "is_local": True,
    }
    duplicate_local = {
        "peer_id": "local-peer", "status": "online",
        "endpoint": "192.168.1.10:8470",
        "known_endpoints": ["192.168.1.10:8470"],
    }
    offline = {
        "peer_id": "remote-peer", "status": "offline",
        "endpoint": "192.168.1.20:8470",
        "known_endpoints": ["192.168.1.20:8470"],
    }
    online = {
        "peer_id": "remote-peer", "status": "online",
        "endpoint": "100.64.0.20:8470",
        "known_endpoints": ["100.64.0.20:8470"],
    }

    members = _deduplicate_monitor_members([local, duplicate_local, offline, online])

    assert [member["peer_id"] for member in members] == ["local-peer", "remote-peer"]
    assert members[0]["endpoint"] == "local"
    assert members[0]["known_endpoints"] == ["192.168.1.10:8470"]
    assert members[1]["status"] == "online"
    assert members[1]["known_endpoints"] == [
        "192.168.1.20:8470", "100.64.0.20:8470",
    ]


def test_unresolved_route_is_removed_when_online_peer_advertises_it() -> None:
    online = {
        "peer_id": "remote-peer", "status": "online",
        "endpoint": "100.64.0.20:8470",
        "known_endpoints": ["100.64.0.20:8470", "192.168.1.20:8470"],
    }
    unresolved_alias = {
        "peer_id": "192.168.1.20:8470", "status": "offline",
        "endpoint": "192.168.1.20:8470",
        "known_endpoints": ["192.168.1.20:8470"],
    }
    unrelated_offline = {
        "peer_id": "192.168.1.30:8470", "status": "offline",
        "endpoint": "192.168.1.30:8470",
        "known_endpoints": ["192.168.1.30:8470"],
    }

    members = _deduplicate_monitor_members([
        online, unresolved_alias, unrelated_offline,
    ])

    assert [member["peer_id"] for member in members] == [
        "remote-peer", "192.168.1.30:8470",
    ]


def test_dashboard_exposes_network_as_initial_monitoring_view() -> None:
    html = _DASHBOARD.read_text()

    assert 'href="#tabNetwork"' in html
    assert 'class="tab-pane fade show active" id="tabNetwork"' in html
    assert 'id="networkNodesBody"' in html
    assert 'id="networkAlertsBody"' in html
    assert 'href="#tabChampions"' in html
    assert 'id="championHistoryBody"' in html
    assert "candidate_evaluations" in html
    assert "resolvePeerLabel(metric.peer_id)" in html
    assert "resolvePeerLabel(metric.generator_id)" in html
    assert "api('/api/network')" in html


def test_blockchain_history_attributes_candidates_and_champions() -> None:
    def transaction(kind, peer_id, tx_id, performance=None):
        payload = {}
        if performance is not None:
            payload = {
                "verified_performance": performance,
                "champion_metrics": {"validation_selection_score": performance},
            }
        return SimpleNamespace(
            tx_type=SimpleNamespace(value=kind),
            peer_id=peer_id,
            domain_id="trading-domain",
            payload=payload,
            id=tx_id,
            timestamp=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        )

    blocks = [
        SimpleNamespace(
            header=SimpleNamespace(
                index=0, generator_id="genesis",
                timestamp=datetime(2026, 7, 13, 11, 0, tzinfo=timezone.utc),
            ),
            transactions=[],
        ),
        SimpleNamespace(
            header=SimpleNamespace(
                index=1, generator_id="generator-alpha-full",
                timestamp=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
            ),
            transactions=[
                transaction("candidate_evaluated", "peer-alpha-full", "candidate-1"),
                transaction("candidate_evaluated", "peer-alpha-full", "candidate-2"),
                transaction("optimae_accepted", "peer-alpha-full", "accepted-1", 0.10),
            ],
        ),
        SimpleNamespace(
            header=SimpleNamespace(
                index=2, generator_id="generator-beta-full",
                timestamp=datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc),
            ),
            transactions=[
                transaction("candidate_evaluated", "peer-beta-full", "candidate-3"),
                transaction("optimae_accepted", "peer-beta-full", "accepted-2", 0.08),
                transaction("optimae_accepted", "peer-beta-full", "accepted-3", 0.15),
            ],
        ),
    ]
    node = _node()
    node.chaindb = SimpleNamespace(
        height=len(blocks),
        get_block=lambda index: blocks[index],
    )

    payload = _blockchain_metrics_payload(node, limit=2)

    assert payload["candidate_evaluations"]["total_committed"] == 3
    assert payload["candidate_evaluations"]["by_peer"] == [
        {
            "peer_id": "peer-alpha-f", "count": 2, "last_block_index": 1,
            "last_timestamp": "2026-07-13T12:00:00+00:00",
            "domains": {"trading-domain": 2},
        },
        {
            "peer_id": "peer-beta-fu", "count": 1, "last_block_index": 2,
            "last_timestamp": "2026-07-13T12:00:00+00:00",
            "domains": {"trading-domain": 1},
        },
    ]
    assert payload["champion_improvements"] == 2
    assert len(payload["metrics"]) == 2
    assert payload["metrics"][0]["is_improvement"] is False
    assert payload["metrics"][1]["is_improvement"] is True
    assert payload["metrics"][1]["improvement_num"] == 2
    assert payload["metrics"][1]["peer_id"] == "peer-beta-fu"
    assert payload["metrics"][1]["generator_id"] == "generator-be"


@pytest.mark.asyncio
async def test_network_overview_falls_back_and_preserves_offline_member() -> None:
    peer_a = {
        "node_label": "dragon",
        "peer_id": "peer-a",
        "status": "online",
        "chain_height": 8,
        "versions": {**_PACKAGE_VERSIONS, "doin-node": "oldrev0"},
        "candidate": {"candidate_num": 7},
        "optimization_history": {"candidate_evaluations_total": 24},
        "alerts": [{
            "severity": "critical",
            "category": "remote-test",
            "message": "remote alert",
            "timestamp": "2026-07-13T12:01:00Z",
        }],
        "alerts_count": 1,
        "alerts_unseen": 1,
        "known_endpoints": ["192.168.1.2:8470", "100.64.0.2:8470"],
    }
    payloads = {
        "http://192.168.1.2:8470/api/monitor": OSError("LAN unavailable"),
        "http://100.64.0.2:8470/api/monitor": peer_a,
        "http://192.168.1.3:8471/api/monitor": TimeoutError("offline"),
    }
    peers = {
        "192.168.1.2:8470": SimpleNamespace(peer_id="peer-a"),
        "100.64.0.2:8470": SimpleNamespace(peer_id="peer-a"),
        "192.168.1.3:8471": SimpleNamespace(peer_id="peer-b"),
    }
    node = _node(peers=peers, transport=_FakeTransport(payloads))

    result = await _build_network_overview(node)

    assert result["summary"] == {
        "total_nodes": 3,
        "online_nodes": 2,
        "offline_nodes": 1,
        "active_candidates": 2,
        "local_candidate_evaluations": 226,
        "alerts_total": 2,
        "alerts_unseen": 2,
        "version_mismatch_nodes": 1,
        "chain_min": 8,
        "chain_max": 8,
    }
    dragon = result["members"][1]
    assert dragon["endpoint"] == "100.64.0.2:8470"
    assert dragon["dashboard_url"] == "http://100.64.0.2:8470/dashboard"
    assert dragon["version_mismatches"]["doin-node"] == {
        "expected": _PACKAGE_VERSIONS["doin-node"],
        "actual": "oldrev0",
    }
    assert result["members"][2]["status"] == "offline"
    assert result["alerts"][0]["node_label"] == "dragon"
