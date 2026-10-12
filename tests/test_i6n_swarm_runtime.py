"""I6-N swarm runtime contract (lane B): tolerance, fail-closed reproduction, result identity, role check."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from doin_core.protocol.messages import Message, MessageType
from doin_node.unified import (
    UnifiedNode,
    UnifiedNodeConfig,
    _shared_improved,
    _shared_reproduction_policy,
    _shared_result_identity_mismatch,
    _shared_role_refusal,
)


def test_improvement_needs_more_than_the_absolute_tolerance():
    assert _shared_improved(0.95, 0.90, False, 1e-4)
    assert not _shared_improved(0.95, 0.94995, False, 1e-4)
    assert _shared_improved(float("inf"), 0.90, False, 1e-4)
    assert not _shared_improved(1.0, None, False, 1e-4)
    assert not _shared_improved(1.0, 1.0, False, 0.0)
    assert _shared_improved(1.0, 1.1, True, 0.0)  # maximizing domains keep their behaviour
    with pytest.raises(ValueError):
        _shared_improved(1.0, 0.5, False, -1.0)


def test_reproduction_policy_is_explicit():
    assert _shared_reproduction_policy({}) == "reevaluate"
    assert _shared_reproduction_policy({"reproduction_failure_policy": "fail_closed"}) == "fail_closed"
    with pytest.raises(ValueError):
        _shared_reproduction_policy({"reproduction_failure_policy": "retry_forever"})


def test_result_identity_binding():
    assert _shared_result_identity_mismatch({"identity": "a" * 64}, {"candidate_identity": "b" * 64})
    assert not _shared_result_identity_mismatch({"identity": "a" * 64}, {"candidate_identity": "a" * 64})
    assert not _shared_result_identity_mismatch({"x": 1}, {"fitness": 1.0})
    # a strict population (I6N wires carry identity) refuses an unbound result
    assert _shared_result_identity_mismatch({"identity": "a" * 64}, {"fitness": 1.0}, strict=True)


def test_role_refusal_comes_from_the_plugin_hook():
    class Plugin:
        def validate_domain_role(self, role):
            if role["higher_is_better"] is not False:
                raise ValueError("must minimize")

    role = SimpleNamespace(higher_is_better=True)
    assert "must minimize" in _shared_role_refusal(Plugin(), role, {"shared_population_seed": 6102026})
    assert _shared_role_refusal(Plugin(), SimpleNamespace(higher_is_better=False), {}) is None
    assert _shared_role_refusal(object(), role, {}) is None


def _node(tmp_path, port):
    node = UnifiedNode(UnifiedNodeConfig(port=port, data_dir=str(tmp_path)))
    domain = "i6n-h12"
    node._shared_pop_generation[domain] = 0
    node._shared_pop_state[domain] = {"generation": 0, "population": [{"identity": "a" * 64}, {"identity": "c" * 64}]}
    node._shared_pop_results[domain] = {}
    return node, domain


def _result_message(domain, identity, tx):
    return Message(msg_type=MessageType.CANDIDATE_EVALUATION, sender_id="peer-b", payload={
        "tx_id": tx, "domain_id": domain, "peer_id": "peer-b",
        "candidate_data": {"_shared_result": True, "generation": 0, "candidate_idx": 0, "fitness": 0.5,
                           "candidate_identity": identity}})


@pytest.mark.asyncio
async def test_peer_result_for_another_candidate_is_not_recorded(tmp_path):
    node, domain = _node(tmp_path, 8611)
    await node._handle_candidate_evaluation(_result_message(domain, "b" * 64, "t1"), "peer-b")
    assert node._shared_pop_results[domain] == {}
    await node._handle_candidate_evaluation(_result_message(domain, "a" * 64, "t2"), "peer-b")
    assert node._shared_pop_results[domain][0]["fitness"] == 0.5
    assert node._shared_pop_results[domain][0]["candidate_identity"] == "a" * 64


def test_persisted_results_keep_candidate_identity(tmp_path):
    node, domain = _node(tmp_path, 8612)
    node._shared_pop_results[domain] = {0: {"fitness": 0.25, "candidate_identity": "a" * 64}}
    node._persist_shared_results(domain)
    again, _ = _node(tmp_path, 8613)
    assert again._restore_shared_results(domain) == 1
    assert again._shared_pop_results[domain][0]["candidate_identity"] == "a" * 64


def test_fail_closed_reproduction_keeps_results_and_stops(tmp_path):
    node, domain = _node(tmp_path, 8614)
    node._shared_pop_results[domain] = {0: {"fitness": 0.25}}
    with pytest.raises(RuntimeError, match="fail_closed"):
        node._on_shared_reproduction_failure(domain, 0, ValueError("boom"), "fail_closed")
    assert node._shared_pop_results[domain] == {0: {"fitness": 0.25}}
    node._on_shared_reproduction_failure(domain, 0, ValueError("boom"), "reevaluate")
    assert node._shared_pop_results[domain] == {}


def test_archive_spool_hook_records_each_reference_once(tmp_path, monkeypatch):
    import sys
    import types
    calls = []

    def ingest_reference_spool(spool, record_transaction, *, known_tx_ids=None):
        for tx_id in ("tx-a", "tx-b", "tx-a"):
            if tx_id in known_tx_ids:
                continue
            record_transaction(SimpleNamespace(id=tx_id))
            known_tx_ids.add(tx_id)
        return {"ingested": 2, "already_known": 1, "refused": []}

    fake = types.ModuleType("doin_node.archive.external")
    fake.ingest_reference_spool = ingest_reference_spool
    monkeypatch.setitem(sys.modules, "doin_node.archive.external", fake)
    node = UnifiedNode(UnifiedNodeConfig(port=8615, data_dir=str(tmp_path), archive_reference_spool=str(tmp_path)))
    monkeypatch.setattr(node, "_has_transaction", lambda tx_id: tx_id == "tx-b")
    monkeypatch.setattr(node.consensus, "record_transaction", lambda tx: calls.append(tx.id))
    assert node.ingest_archive_spool_once()["ingested"] == 2
    assert calls == ["tx-a"]  # tx-b already on chain: never re-recorded


@pytest.mark.asyncio
async def test_rejected_peer_result_without_numeric_fitness_is_recorded_as_worst(tmp_path):
    """Two-node smoke regression: a peer's rejected candidate arrives with fitness null (inf is not JSON);
    it must be recorded as a rejected worst result, persisted, and never crash generation completion."""
    node, domain = _node(tmp_path, 8616)
    msg = Message(msg_type=MessageType.CANDIDATE_EVALUATION, sender_id="peer-b", payload={
        "tx_id": "t9", "domain_id": domain, "peer_id": "peer-b",
        "candidate_data": {"_shared_result": True, "generation": 0, "candidate_idx": 0, "fitness": None,
                           "candidate_rejected": True, "candidate_rejected_reason": "folds not complete",
                           "candidate_identity": "a" * 64}})
    await node._handle_candidate_evaluation(msg, "peer-b")
    recorded = node._shared_pop_results[domain][0]
    assert recorded["candidate_rejected"] is True and recorded["fitness"] == float("inf")
    node._persist_shared_results(domain)
    again, _ = _node(tmp_path, 8617)
    assert again._restore_shared_results(domain) == 1
    restored = again._shared_pop_results[domain][0]
    assert restored["fitness"] == float("inf") and restored["candidate_rejected"] is True


def test_local_rejected_result_snapshot_is_persistable():
    snap = UnifiedNode._shared_result_snapshot({"fitness": float("inf"), "candidate_rejected": True,
                                                "candidate_rejected_reason": "x"})
    assert snap["fitness"] is None and snap["candidate_rejected"] is True
