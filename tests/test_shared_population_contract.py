from __future__ import annotations

from types import SimpleNamespace

import pytest

from doin_node.unified import (
    UnifiedNode,
    UnifiedNodeConfig,
    _build_shared_population_tx,
    _candidate_rejection_reason,
    _shared_generation_fingerprint,
    _shared_population_fingerprint,
    _shared_population_seed,
)


def test_candidate_rejection_reason_is_explicit_and_fail_closed() -> None:
    assert _candidate_rejection_reason({"fitness": 1.0}) is None
    assert _candidate_rejection_reason({"candidate_rejected": True}) == "candidate_rejected"
    assert _candidate_rejection_reason({
        "candidate_rejected_reason": "policy_action_collapse",
    }) == "policy_action_collapse"
    assert _candidate_rejection_reason({
        "candidate_rejected": False,
        "candidate_rejected_reason": "  collapse  ",
    }) == "collapse"


def test_shared_population_seed_prefers_explicit_campaign_seed() -> None:
    assert _shared_population_seed("domain-a", {"ga_seed": 1701}) == 1701
    assert _shared_population_seed(
        "domain-a",
        {"ga_seed": 1701, "shared_population_seed": 42},
    ) == 42


def test_shared_population_seed_fallback_is_domain_deterministic() -> None:
    first = _shared_population_seed("domain-a", {})
    second = _shared_population_seed("domain-a", {})
    assert first == second
    assert first != _shared_population_seed("domain-b", {})


def test_shared_population_fingerprint_is_canonical() -> None:
    left = {"generation": 0, "population": [{"x": 1}, {"x": 2}]}
    right = {"population": [{"x": 1}, {"x": 2}], "generation": 0}
    assert _shared_population_fingerprint(left) == _shared_population_fingerprint(right)
    right["population"][1]["x"] = 3
    assert _shared_population_fingerprint(left) != _shared_population_fingerprint(right)


def test_shared_generation_fingerprint_ignores_live_fitness_only() -> None:
    left = {
        "generation": 2,
        "stage_idx": 1,
        "population": [{"x": 1}, {"x": 2}],
    }
    right = {
        "generation": 2,
        "stage_idx": 1,
        "population": [{"x": 1, "fitness": 0.5}, {"x": 2, "fitness": -1.0}],
    }
    assert _shared_generation_fingerprint(left) == _shared_generation_fingerprint(right)
    right["population"][1]["x"] = 3
    assert _shared_generation_fingerprint(left) != _shared_generation_fingerprint(right)


def test_shared_results_survive_full_process_restart(tmp_path) -> None:
    domain_id = "domain-a"
    config = UnifiedNodeConfig(port=8491, data_dir=str(tmp_path))
    population = {
        "generation": 3,
        "stage_idx": 1,
        "population": [{"x": 1}, {"x": 2}],
    }
    first = UnifiedNode(config)
    first._shared_pop_state[domain_id] = population
    first._shared_pop_generation[domain_id] = 3
    first._shared_pop_results[domain_id] = {
        0: {
            "fitness": 0.125,
            "candidate_rejected": False,
            "candidate_rejected_reason": None,
            "val_mae": 0.25,
        },
    }
    first._persist_shared_results(domain_id)

    restored_population = {
        "generation": 3,
        "stage_idx": 1,
        "population": [{"x": 1}, {"x": 2}],
    }
    second = UnifiedNode(config)
    second._shared_pop_state[domain_id] = restored_population
    second._shared_pop_generation[domain_id] = 3
    second._shared_pop_results[domain_id] = {}
    second._shared_pop_claims[domain_id] = {0, 1}

    assert second._restore_shared_results(domain_id) == 1
    assert second._shared_pop_results[domain_id][0]["fitness"] == 0.125
    assert second._shared_pop_results[domain_id][0]["val_mae"] == 0.25
    assert restored_population["population"][0]["fitness"] == 0.125
    assert second._shared_pop_claims[domain_id] == {1}


def test_shared_results_reject_different_population(tmp_path) -> None:
    domain_id = "domain-a"
    config = UnifiedNodeConfig(port=8492, data_dir=str(tmp_path))
    first = UnifiedNode(config)
    first._shared_pop_state[domain_id] = {
        "generation": 0,
        "population": [{"x": 1}],
    }
    first._shared_pop_generation[domain_id] = 0
    first._shared_pop_results[domain_id] = {0: {"fitness": 0.5}}
    first._persist_shared_results(domain_id)

    second = UnifiedNode(config)
    second._shared_pop_state[domain_id] = {
        "generation": 0,
        "population": [{"x": 999}],
    }
    second._shared_pop_generation[domain_id] = 0
    second._shared_pop_results[domain_id] = {}

    assert second._restore_shared_results(domain_id) == 0
    assert second._shared_pop_results[domain_id] == {}


@pytest.mark.asyncio
async def test_shared_population_store_is_idempotent_after_peer_commit() -> None:
    node = SimpleNamespace(
        peer_id="peer-a",
        _has_transaction=lambda _tx_id: True,
    )
    await UnifiedNode._store_shared_population_in_chain(
        node,
        "domain-a",
        {"generation": 3, "population": [{"x": 1}]},
    )


@pytest.mark.asyncio
async def test_shared_population_store_is_idempotent_while_pending() -> None:
    pop_state = {"generation": 3, "population": [{"x": 1}]}

    # The pending transaction is built exactly as production builds it:
    # deterministic content, content-derived ID (finding 201 — dedup keys
    # are no longer allowed to masquerade as transaction IDs).
    pending = _build_shared_population_tx("domain-a", pop_state, "peer-a")
    assert pending.id == pending.compute_id()
    node = SimpleNamespace(
        peer_id="peer-a",
        _has_transaction=lambda _tx_id: False,
        consensus=SimpleNamespace(
            state=SimpleNamespace(pending_transactions=[pending]),
        ),
    )

    await UnifiedNode._store_shared_population_in_chain(node, "domain-a", pop_state)


# ── Finding 211: shared populations require explicit chain identity ──

def test_unified_node_refuses_shared_population_without_chain_identity(tmp_path) -> None:
    from doin_node.unified import ChainIdentityConfigError, DomainRole

    role = DomainRole(
        domain_id="shared-domain-211",
        optimize=True,
        optimization_config={"shared_population": True},
    )
    config = UnifiedNodeConfig(
        port=8497, data_dir=str(tmp_path / "n211"), domains=[role],
    )
    with pytest.raises(ChainIdentityConfigError) as ei:
        UnifiedNode(config)
    assert "chain_id" in str(ei.value)
    assert "genesis_hash" in str(ei.value)


def test_unified_node_accepts_shared_population_with_explicit_identity(tmp_path) -> None:
    from doin_node.unified import DomainRole

    role = DomainRole(
        domain_id="shared-domain-211-ok",
        optimize=True,
        optimization_config={"shared_population": True},
    )
    config = UnifiedNodeConfig(
        port=8498,
        data_dir=str(tmp_path / "n211ok"),
        domains=[role],
        chain_id="doin-explicit-fleet",
        genesis_hash="ef" * 32,
    )
    node = UnifiedNode(config)
    assert node.chain_id == "doin-explicit-fleet"
    assert node.expected_genesis_hash == "ef" * 32
