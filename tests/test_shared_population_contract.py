from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from doin_node.unified import (
    UnifiedNode,
    _shared_generation_fingerprint,
    _shared_population_fingerprint,
    _shared_population_seed,
)


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
    fingerprint = _shared_population_fingerprint(pop_state)

    from doin_core.models.transaction import Transaction, TransactionType

    pending = Transaction(
        id=hashlib.sha256(
            f"shared_pop:domain-a:3:{fingerprint}".encode()
        ).hexdigest(),
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="domain-a",
        peer_id="peer-a",
        payload={},
    )
    node = SimpleNamespace(
        peer_id="peer-a",
        _has_transaction=lambda _tx_id: False,
        consensus=SimpleNamespace(
            state=SimpleNamespace(pending_transactions=[pending]),
        ),
    )

    await UnifiedNode._store_shared_population_in_chain(node, "domain-a", pop_state)
