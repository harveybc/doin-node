"""Contract for explicit I6N island terminals exposed by the node API."""

from __future__ import annotations

import hashlib
import json

import pytest

from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig


PLAN = "a" * 64


def _role() -> DomainRole:
    return DomainRole(
        domain_id="i6n-island-sequence-h4",
        optimize=True,
        higher_is_better=False,
        optimization_config={
            "i6n_island_mode": True,
            "i6n_horizon": 4,
            "i6n_plan": {"plan_sha256": PLAN},
            "population_size": 20,
            "n_generations": 100,
            "optimization_patience": 10,
            "min_relative_improvement": 1e-8,
            "seed": 6102026,
            "ga_seed": 6102026,
        },
    )


def _terminal(**changes):
    terminal = {
        "status": "COMPLETE",
        "island_id": "gamma-5090",
        "horizon": 4,
        "seed": 6102026,
        "stop_reason": "patience",
        "terminal_generation": 37,
        "champion_sha256": "b" * 64,
        "plan_sha256": PLAN,
        "contract_sha256": "c" * 64,
        "checkpoint_identity_sha256": "d" * 64,
        "champion_fitness": -0.01,
        "final_population_sha256": "e" * 64,
        "candidate_ledger_sha256": "f" * 64,
        "evaluated_candidate_count": 61,
    }
    terminal.update(changes)
    return terminal


class _Plugin:
    def __init__(self, terminal):
        self.terminal = terminal

    def optimize(self, _parameters, _performance):
        return {"filters": 8}, -0.01

    def get_terminal_receipt(self):
        return None if self.terminal is None else dict(self.terminal)


@pytest.mark.asyncio
async def test_completed_optimizer_publishes_config_bound_terminal_in_status(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("I6N_ISLAND_ID", "gamma-5090")
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / "node"), domains=[role],
        storage_backend="json",
    ))

    assert await node._run_full_optimization(role.domain_id, _Plugin(_terminal()))

    expected_config_digest = hashlib.sha256(json.dumps(
        role.optimization_config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    response = await node._http_status(None)
    payload = json.loads(response.text)
    terminal = payload["optimization"]["terminal"]
    assert terminal["status"] == "COMPLETE"
    assert terminal["scientific_config_sha256"] == expected_config_digest
    assert terminal["champion_sha256"] == "b" * 64

    restarted = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / "node"), domains=[role],
        storage_backend="json",
    ))
    assert restarted._optimization_terminals[role.domain_id] == terminal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal, message",
    [
        (None, "explicit terminal"),
        (_terminal(seed=7), "seed"),
        (_terminal(plan_sha256="e" * 64), "plan"),
        (_terminal(stop_reason=None), "stop reason"),
        (_terminal(champion_sha256="not-a-digest"), "champion_sha256"),
        (_terminal(champion_fitness=float("nan")), "champion_fitness"),
        (_terminal(evaluated_candidate_count=0), "evaluated_candidate_count"),
    ],
)
async def test_invalid_or_inferred_terminal_never_completes_the_node(
    tmp_path, monkeypatch, terminal, message,
):
    monkeypatch.setenv("I6N_ISLAND_ID", "gamma-5090")
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / "node"), domains=[role],
        storage_backend="json",
    ))

    with pytest.raises(RuntimeError, match=message):
        await node._run_full_optimization(role.domain_id, _Plugin(terminal))

    assert role.domain_id not in node._optimization_terminals
