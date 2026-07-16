from __future__ import annotations

from doin_node.unified import (
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
