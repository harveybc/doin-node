"""Focused loader / plugin-setup / identity tests for the unified node CLI.

Covers requirements R01–R15 of DOIN-CONFIG-001: every ``UnifiedNodeConfig``
and ``DomainRole`` field must be materialized from JSON, each plugin must
receive its own (copied) config subtree with the declared fallback and
injection rules, and identity precedence must be deterministic.

These tests do not start a node, touch the network, load real plugins,
fit models, or generate identities.  Helpers are exercised as pure functions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from doin_node import cli
from doin_node.unified import DomainRole, UnifiedNodeConfig
from doin_core.consensus import IncentiveConfig
from doin_core.models import ResourceLimits
from doin_core.models.fee_market import FeeConfig

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _write(tmp_path: Path, obj: dict) -> str:
    p = tmp_path / "node.json"
    p.write_text(json.dumps(obj))
    return str(p)


# A config that sets EVERY top-level field to a non-default value so a dropped
# field is detectable (a coincidental default match cannot hide it).
FULL_NODE = {
    "node_label": "test-node",
    "host": "127.0.0.1",
    "port": 9999,
    "data_dir": "/tmp/xdata",
    "identity_file": "/tmp/x/id.pem",
    "bootstrap_peers": ["1.2.3.4:8470"],
    "target_block_time": 123.0,
    "initial_threshold": 0.5,
    "acceptance_tolerance": 2e-4,
    "quorum_min_evaluators": 7,
    "quorum_fraction": 0.9,
    "quorum_tolerance": 0.11,
    "commit_reveal_max_age": 111.0,
    "finality_confirmation_depth": 9,
    "external_anchor_interval": 50,
    "require_deterministic_seed": False,
    "eval_poll_interval": 3.0,
    "eval_max_concurrent": 8,
    "optimizer_loop_interval": 15.0,
    "shared_claim_timeout": 3600.0,
    "shared_claim_result_patience": 20,
    "shared_min_peers": 3,
    "shared_initialize_before_peers": True,
    "shared_peer_wait_timeout": 45.0,
    "shared_claim_settle_seconds": 2.0,
    "shared_claim_confirmation_rounds": 3,
    "storage_backend": "json",
    "db_path": "/tmp/x/chain.db",
    "snapshot_interval": 42,
    "prune_keep_blocks": 55,
    "network_protocol": "flooding",
    "gossip_heartbeat_interval": 2.5,
    "discovery_enabled": False,
    "discovery_interval": 7.0,
    "dashboard_enabled": False,
    "experiment_stats_file": "/tmp/x/stats.csv",
    "olap_db_path": "/tmp/x/olap.db",
    "reset_chain": True,
    "fee_market_enabled": False,
    "fee_config": {
        "min_base_fee": 0.5,
        "max_base_fee": 50.0,
        "target_block_fullness": 0.7,
        "base_fee_change_denom": 4,
        "target_block_size": 10,
        "max_block_size": 20,
        "optimae_stake_multiplier": 3.0,
        "optimae_burn_fraction": 0.1,
    },
    "domains": [
        {
            "domain_id": "d-full",
            "optimize": True,
            "evaluate": True,
            "optimization_plugin": "opt_plug",
            "inference_plugin": "inf_plug",
            "synthetic_data_plugin": "syn_plug",
            "has_synthetic_data": True,
            "synthetic_data_validation": False,
            "higher_is_better": False,
            "target_performance": 0.42,
            "metric_type": "binary",
            "optimization_config": {"a": 1, "metric_type": "binary"},
            "inference_config": {"b": 2},
            "synthetic_data_config": {"c": 3},
            "param_bounds": {"lr": [1e-5, 0.01], "units": [16, 64]},
            "resource_limits": {
                "max_training_seconds": 100.0,
                "max_memory_mb": 2048.0,
                "max_epochs": 33,
                "max_batch_size": 8,
            },
            "incentive_config": {
                "higher_is_better": False,
                "tolerance_margin": 0.28,
                "bonus_threshold": 0.06,
                "min_reward_fraction": 0.25,
                "max_bonus_multiplier": 1.3,
            },
        }
    ],
}


# ── Loader tests ─────────────────────────────────────────────────────

def test_r05_every_top_level_field_materialized(tmp_path):
    cfg = cli.load_config(_write(tmp_path, FULL_NODE), {})
    assert cfg.node_label == "test-node"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999
    assert cfg.data_dir == "/tmp/xdata"
    assert cfg.identity_file == "/tmp/x/id.pem"
    assert cfg.bootstrap_peers == ["1.2.3.4:8470"]
    assert cfg.target_block_time == 123.0
    assert cfg.initial_threshold == 0.5
    assert cfg.acceptance_tolerance == 2e-4
    assert cfg.quorum_min_evaluators == 7
    assert cfg.quorum_fraction == 0.9
    assert cfg.quorum_tolerance == 0.11
    assert cfg.commit_reveal_max_age == 111.0
    assert cfg.finality_confirmation_depth == 9
    assert cfg.external_anchor_interval == 50
    assert cfg.require_deterministic_seed is False
    assert cfg.eval_poll_interval == 3.0
    assert cfg.eval_max_concurrent == 8
    assert cfg.optimizer_loop_interval == 15.0
    assert cfg.shared_claim_timeout == 3600.0
    assert cfg.shared_claim_result_patience == 20
    assert cfg.shared_min_peers == 3
    assert cfg.shared_initialize_before_peers is True
    assert cfg.shared_peer_wait_timeout == 45.0
    assert cfg.shared_claim_settle_seconds == 2.0
    assert cfg.shared_claim_confirmation_rounds == 3
    assert cfg.storage_backend == "json"
    assert cfg.db_path == "/tmp/x/chain.db"
    assert cfg.snapshot_interval == 42
    assert cfg.prune_keep_blocks == 55
    assert cfg.network_protocol == "flooding"
    assert cfg.gossip_heartbeat_interval == 2.5
    assert cfg.discovery_enabled is False
    assert cfg.discovery_interval == 7.0
    assert cfg.dashboard_enabled is False
    assert cfg.experiment_stats_file == "/tmp/x/stats.csv"
    assert cfg.olap_db_path == "/tmp/x/olap.db"
    assert cfg.reset_chain is True
    assert cfg.fee_market_enabled is False


def test_r06_nested_fee_config_materialized(tmp_path):
    cfg = cli.load_config(_write(tmp_path, FULL_NODE), {})
    assert isinstance(cfg.fee_config, FeeConfig)
    assert cfg.fee_config.min_base_fee == 0.5
    assert cfg.fee_config.max_base_fee == 50.0
    assert cfg.fee_config.target_block_fullness == 0.7
    assert cfg.fee_config.base_fee_change_denom == 4
    assert cfg.fee_config.target_block_size == 10
    assert cfg.fee_config.max_block_size == 20
    assert cfg.fee_config.optimae_stake_multiplier == 3.0
    assert cfg.fee_config.optimae_burn_fraction == 0.1


def test_r01_r02_r03_r04_every_domain_field_materialized(tmp_path):
    cfg = cli.load_config(_write(tmp_path, FULL_NODE), {})
    role = cfg.domains[0]
    assert role.domain_id == "d-full"
    assert role.optimize is True
    assert role.evaluate is True
    assert role.optimization_plugin == "opt_plug"
    assert role.inference_plugin == "inf_plug"
    assert role.synthetic_data_plugin == "syn_plug"
    assert role.has_synthetic_data is True
    assert role.synthetic_data_validation is False
    assert role.higher_is_better is False
    assert role.target_performance == 0.42
    # R01: new fields
    assert role.metric_type == "binary"
    assert role.inference_config == {"b": 2}
    assert role.synthetic_data_config == {"c": 3}
    assert role.optimization_config == {"a": 1, "metric_type": "binary"}
    # R02: param bounds converted to tuples
    assert role.param_bounds == {"lr": (1e-5, 0.01), "units": (16, 64)}
    assert isinstance(role.param_bounds["lr"], tuple)
    # R03: nested ResourceLimits is a real dataclass instance with the values
    assert isinstance(role.resource_limits, ResourceLimits)
    assert role.resource_limits.max_training_seconds == 100.0
    assert role.resource_limits.max_memory_mb == 2048.0
    assert role.resource_limits.max_epochs == 33
    assert role.resource_limits.max_batch_size == 8
    # R04: nested IncentiveConfig with exact direction/tolerance semantics
    assert isinstance(role.incentive_config, IncentiveConfig)
    assert role.incentive_config.higher_is_better is False
    assert role.incentive_config.tolerance_margin == 0.28
    assert role.incentive_config.bonus_threshold == 0.06
    assert role.incentive_config.min_reward_fraction == 0.25
    assert role.incentive_config.max_bonus_multiplier == 1.3


def test_r13_omega_example_values():
    cfg = cli.load_config(str(EXAMPLES / "predictor_omega_node_tft_binary_neat.json"), {})
    assert cfg.storage_backend == "sqlite"
    assert cfg.network_protocol == "flooding"
    assert cfg.discovery_enabled is True
    assert cfg.fee_market_enabled is False
    role = cfg.domains[0]
    assert role.incentive_config.tolerance_margin == 0.28
    assert role.incentive_config.higher_is_better is False
    assert role.metric_type == "binary"
    assert role.synthetic_data_validation is False
    assert role.higher_is_better is False


def test_r13_gamma_example_three_distinct_subtrees():
    cfg = cli.load_config(str(EXAMPLES / "predictor_gamma_node_tft_binary_neat.json"), {})
    role = cfg.domains[0]
    # Representative characteristic fields (not just identity)
    assert role.optimization_config["shared_population"] is True
    assert role.inference_config["predictor_plugin"] == "binary_tft"
    assert "model_file" in role.synthetic_data_config
    # Three genuinely distinct dictionaries
    assert role.inference_config != role.optimization_config
    assert role.synthetic_data_config != role.optimization_config
    assert role.inference_config is not role.optimization_config
    assert role.synthetic_data_config is not role.optimization_config
    # inference subtree lacks optimizer-only keys -> genuinely separate content
    assert "optimizer_plugin" not in role.inference_config
    assert "shared_population" not in role.inference_config


@pytest.mark.parametrize(
    "campaign",
    [
        "phase_1_asset_policy_btcusdt_1h_shared_v1",
        "phase_1_asset_policy_adausdt_1h_shared_v1",
        "phase_1_asset_policy_eurusd_4h_shared_v1",
        "phase_1_asset_policy_dogeusdt_4h_shared_v1",
    ],
)
@pytest.mark.parametrize(
    "worker_config",
    ["omega_node.json", "dragon_node.json", "gamma_5070ti_node.json", "gamma_5090_node.json"],
)
def test_phase_1_fleet_configs_require_the_complete_swarm(campaign, worker_config):
    cfg = cli.load_config(str(EXAMPLES / "trading" / campaign / worker_config), {})
    assert cfg.shared_min_peers == 3
    assert cfg.shared_peer_wait_timeout == 60
    assert cfg.shared_claim_settle_seconds == 2
    assert cfg.shared_claim_confirmation_rounds == 2
    expected_routes = {
        "omega_node.json": {
            "100.110.215.85:8470",
            "100.107.204.49:8470",
            "100.107.204.49:8471",
        },
        "dragon_node.json": {
            "100.99.54.79:8470",
            "100.107.204.49:8470",
            "100.107.204.49:8471",
        },
        "gamma_5070ti_node.json": {
            "100.99.54.79:8470",
            "100.110.215.85:8470",
            "127.0.0.1:8471",
        },
        "gamma_5090_node.json": {
            "127.0.0.1:8470",
            "100.99.54.79:8470",
            "100.110.215.85:8470",
        },
    }
    assert set(cfg.bootstrap_peers) == expected_routes[worker_config]


@pytest.mark.parametrize(
    "campaign",
    [
        "phase_1_asset_policy_btcusdt_1h_shared_v2",
        "phase_1_asset_policy_adausdt_1h_shared_v2",
        "phase_1_asset_policy_eurusd_4h_shared_v2",
        "phase_1_asset_policy_dogeusdt_4h_shared_v2",
    ],
)
@pytest.mark.parametrize(
    "worker_config",
    ["omega_node.json", "dragon_node.json", "gamma_5070ti_node.json", "gamma_5090_node.json"],
)
def test_phase_1_v2_fleet_initializes_before_full_compute_barrier(campaign, worker_config):
    cfg = cli.load_config(str(EXAMPLES / "trading" / campaign / worker_config), {})
    assert cfg.shared_initialize_before_peers is True
    assert cfg.shared_min_peers == 3


def test_r09_r10_absent_new_subtrees_default_empty(tmp_path):
    obj = {
        "domains": [
            {
                "domain_id": "legacy",
                "optimize": True,
                "evaluate": True,
                "optimization_plugin": "opt",
                "inference_plugin": "inf",
                "synthetic_data_plugin": "syn",
                "optimization_config": {"k": 1},
            }
        ]
    }
    cfg = cli.load_config(_write(tmp_path, obj), {})
    role = cfg.domains[0]
    # No inference/synthetic subtree declared -> empty, which drives the
    # setup_plugins legacy fallback to optimization_config.
    assert role.inference_config == {}
    assert role.synthetic_data_config == {}
    assert role.metric_type == ""


def test_r14_malformed_bounds_fail_clearly(tmp_path):
    bad_cases = [
        {"lr": [1.0]},              # wrong length
        {"lr": [1.0, 2.0, 3.0]},    # wrong length
        {"lr": "not-a-list"},       # not a sequence
        {"lr": [1.0, "x"]},         # non-numeric
        {"lr": [True, 2.0]},        # boolean rejected
        {"lr": [float("inf"), 2]},  # non-finite
        {"lr": [5.0, 1.0]},         # lower > upper
    ]
    for bad in bad_cases:
        obj = {"domains": [{"domain_id": "dbad", "param_bounds": bad}]}
        with pytest.raises(Exception) as ei:
            cli.load_config(_write(tmp_path, obj), {})
        assert "dbad" in str(ei.value), f"error must name domain for {bad!r}"
        assert "lr" in str(ei.value), f"error must name parameter for {bad!r}"


def test_r03_r14_invalid_nested_sections_fail_clearly(tmp_path):
    # Non-dict resource_limits
    obj = {"domains": [{"domain_id": "drl", "resource_limits": [1, 2]}]}
    with pytest.raises(Exception) as ei:
        cli.load_config(_write(tmp_path, obj), {})
    assert "drl" in str(ei.value) and "resource_limits" in str(ei.value)

    # Bad type inside resource_limits (pydantic ValidationError, wrapped)
    obj = {"domains": [{"domain_id": "drl2", "resource_limits": {"max_epochs": "lots"}}]}
    with pytest.raises(Exception) as ei:
        cli.load_config(_write(tmp_path, obj), {})
    assert "drl2" in str(ei.value) and "resource_limits" in str(ei.value)

    # Unknown resource key must not be silently discarded by pydantic.
    obj = {"domains": [{"domain_id": "drl3", "resource_limits": {"max_epohcs": 5}}]}
    with pytest.raises(Exception) as ei:
        cli.load_config(_write(tmp_path, obj), {})
    assert "drl3" in str(ei.value) and "max_epohcs" in str(ei.value)

    # Unknown key inside incentive_config (dataclass TypeError, wrapped)
    obj = {"domains": [{"domain_id": "dic", "incentive_config": {"nope": 1}}]}
    with pytest.raises(Exception) as ei:
        cli.load_config(_write(tmp_path, obj), {})
    assert "dic" in str(ei.value) and "incentive_config" in str(ei.value)

    # Non-dict incentive_config
    obj = {"domains": [{"domain_id": "dic2", "incentive_config": 5}]}
    with pytest.raises(Exception) as ei:
        cli.load_config(_write(tmp_path, obj), {})
    assert "dic2" in str(ei.value) and "incentive_config" in str(ei.value)


def test_r05_empty_object_matches_defaults(tmp_path):
    cfg = cli.load_config(_write(tmp_path, {}), {})
    default = UnifiedNodeConfig()
    for f in [
        "host", "port", "data_dir", "identity_file", "bootstrap_peers",
        "target_block_time", "initial_threshold", "acceptance_tolerance",
        "quorum_min_evaluators", "quorum_fraction", "quorum_tolerance",
        "commit_reveal_max_age", "finality_confirmation_depth",
        "external_anchor_interval", "require_deterministic_seed",
        "eval_poll_interval", "eval_max_concurrent", "optimizer_loop_interval",
        "shared_claim_timeout", "shared_claim_result_patience",
        "shared_min_peers", "shared_peer_wait_timeout",
        "shared_claim_settle_seconds", "shared_claim_confirmation_rounds",
        "storage_backend", "db_path", "snapshot_interval", "prune_keep_blocks",
        "network_protocol", "gossip_heartbeat_interval", "discovery_enabled",
        "discovery_interval", "dashboard_enabled", "experiment_stats_file",
        "olap_db_path", "reset_chain", "fee_market_enabled",
    ]:
        assert getattr(cfg, f) == getattr(default, f), f"field {f} default drifted"
    assert cfg.domains == []
    # nested fee_config is a real FeeConfig equal to the default
    assert isinstance(cfg.fee_config, FeeConfig)
    assert cfg.fee_config == UnifiedNodeConfig().fee_config


def test_r12_role_configs_independent_from_raw(tmp_path):
    path = _write(tmp_path, FULL_NODE)
    cfg = cli.load_config(path, {})
    raw = json.loads(Path(path).read_text())
    role = cfg.domains[0]
    raw_dom = raw["domains"][0]
    # Stored role config subtrees must be independent objects from a fresh
    # decode of the same JSON (loader must not alias/keep the raw dict).
    assert role.optimization_config == raw_dom["optimization_config"]
    assert role.optimization_config is not raw_dom["optimization_config"]
    assert role.inference_config is not raw_dom["inference_config"]
    assert role.synthetic_data_config is not raw_dom["synthetic_data_config"]


# ── Plugin-setup tests ───────────────────────────────────────────────

class _RecordingPlugin:
    """Base recording fake plugin — captures the exact dict it was configured with."""

    def __init__(self):
        self.received = None

    def configure(self, config):
        self.received = config


class _OptPlugin(_RecordingPlugin):
    pass


class _InfPlugin(_RecordingPlugin):
    pass


class _SynPlugin(_RecordingPlugin):
    pass


class _MutatingPlugin(_RecordingPlugin):
    """Deliberately hostile plugin: mutates the dict it is handed."""

    def configure(self, config):
        self.received = config
        config["POISON"] = True
        config["metric_type"] = "MUTATED"


class _NestedMutatingPlugin(_RecordingPlugin):
    """Hostile plugin that mutates a nested list in its config."""

    def configure(self, config):
        self.received = config
        config["nested"]["values"].append("POISON")


class _FakeNode:
    """Minimal stand-in exposing exactly what setup_plugins() needs."""

    def __init__(self, roles):
        self._domain_roles = {r.domain_id: r for r in roles}
        self._optimizer_plugins = {}
        self._evaluator_plugins = {}
        self._synthetic_plugins = {}

    def register_optimizer_plugin(self, domain_id, plugin):
        self._optimizer_plugins[domain_id] = plugin

    def register_evaluator_plugin(self, domain_id, plugin):
        self._evaluator_plugins[domain_id] = plugin

    def register_synthetic_plugin(self, domain_id, plugin):
        self._synthetic_plugins[domain_id] = plugin


def _patch_loaders(monkeypatch, opt=_OptPlugin, inf=_InfPlugin, syn=_SynPlugin):
    monkeypatch.setattr(cli, "load_optimization_plugin", lambda name: opt)
    monkeypatch.setattr(cli, "load_inference_plugin", lambda name: inf)
    monkeypatch.setattr(cli, "load_synthetic_data_plugin", lambda name: syn)


def test_r08_r09_r10_each_plugin_receives_intended_subtree(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        optimization_config={"opt": 1},
        inference_config={"inf": 1},
        synthetic_data_config={"syn": 1},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert node._optimizer_plugins["d"].received["opt"] == 1
    assert node._evaluator_plugins["d"].received["inf"] == 1
    assert node._synthetic_plugins["d"].received["syn"] == 1
    # Each is its own subtree, not cross-contaminated
    assert "inf" not in node._optimizer_plugins["d"].received
    assert "opt" not in node._evaluator_plugins["d"].received
    assert "opt" not in node._synthetic_plugins["d"].received


def test_r09_inference_fallback_to_optimization_config(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=False,
        evaluate=True,
        inference_plugin="inf",
        optimization_config={"marker": "opt"},
        inference_config={},  # empty -> fallback
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert node._evaluator_plugins["d"].received["marker"] == "opt"
    # fallback must be a copy, not the stored role dict
    assert node._evaluator_plugins["d"].received is not role.optimization_config


def test_r10_synthetic_fallback_to_optimization_config(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        evaluate=True,
        synthetic_data_plugin="syn",
        optimization_config={"marker": "opt"},
        synthetic_data_config={},  # empty -> fallback
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert node._synthetic_plugins["d"].received["marker"] == "opt"
    assert node._synthetic_plugins["d"].received is not role.optimization_config


def test_r11_metric_type_injection_and_no_overwrite(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        metric_type="domain_metric",
        # optimizer already declares metric_type -> must NOT be overwritten
        optimization_config={"metric_type": "opt_explicit"},
        # inference lacks metric_type -> domain metric injected
        inference_config={"x": 1},
        # synthetic lacks metric_type and must NOT get one injected
        synthetic_data_config={"y": 1},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert node._optimizer_plugins["d"].received["metric_type"] == "opt_explicit"
    assert node._evaluator_plugins["d"].received["metric_type"] == "domain_metric"
    assert "metric_type" not in node._synthetic_plugins["d"].received


def test_r11_param_bounds_only_in_optimizer(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        optimization_config={"o": 1},
        inference_config={"i": 1},
        synthetic_data_config={"s": 1},
        param_bounds={"lr": (1e-5, 0.01)},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert node._optimizer_plugins["d"].received["param_bounds"] == {"lr": [1e-5, 0.01]}
    assert "param_bounds" not in node._evaluator_plugins["d"].received
    assert "param_bounds" not in node._synthetic_plugins["d"].received


def test_r11_param_bounds_not_injected_when_key_present(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        optimization_plugin="opt",
        optimization_config={"param_bounds": {"pre": [0, 1]}},
        param_bounds={"lr": (1e-5, 0.01)},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    # existing key wins, domain bounds not injected over it
    assert node._optimizer_plugins["d"].received["param_bounds"] == {"pre": [0, 1]}


def test_r12_source_dicts_unchanged_after_configure(monkeypatch):
    _patch_loaders(monkeypatch)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        metric_type="m",
        optimization_config={"o": 1},
        inference_config={"i": 1},
        synthetic_data_config={"s": 1},
        param_bounds={"lr": (1e-5, 0.01)},
    )
    opt_before = copy.deepcopy(role.optimization_config)
    inf_before = copy.deepcopy(role.inference_config)
    syn_before = copy.deepcopy(role.synthetic_data_config)
    node = _FakeNode([role])
    cli.setup_plugins(node)
    assert role.optimization_config == opt_before
    assert role.inference_config == inf_before
    assert role.synthetic_data_config == syn_before


def test_r12_mutating_plugin_cannot_contaminate_others(monkeypatch):
    # Optimizer, inference and synthetic all fall back to optimization_config.
    _patch_loaders(monkeypatch, opt=_MutatingPlugin, inf=_InfPlugin, syn=_SynPlugin)
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        optimization_config={"base": 1},
        inference_config={},
        synthetic_data_config={},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)
    # Hostile optimizer mutated only its own copy
    assert node._optimizer_plugins["d"].received.get("POISON") is True
    # Role config and the other plugins' configs are untouched
    assert "POISON" not in role.optimization_config
    assert "POISON" not in node._evaluator_plugins["d"].received
    assert "POISON" not in node._synthetic_plugins["d"].received
    assert node._evaluator_plugins["d"].received.get("metric_type") != "MUTATED"


def test_r12_nested_mutation_cannot_contaminate_role_or_fallbacks(monkeypatch):
    _patch_loaders(
        monkeypatch,
        opt=_NestedMutatingPlugin,
        inf=_InfPlugin,
        syn=_SynPlugin,
    )
    role = DomainRole(
        domain_id="d",
        optimize=True,
        evaluate=True,
        optimization_plugin="opt",
        inference_plugin="inf",
        synthetic_data_plugin="syn",
        optimization_config={"nested": {"values": [1]}},
        inference_config={},
        synthetic_data_config={},
    )
    node = _FakeNode([role])
    cli.setup_plugins(node)

    assert node._optimizer_plugins["d"].received["nested"]["values"] == [1, "POISON"]
    assert role.optimization_config["nested"]["values"] == [1]
    assert node._evaluator_plugins["d"].received["nested"]["values"] == [1]
    assert node._synthetic_plugins["d"].received["nested"]["values"] == [1]


# ── Identity precedence test (pure selection, no key generation) ──────

def test_r07_identity_precedence():
    # 1. explicit CLI identity wins
    assert cli._select_identity_path("/cli/id.pem", "/json/id.pem") == "/cli/id.pem"
    # 2. JSON identity_file when no CLI identity
    assert cli._select_identity_path(None, "/json/id.pem") == "/json/id.pem"
    assert cli._select_identity_path("", "/json/id.pem") == "/json/id.pem"
    # 3. neither -> None (caller falls back to data_dir default)
    assert cli._select_identity_path(None, "") is None
    assert cli._select_identity_path("", "") is None
