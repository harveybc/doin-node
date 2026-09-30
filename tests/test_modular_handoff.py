import json

import pytest

from doin_node import cli
from doin_node.cli import load_config
from doin_node.modular_dry_run import main as dry_run_main


def experiment():
    return {
        "schema_version": "modular.experiment.v1",
        "architecture": {
            "branches": [
                {"name": "close", "features": ["close"], "encoder": "temporal_conv"},
                {"name": "context", "features": ["volume"], "encoder": "recurrent"},
            ],
            "fusion": "sequence_concat",
            "temporal_core": "recurrent",
        },
        "regime": "R0",
        "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False, "unit": "price"},
        "data": {"dataset_id": "daily-v1", "train_end": "2024-01-01", "validation_start": "2024-01-02"},
    }


def test_modular_config_handoff_to_both_plugin_subtrees(tmp_path):
    path = tmp_path / "node.json"
    path.write_text(json.dumps({"domains": [{
        "domain_id": "forecast", "optimize": True, "evaluate": True,
        "optimization_plugin": "predictor", "inference_plugin": "predictor",
        "higher_is_better": False,
        "optimization_config": {"modular_experiment": experiment()},
        "inference_config": {"batch_size": 8},
    }]}))
    role = load_config(str(path), {}).domains[0]
    assert role.optimization_config["modular_experiment"] == experiment()
    assert role.inference_config["modular_experiment"] == experiment()
    assert role.optimization_config["optimization_metric"] == "MAE"
    assert role.inference_config["optimization_metric"] == "MAE"

    configured = []

    class Plugin:
        def configure(self, config):
            configured.append(config)

    class Node:
        _domain_roles = {"forecast": role}

        def register_optimizer_plugin(self, domain_id, plugin):
            assert domain_id == "forecast"

        def register_evaluator_plugin(self, domain_id, plugin):
            assert domain_id == "forecast"

    from pytest import MonkeyPatch
    with MonkeyPatch.context() as patch:
        patch.setattr(cli, "load_optimization_plugin", lambda _: Plugin)
        patch.setattr(cli, "load_inference_plugin", lambda _: Plugin)
        cli.setup_plugins(Node())
    assert len(configured) == 2
    assert all(config["modular_experiment"] == experiment() for config in configured)
    assert all(config["optimization_metric"] == "MAE" for config in configured)


@pytest.mark.parametrize("field,value", [
    ("higher_is_better", True),
    ("optimization_config", {"modular_experiment": experiment(), "optimization_metric": "R2"}),
])
def test_objective_conflict_refused(tmp_path, field, value):
    domain = {"domain_id": "forecast", "higher_is_better": False,
              "optimization_config": {"modular_experiment": experiment()}}
    domain[field] = value
    path = tmp_path / "node.json"
    path.write_text(json.dumps({"domains": [domain]}))
    with pytest.raises(ValueError, match="forecast"):
        load_config(str(path), {})


def test_offline_dry_run_reports_no_training(monkeypatch, capsys):
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[1] / "examples/modular_handoff_dry_run.json"
    monkeypatch.setattr("sys.argv", ["modular_dry_run", str(fixture)])
    dry_run_main()
    result = json.loads(capsys.readouterr().out)
    assert result[0]["status"] == "config_validated_no_training"
    assert result[0]["optimizer_evaluator_match"] is True


def test_evaluator_only_experiment_refused(tmp_path):
    path = tmp_path / "node.json"
    path.write_text(json.dumps({"domains": [{
        "domain_id": "forecast", "inference_config": {"modular_experiment": experiment()}
    }]}))
    with pytest.raises(ValueError, match="forecast"):
        load_config(str(path), {})


def test_null_experiment_refused(tmp_path):
    path = tmp_path / "node.json"
    path.write_text(json.dumps({"domains": [{
        "domain_id": "forecast", "optimization_config": {"modular_experiment": None}
    }]}))
    with pytest.raises(ValueError, match="forecast"):
        load_config(str(path), {})
