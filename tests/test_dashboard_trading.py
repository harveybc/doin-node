from doin_node.dashboard.routes import (
    _PACKAGE_VERSIONS,
    _compact_metric_evidence,
    _dashboard_transaction_payload,
)


def test_trading_dashboard_metrics_are_compact_and_decision_grade() -> None:
    compact = _compact_metric_evidence({
        "train_validation_l1_score": 0.03,
        "validation_selection_score": 0.10,
        "max_drawdown_fraction": 0.04,
        "history": [{"epoch": 1}] * 100,
        "splits": {"train": {"trace": list(range(100))}},
        "_model_b64": "large-model",
    })

    assert compact == {
        "train_validation_l1_score": 0.03,
        "validation_selection_score": 0.10,
        "max_drawdown_fraction": 0.04,
    }


def test_chain_dashboard_redacts_model_but_preserves_chain_payload_source() -> None:
    source = {
        "parameters": {"learning_rate": 0.001, "_model_b64": "abcd"},
        "champion_metrics": {
            "total_return": 0.2,
            "history": [{"epoch": 1}],
        },
        "metrics": {
            "train_validation_l1_score": 0.03,
            "splits": {"validation": {"trace": list(range(100))}},
        },
    }

    displayed = _dashboard_transaction_payload(source)

    assert displayed["parameters"] == {"learning_rate": 0.001}
    assert displayed["model_artifact_embedded"] is True
    assert displayed["model_artifact_base64_chars"] == 4
    assert displayed["champion_metrics"] == {"total_return": 0.2}
    assert displayed["metrics"] == {"train_validation_l1_score": 0.03}
    assert source["parameters"]["_model_b64"] == "abcd"
    assert "history" in source["champion_metrics"]
    assert "splits" in source["metrics"]


def test_dashboard_versions_cover_only_active_trading_components() -> None:
    assert set(_PACKAGE_VERSIONS) == {
        "agent-multi",
        "doin-core",
        "doin-node",
        "doin-plugins",
    }
    assert "predictor" not in _PACKAGE_VERSIONS
