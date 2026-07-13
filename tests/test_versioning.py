from doin_node import versioning


def test_component_versions_cover_only_active_trading_components(monkeypatch) -> None:
    monkeypatch.setattr(versioning, "_editable_source", lambda _name: None)
    monkeypatch.setattr(versioning, "_git_short_hash", lambda _path: "abcdef0")
    versions = versioning.compute_component_versions()

    assert set(versions) == {
        "agent-multi",
        "doin-core",
        "doin-node",
        "doin-plugins",
        "gym-fx",
        "trading-contracts",
    }
    assert set(versions.values()) == {"abcdef0"}
