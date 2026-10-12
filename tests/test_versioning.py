from doin_node import versioning


def test_component_versions_cover_exact_campaign_components(
    monkeypatch,
    tmp_path,
) -> None:
    full_sha = "a" * 40
    monkeypatch.setattr(versioning, "_editable_source", lambda _name: tmp_path)
    monkeypatch.setattr(versioning, "_git_short_hash", lambda _path: full_sha)
    versions = versioning.compute_component_versions()

    assert set(versions) == {
        "agent-multi",
        "doin-core",
        "doin-node",
        "doin-plugins",
        "gym-fx",
        "predictor",
        "trading-contracts",
    }
    assert set(versions.values()) == {full_sha}
