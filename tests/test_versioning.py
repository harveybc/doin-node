from doin_node import versioning


def test_component_versions_cover_exact_campaign_components(
    monkeypatch,
    tmp_path,
) -> None:
    full_sha = "a" * 40
    monkeypatch.setattr(versioning, "_runtime_source", lambda _label: tmp_path)
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


def test_runtime_source_precedes_stale_editable_install(monkeypatch, tmp_path) -> None:
    runtime = tmp_path / "runtime"
    stale = tmp_path / "stale"
    runtime.mkdir()
    stale.mkdir()
    monkeypatch.setattr(versioning, "_runtime_source", lambda label: runtime if label == "predictor" else None)
    monkeypatch.setattr(versioning, "_editable_source", lambda _name: stale)
    monkeypatch.setattr(
        versioning,
        "_git_short_hash",
        lambda path: "b" * 40 if path == runtime else "c" * 40,
    )

    versions = versioning.compute_component_versions()

    assert versions["predictor"] == "b" * 40
