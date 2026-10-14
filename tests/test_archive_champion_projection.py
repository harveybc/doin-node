"""Synthetic acceptance projection contracts; no services or live state."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from doin_core.archive.body import ArchiveRefusal, digest_bytes
from doin_core.archive.external import build_external_body
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.archive.champion import resolve_champion_payload
from doin_node.archive.external import ExternalPayloadArchive
from doin_node.dashboard.routes import _blockchain_metrics_payload
from doin_node.storage.chaindb import ChainDB
from tools.i6n_archive import records
from tools.i6n_search_space import GENOTYPE_SCHEMA


DOMAIN = "synthetic-island-h4"


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.writes = 0

    def put_bytes(self, content):
        self.writes += 1
        digest = digest_bytes(content)
        self.objects[digest] = content
        return digest

    def get_bytes(self, digest):
        return self.objects[digest]


@pytest.fixture
def candidate():
    effective = {"horizon": 4, "core": {"width": 8}}
    identities = {
        "genotype_canonical_sha256": digest_bytes(records.canonical({
            "schema": "i6n-search-v1", "config": effective,
        })),
        "effective_config_sha256": records.e_config_sha256(effective),
        "plan_sha256": "a" * 64,
        "contract_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
    }
    folds, receipts, origins = {}, {}, {}
    for index in range(4):
        fold = f"fold-{index}"
        support = {
            "counts": {"train": 10, "inner": 5},
            "naive_sums": {"train": 10.0, "inner": 5.0},
            "support_digest": {"train": str(index + 1) * 64, "inner": str(index + 5) * 64},
        }
        origins[fold] = support["support_digest"]
        folds[fold] = {
            "i6n_support": support,
            "i6n_origins_sha256": origins[fold],
            "i6n_evaluation_context": {"fold": fold},
        }
        receipts[fold] = {
            "status": "ok", **support,
            "error_sums": {"train": 8.0, "inner": 4.0},
            "metadata": {"candidate_receipt": {
                "genotype_canonical_sha256": identities["genotype_canonical_sha256"],
                "fold": fold, "horizon_hours": 4,
                "rows": [{
                    "split": split.upper(), "head_offset_hours": 4,
                    "head_role": "ENDPOINT", "comparison": True,
                    "support_sha256": support["support_digest"][split],
                } for split in ("train", "inner")],
            }},
        }
    return records.island_candidate_payload(DOMAIN, {
        **identities,
        "parameters": effective, "fitness": -0.2,
        "generation": 2, "candidate_in_gen": 1, "total_eval": 3,
        "training_seed": 42, "island_id": "synthetic", "is_champion": True,
        "train_mae": 0.8, "train_naive_mae": 1.0,
        "val_mae": 0.8, "val_naive_mae": 1.0,
        "metrics": {
            **identities, "relative_error_vs_naive": -0.2,
            "per_fold": dict.fromkeys(folds, 0.8), "receipts": receipts,
            "origins_binding": "VERIFIED", "origins_sha256": origins,
            "aggregation": "mean_four_folds",
        },
    }, {
        "i6n_island_mode": True, "i6n_horizon": 4,
        "i6n_plan": {"plan_sha256": identities["plan_sha256"]},
        "i6n_folds": folds, "seed": 42,
    })


def archived(candidate, *, mode="external_payload", domain=DOMAIN, schema=None):
    store = MemoryStore()
    archive = ExternalPayloadArchive({"synthetic": store}, write_store_id="synthetic")
    external = build_external_body(
        domain_id=domain, producer_peer_id="synthetic-peer",
        payload_schema=schema or records.ISLAND_CANDIDATE_SCHEMA, payload=candidate,
    )
    reference = archive.store_body(external, mode=mode)
    return archive, store, {"optimae_id": "synthetic-accepted", "candidate_archive_reference": reference}


def test_dashboard_reads_compact_acceptance_pair_and_flags_regression(tmp_path, candidate):
    archive, _, evidence = archived(candidate)
    db = ChainDB(tmp_path / "chain.sqlite")
    db.open()
    genesis = db.initialize()
    for index, score in enumerate((-0.2, -0.1), start=1):
        ident = f"accepted-{index}"
        work = Transaction(tx_type=TransactionType.OPTIMAE_ACCEPTED,
                           domain_id=DOMAIN, peer_id="origin",
                           payload={"optimae_id": ident, "performance": score,
                                    "increment": 0.01})
        reference = Transaction(tx_type=TransactionType.OPTIMAE_ACCEPTED,
                                domain_id=DOMAIN, peer_id="origin",
                                payload={**evidence, "optimae_id": ident})
        block = Block(header=BlockHeader(
            index=index, previous_hash=genesis.hash,
            merkle_root=compute_merkle_root([work.id, reference.id]),
            generator_id="origin", weighted_performance_sum=0.01,
            threshold=0.001), transactions=[work, reference])
        db.append_block(block)
        genesis = block
    try:
        node = SimpleNamespace(
            config=SimpleNamespace(domains=[SimpleNamespace(domain_id=DOMAIN, higher_is_better=False)]),
            chaindb=db, _candidate_external_archive=archive,
        )
        result = _blockchain_metrics_payload(node)
        assert [row["performance"] for row in result["metrics"]] == [-0.2, -0.1]
        assert [row["is_improvement"] for row in result["metrics"]] == [True, False]
        assert [row["val_mae"] for row in result["metrics"]] == [0.8, 0.8]
        assert result["champion_improvements"] == 1
        assert result["invalid_acceptance_count"] == 1
    finally:
        db.close()


def test_legacy_projection_is_detached_and_public():
    payload = {
        "verified_performance": 0, "parameters": {"core": {"width": 8}},
        "champion_metrics": {"val_mae": 0.8, "_model_b64": "private", "_best_model_b64": "private", "test_mae": None},
    }
    original = deepcopy(payload)
    result = resolve_champion_payload(payload, DOMAIN, None)
    assert result["verified_performance"] == 0.0
    assert result["champion_metrics"] == {"val_mae": 0.8}
    result["parameters"]["core"]["width"] = 99
    assert payload == original


def test_legacy_metric_columns():
    result = resolve_champion_payload({
        "parameters": {}, "verified_performance": -0.1,
        "train_mae": 0.8, "test_mae": None,
    }, DOMAIN, None)
    assert result["champion_metrics"] == {"train_mae": 0.8}


@pytest.mark.parametrize("mode", ["external_payload", "full_payload"])
@pytest.mark.parametrize("retained_score", [False, True])
def test_reference_projection_uses_consensus_score(candidate, mode, retained_score):
    archive, store, payload = archived(candidate, mode=mode)
    if retained_score:
        payload.update(verified_performance=-0.125, effective_increment=0.01,
                       reward_fraction=0.5, quorum_agree_fraction=0.8, incentive_reason="accepted")
    original, objects, writes = deepcopy(payload), dict(store.objects), store.writes
    result = resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.125)
    assert result["verified_performance"] == -0.125 != candidate["fitness"]
    assert result["parameters"] == {
        **candidate["effective_config"], "_i6n_archive_reference": payload["candidate_archive_reference"],
        "_i6n_genotype_sha256": candidate["genotype_canonical_sha256"],
        "_i6n_origin_island": candidate["island_id"],
        "_i6n_envelope": {
            "schema": "i6n.island.champion.v1", "plan_sha256": candidate["plan_sha256"],
            "context_sha256": candidate["contract_sha256"], "horizon": 4,
            "training_seed": 42, "genotype_schema": GENOTYPE_SCHEMA,
        },
    }
    metrics = result["champion_metrics"]
    assert metrics["val_mae"] == metrics["train_mae"] == 0.8
    assert metrics["val_naive_mae"] == 1.0
    assert metrics["relative_error_vs_naive"] == pytest.approx(-0.2)
    assert metrics["genotype_canonical_sha256"] == candidate["genotype_canonical_sha256"]
    assert metrics["per_fold"] == dict.fromkeys((f"fold-{i}" for i in range(4)), 0.8)
    assert resolve_champion_payload(result, DOMAIN, archive) == result
    result["parameters"]["core"]["width"] = 99
    result["parameters"]["_i6n_archive_reference"]["domain_id"] = "changed"
    assert payload == original and store.objects == objects and store.writes == writes


def test_retained_score_needs_no_companion(candidate):
    archive, _, payload = archived(candidate)
    payload["verified_performance"] = -0.125
    result = resolve_champion_payload(payload, DOMAIN, archive)
    assert result["verified_performance"] == -0.125


@pytest.mark.parametrize("key,value", [
    ("_i6n_genotype_sha256", "f" * 64),
    ("_i6n_origin_island", "other"), ("_i6n_envelope", {}),
])
def test_migration_metadata_must_match_verified_body(candidate, key, value):
    archive, _, payload = archived(candidate)
    payload["parameters"] = {**candidate["effective_config"], key: value}
    with pytest.raises(ArchiveRefusal, match="IDENTITY_MISMATCH"):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


def test_outer_metrics_cannot_override_archived_evidence(candidate):
    archive, _, payload = archived(candidate)
    payload["champion_metrics"] = {"val_mae": 999, "_model_b64": "private"}
    result = resolve_champion_payload(payload, DOMAIN, archive, verified_performance=0.0)
    assert result["champion_metrics"]["val_mae"] == 0.8
    assert "_model_b64" not in result["champion_metrics"]


def test_unknown_store(candidate):
    _, _, payload = archived(candidate)
    with pytest.raises(ArchiveRefusal, match="LOCATOR_STORE_UNKNOWN"):
        resolve_champion_payload(payload, DOMAIN, ExternalPayloadArchive({}), verified_performance=0.0)


def test_rehashed_receipt_mutation(candidate):
    candidate["per_fold"][0]["receipt"]["error_sums"]["train"] = 99.0
    archive, _, payload = archived(candidate)
    with pytest.raises(ArchiveRefusal, match="BODY_INVALID"):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=0.0)


@pytest.mark.parametrize("score", [True, False, float("nan"), float("inf"), -float("inf"), "0.5", None, 10 ** 400])
@pytest.mark.parametrize("source", ["payload", "companion"])
def test_invalid_scores_never_recover(candidate, score, source):
    archive, _, payload = archived(candidate)
    kwargs = {}
    if source == "payload":
        payload["verified_performance"] = score
    else:
        kwargs["verified_performance"] = score
    with pytest.raises(ArchiveRefusal):
        resolve_champion_payload(payload, DOMAIN, archive, **kwargs)


def test_no_fitness_or_increment_fallback(candidate):
    archive, _, payload = archived(candidate)
    payload.update(performance=-0.2, fitness=-0.2, increment=0.1, effective_increment=0.1)
    with pytest.raises(ArchiveRefusal, match="PERFORMANCE_REQUIRED"):
        resolve_champion_payload(payload, DOMAIN, archive)


def test_conflicting_consensus_scores(candidate):
    archive, _, payload = archived(candidate)
    payload["verified_performance"] = -0.1
    with pytest.raises(ArchiveRefusal, match="PERFORMANCE_MISMATCH"):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.2)


@pytest.mark.parametrize("key", ["body_digest", "manifest_digest"])
@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_missing_or_corrupt_bytes_never_fall_back(candidate, key, mutation):
    archive, store, payload = archived(candidate)
    payload.update(parameters=candidate["effective_config"], champion_metrics={"val_mae": 123})
    digest = payload["candidate_archive_reference"][key]
    if mutation == "missing":
        del store.objects[digest]
    else:
        store.objects[digest] += b" "
    with pytest.raises(ArchiveRefusal):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


@pytest.mark.parametrize("key,value", [
    ("domain_id", "other"), ("body_size", 1), ("payload_schema", "unknown.v1"),
    ("reference_schema", "unknown.v1"), ("body_digest", "f" * 64),
    ("manifest_digest", "f" * 64), ("locator", "doin-cas://unknown/sha256/" + "f" * 64),
])
def test_reference_mutations(candidate, key, value):
    archive, _, payload = archived(candidate)
    payload["candidate_archive_reference"][key] = value
    with pytest.raises(ArchiveRefusal):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


@pytest.mark.parametrize("key,value", [
    ("schema", "unknown.v1"), ("experiment", "other"),
    ("genotype_canonical_sha256", "f" * 64), ("effective_config_sha256", "f" * 64),
    ("archive_event_sha256", "f" * 64), ("search_candidate_sha256", "f" * 64),
    ("genotype", {}), ("effective_config", {"horizon": 4, "core": {"width": 99}}),
    ("fitness", True), ("fitness", -0.9), ("aggregate_metrics", {"val_mae": 99}),
])
def test_rehashed_invalid_bodies_are_not_valid_evidence(candidate, key, value):
    candidate[key] = value
    archive, _, payload = archived(candidate)
    with pytest.raises(ArchiveRefusal):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


@pytest.mark.parametrize("mismatch", ["domain", "parameters", "identity", "embedded", "null-reference"])
def test_payload_binding(candidate, mismatch):
    archive, _, payload = archived(candidate)
    if mismatch == "domain":
        payload["domain_id"] = "other"
    elif mismatch == "parameters":
        payload["parameters"] = {"horizon": 4, "core": {"width": 99}}
    elif mismatch == "identity":
        payload["genotype_canonical_sha256"] = "f" * 64
    elif mismatch == "embedded":
        payload["parameters"] = {**candidate["effective_config"], "_i6n_archive_reference": {}}
    else:
        payload["candidate_archive_reference"] = None
        payload["parameters"] = candidate["effective_config"]
    with pytest.raises(ArchiveRefusal):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


def test_valid_body_from_other_domain(candidate):
    candidate["experiment"] = "other"
    event = {key: candidate[key] for key in (
        "plan_sha256", "horizon_hours", "seed", "island_id", "generation",
        "population_index", "total_evaluations", "search_candidate_sha256",
    )}
    candidate["archive_event_sha256"] = digest_bytes(records.canonical(dict(event, domain_id="other")))
    records.validate_island_candidate_payload(candidate)
    archive, _, payload = archived(candidate)
    with pytest.raises(ArchiveRefusal, match="DOMAIN_MISMATCH"):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


def test_unknown_body_schema(candidate):
    archive, _, payload = archived(candidate, schema="unsupported.v1")
    with pytest.raises(ArchiveRefusal, match="PAYLOAD_SCHEMA"):
        resolve_champion_payload(payload, DOMAIN, archive, verified_performance=-0.1)


def test_archive_is_required(candidate):
    _, _, payload = archived(candidate)
    with pytest.raises(ArchiveRefusal, match="ARCHIVE_REQUIRED"):
        resolve_champion_payload(payload, DOMAIN, None, verified_performance=-0.1)
