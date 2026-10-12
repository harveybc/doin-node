"""Focused contract for hash-only candidate publication from the node callback."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from doin_core.models import TransactionType
from doin_core.archive.external import build_external_body, reference_payload
from doin_node.archive.external import build_reference_transaction, spool_submitter
from doin_node.cli import load_config
from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig


PLAN = "9" * 64


def _role() -> DomainRole:
    return DomainRole(
        domain_id="i6n-island-h4",
        optimize=True,
        optimization_config={
            "i6n_island_mode": True,
            "i6n_horizon": 4,
            "i6n_plan": {"plan_sha256": PLAN},
            "i6n_folds": {
                fold: {
                    "i6n_support": {"support_digest": {"train": str(index + 1) * 64}},
                    "i6n_evaluation_context": {"fold": fold},
                }
                for index, fold in enumerate(("2023Q1", "2023Q2", "2023Q3", "2023Q4"))
            },
            "seed": 6102026,
        },
    )


def _candidate() -> dict:
    return {
        "generation": 3,
        "candidate_in_gen": 7,
        "total_eval": 68,
        "fitness": -0.2,
        "is_champion": True,
        "parameters": {
            "horizon": 4,
            "multiplier": 6,
            "delta_x": 1,
            "input_transform": "identity",
            "output_mode": "endpoint",
            "delta_y": None,
            "core": {"entrance": "conv", "width": 8, "transformer_blocks": 0,
                     "reducer_filters": [4], "reducer_strides": [1],
                     "adapter": "none", "adapter_width": None},
            "history_hours": 24,
            "samples": 25,
            "final_steps": 25,
            "input_offsets": list(range(-24, 1)),
            "output_offsets": [4],
        },
        "train_mae": 0.8,
        "train_naive_mae": 1.0,
        "val_mae": 0.4,
        "val_naive_mae": 0.5,
        "metrics": {
            "relative_error_vs_naive": -0.2,
            "per_fold_ratio": {fold: 0.8 for fold in ("2023Q1", "2023Q2", "2023Q3", "2023Q4")},
            "aggregation": "mean_four_folds",
        },
        "training_seed": 6102026,
        "island_id": "gamma-5090",
    }


class _Publisher:
    def __init__(self):
        self.recorded = []
        self.pumps = 0
        external = build_external_body(
            domain_id="i6n-island-h4",
            producer_peer_id="peer-a",
            payload_schema="fixture.v1",
            payload={"value": 1},
        )
        self.reference = reference_payload(
            external, mode="external_payload", store_id="cas"
        )
        self.outbox = self

    def record_island_candidate(self, candidate_info, optimization_config):
        self.recorded.append((candidate_info, optimization_config))
        return {"item_id": "a" * 64, "body_digest": "a" * 64, "created": True}

    def pump(self, archive, submit):
        self.pumps += 1
        return {"stored": 1, "submitted": 1, "errors": []}

    def item(self, _item_id):
        return {"reference": self.reference}


def test_external_mode_routes_callback_only_to_archive(tmp_path, monkeypatch):
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0,
        data_dir=str(tmp_path / "node"),
        domains=[role],
        candidate_archive_mode="external_payload",
        candidate_archive_root=str(tmp_path / "archive"),
        candidate_archive_store_id="cas",
        candidate_archive_store_url="http://lake.invalid",
        candidate_archive_token_env="TEST_LAKE_TOKEN",
        archive_reference_spool=str(tmp_path / "spool"),
    ))
    publisher = _Publisher()
    node._candidate_archive_publishers[role.domain_id] = publisher
    node._candidate_external_archive = object()
    recorded = []
    monkeypatch.setattr(node.consensus, "record_transaction", recorded.append)

    result = node._publish_candidate_evaluation(role.domain_id, _candidate(), loop=None)

    assert result["body_digest"] == "a" * 64
    assert publisher.pumps == 1 and len(publisher.recorded) == 1
    assert recorded == []  # no full candidate transaction can reach consensus


def test_external_mode_champion_transaction_contains_only_archive_reference(tmp_path):
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0,
        data_dir=str(tmp_path / "node"),
        domains=[role],
        candidate_archive_mode="external_payload",
        candidate_archive_root=str(tmp_path / "archive"),
        candidate_archive_store_id="cas",
        candidate_archive_store_url="http://lake.invalid",
        candidate_archive_token_env="TEST_LAKE_TOKEN",
        archive_reference_spool=str(tmp_path / "spool"),
    ))
    external = build_external_body(
        domain_id=role.domain_id,
        producer_peer_id="peer-a",
        payload_schema="fixture.v1",
        payload={"metrics": [1, 2, 3], "parameters": {"filters": 8}},
    )
    reference = reference_payload(
        external, mode="external_payload", store_id="cas"
    )

    payload = node._archive_only_optimae_payload(
        role.domain_id,
        "opt-1",
        {
            "filters": 8,
            "metrics": {"mae": 0.1},
            "_i6n_archive_reference": reference,
        },
    )

    assert set(payload) == {"optimae_id", "candidate_archive_reference"}
    assert payload["candidate_archive_reference"] == reference
    assert "parameters" not in payload and "metrics" not in payload


def test_restart_restores_durable_candidate_reference_by_genotype(tmp_path):
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0,
        data_dir=str(tmp_path / "node"),
        domains=[role],
        candidate_archive_mode="external_payload",
        candidate_archive_root=str(tmp_path / "archive"),
        candidate_archive_store_id="cas",
        candidate_archive_store_url="http://lake.invalid",
        candidate_archive_token_env="TEST_LAKE_TOKEN",
        archive_reference_spool=str(tmp_path / "spool"),
    ))
    genotype = "a" * 64
    external = build_external_body(
        domain_id=role.domain_id,
        producer_peer_id="peer-a",
        payload_schema="i6n.archive.island_candidate.v1",
        payload={
            "schema": "i6n.archive.island_candidate.v1",
            "experiment": role.domain_id,
            "genotype_canonical_sha256": genotype,
        },
    )
    reference = reference_payload(
        external, mode="external_payload", store_id="cas"
    )

    class Outbox:
        @staticmethod
        def items():
            return [{
                "item_id": external.body_digest,
                "body_digest": external.body_digest,
                "manifest_digest": external.manifest_digest,
                "reference": reference,
                "states": {"OBJECT_DURABLE": {"ok": True}},
            }]

        @staticmethod
        def bytes_of(_item_id):
            return external.body, external.manifest

    publisher = type("Publisher", (), {"outbox": Outbox()})()
    node._candidate_archive_publishers[role.domain_id] = publisher

    restored = node._restore_candidate_archive_references(role.domain_id)

    assert restored == 1
    assert node._candidate_archive_references[role.domain_id][genotype] == reference


def test_legacy_full_mode_still_builds_the_original_candidate_transaction(tmp_path):
    role = _role()
    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / "node"), domains=[role],
        candidate_archive_mode="full_payload",
    ))
    candidate = _candidate()

    tx = node._candidate_evaluation_transaction(role.domain_id, candidate)

    assert tx.tx_type == TransactionType.CANDIDATE_EVALUATED
    assert tx.payload["parameters"] == candidate["parameters"]
    assert tx.payload["metrics"] == candidate["metrics"]


def test_external_archive_config_is_loaded_and_fail_closed(tmp_path):
    config_path = tmp_path / "node.json"
    config_path.write_text(json.dumps({
        "candidate_archive_mode": "external_payload",
        "candidate_archive_root": str(tmp_path / "archive"),
        "candidate_archive_store_id": "cas",
        "candidate_archive_store_url": "http://lake.invalid",
        "candidate_archive_token_env": "TEST_LAKE_TOKEN",
        "candidate_archive_pump_interval": 17.0,
        "archive_reference_spool": str(tmp_path / "spool"),
    }))
    loaded = load_config(str(config_path), {})
    assert loaded.candidate_archive_mode == "external_payload"
    assert loaded.candidate_archive_pump_interval == 17.0

    with pytest.raises(ValueError, match="candidate_archive_store_url"):
        UnifiedNodeConfig(
            candidate_archive_mode="external_payload",
            candidate_archive_root=str(tmp_path),
            candidate_archive_store_id="cas",
            candidate_archive_token_env="TEST_LAKE_TOKEN",
            archive_reference_spool=str(tmp_path / "spool"),
        ).validate_candidate_archive()


def test_reference_spool_writer_is_byte_idempotent(tmp_path):
    external = build_external_body(
        domain_id="i6n-island-h4",
        producer_peer_id="peer-a",
        payload_schema="i6n.archive.island_candidate.v1",
        payload={"schema": "fixture", "value": 1},
    )
    reference = reference_payload(external, mode="external_payload", store_id="cas")
    tx = build_reference_transaction(
        reference, peer_id="peer-a", timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc)
    )
    submit = spool_submitter(tmp_path / "spool")

    submit(tx)
    first = (tmp_path / "spool" / f"{tx.id}.json").read_bytes()
    submit(tx)

    assert (tmp_path / "spool" / f"{tx.id}.json").read_bytes() == first
    assert len(list((tmp_path / "spool").glob("*.json"))) == 1
