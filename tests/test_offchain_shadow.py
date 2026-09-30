"""Stage-1 shadow archive: disposable file bytes, not a lake upload.

One fixture block is written, read back, and checked. A flipped byte is
refused. A second put of the same bytes does not create a second record.
A missing or unreadable file is a refusal, not a successful validation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from doin_core.archive.body import (
    ArchiveRefusal,
    build_archive,
    canonical_bytes,
    digest_bytes,
    verify_envelope,
)
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.archive.file_adapter import (
    LAKE_WRITE_FACT,
    MANIFEST_IDENTITY_RULE,
    FileArchiveAdapter,
)
from doin_node.archive.warehouse import (
    DisposableWarehouse,
    project_from_reference,
    project_metrics,
)
from doin_node.stats.chain_metrics import collect_chain_metrics
from doin_node.stats.olap_db import OLAPDatabase

TS = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _fixture():
    tx = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="fixture-domain",
        peer_id="peer-fixture",
        payload={
            "optimae_id": "opt-win",
            "parameters": {"w": 1},
            "verified_performance": 0.9,
            "reported_performance": 0.9,
        },
        timestamp=TS,
    )
    header = BlockHeader(
        index=3,
        previous_hash="ab" * 32,
        timestamp=TS,
        merkle_root=compute_merkle_root([tx.id]),
        generator_id="node-fixture",
        weighted_performance_sum=1.5,
        threshold=0.5,
    )
    block = Block(header=header, transactions=[tx])
    lose = {
        "record_id": "rec-lose-1",
        "candidate_id": "cand-lose",
        "attempt": 1,
        "won": False,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.2,
        "parameters": {"w": 0.2},
        "metrics": {"split": "val"},
    }
    win = {
        "record_id": "rec-win-1",
        "candidate_id": "cand-win",
        "attempt": 1,
        "won": True,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.9,
        "parameters": {"w": 1},
        "metrics": {"split": "val"},
    }
    retry = {
        "record_id": "rec-lose-2",
        "candidate_id": "cand-lose",
        "attempt": 2,
        "won": False,
        "domain_id": "fixture-domain",
        "peer_id": "peer-fixture",
        "performance": 0.25,
        "parameters": {"w": 0.25},
        "metrics": {"split": "val"},
    }
    envelope = build_archive(block=block, candidates=[lose, win], retries=[retry])
    return block, envelope


def test_lake_registration_is_not_the_byte_store() -> None:
    assert FileArchiveAdapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    assert "does not copy resource bytes" in LAKE_WRITE_FACT
    assert "write_metrics" in LAKE_WRITE_FACT


def test_disposable_warehouse_matches_olap_call_shape() -> None:
    assert DisposableWarehouse.LABEL == "DISPOSABLE"
    for name in ("create_experiment", "record_round"):
        real = inspect.signature(getattr(OLAPDatabase, name))
        double = inspect.signature(getattr(DisposableWarehouse, name))
        assert list(real.parameters) == list(double.parameters)
    with pytest.raises(ArchiveRefusal, match="POSTGRES_REFUSED"):
        DisposableWarehouse("postgresql://example/doin")


def test_round_trip_tamper_retry_and_missing(tmp_path) -> None:
    _block, envelope = _fixture()
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(envelope)
    assert receipt.verified is True
    assert receipt.already_stored is False
    assert receipt.body_digest == envelope.body_digest
    assert receipt.manifest_digest == envelope.manifest_digest
    body, manifest = adapter.read_verified(receipt)
    assert body == envelope.body
    assert manifest == envelope.manifest
    assert hashlib.sha256(body).hexdigest() == receipt.body_digest

    again = adapter.put(envelope)
    assert again.already_stored is True
    assert again.body_digest == receipt.body_digest
    assert adapter.record_count() == 1
    assert len(list((tmp_path / "objects").glob(receipt.body_digest))) == 1

    body_path = tmp_path / "objects" / receipt.body_digest
    original = body_path.read_bytes()
    body_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        adapter.read_verified(receipt)
    body_path.write_bytes(original)

    body_path.unlink()
    with pytest.raises(ArchiveRefusal, match="MISSING"):
        adapter.read_body(receipt.body_digest)


def test_unreadable_file_refuses(tmp_path) -> None:
    _block, envelope = _fixture()
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(envelope)
    body_path = tmp_path / "objects" / receipt.body_digest
    os.chmod(body_path, 0)
    try:
        with pytest.raises(ArchiveRefusal, match="UNREADABLE"):
            adapter.read_body(receipt.body_digest)
    finally:
        os.chmod(body_path, 0o644)


def test_projection_keeps_non_winners_retries_and_unanchored(tmp_path) -> None:
    block, envelope = _fixture()
    accepted_only = collect_chain_metrics([block])
    assert len(accepted_only) == 1
    assert accepted_only[0]["optimae_id"] == "opt-win"

    warehouse = DisposableWarehouse(":memory:")
    assert warehouse.describe()["label"] == "DISPOSABLE"
    assert warehouse.describe()["postgres"] is False
    experiment_id = warehouse.create_experiment(
        domain_id="fixture-domain",
        node_id="node-fixture",
        hostname="",
    )
    inserted = project_metrics(warehouse, envelope, experiment_id=experiment_id)
    assert inserted == 3
    rows = warehouse.get_rounds(experiment_id)
    assert len(rows) == 3
    metrics = [json.loads(row["metrics"]) for row in rows]
    assert {item["candidate_id"] for item in metrics} == {"cand-lose", "cand-win"}
    assert any(item["won"] is False for item in metrics)
    assert any(item["kind"] == "retry" and item["attempt"] == 2 for item in metrics)
    assert {item["anchor"] for item in metrics} == {"ANCHORED"}
    assert all(item["chain_verified"] is False for item in metrics)
    assert project_metrics(warehouse, envelope, experiment_id=experiment_id) == 0
    assert len(warehouse.get_rounds(experiment_id)) == 3

    unanchored = build_archive(
        block=None,
        candidates=[
            {
                "record_id": "rec-open-1",
                "candidate_id": "cand-open",
                "attempt": 1,
                "won": False,
                "domain_id": "fixture-domain",
                "peer_id": "peer-fixture",
                "performance": 0.1,
                "parameters": {},
                "metrics": {},
            }
        ],
        retries=[],
    )
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(unanchored)
    assert receipt.anchor == "UNANCHORED"
    assert receipt.header_hash is None
    reloaded, manifest = adapter.read_verified(receipt)
    assert json.loads(reloaded)["schema"] == "doin.unanchored_records.v1"
    assert json.loads(manifest)["header_hash"] is None
    other = warehouse.create_experiment(
        domain_id="fixture-domain",
        node_id="node-fixture",
        hostname="",
        experiment_id="exp-unanchored",
    )
    assert project_metrics(warehouse, unanchored, experiment_id=other) == 1
    projected = json.loads(warehouse.get_rounds(other)[0]["metrics"])
    assert projected["anchor"] == "UNANCHORED"
    assert projected["chain_verified"] is False
    assert projected["header_hash"] is None


def _mae_row(*, performance: float = 0.1, record_id: str = "r1") -> dict:
    return {
        "record_id": record_id,
        "candidate_id": "c1",
        "attempt": 1,
        "won": False,
        "domain_id": "test",
        "peer_id": "test-peer",
        "performance": performance,
        "parameters": {"depth": 2},
        "metrics": {"MAE": 0.1},
    }


def test_forged_section_is_rejected_before_any_row() -> None:
    """O03/AT02, O04. Altered section bytes with the original digests do not project."""
    row = _mae_row()
    envelope = build_archive(block=None, candidates=[row])
    original = next(section for section in envelope.sections if section.name == "candidates")
    forged_content = canonical_bytes([{**row, "kind": "candidate", "performance": 999.0}])
    forged = replace(
        envelope,
        sections=(replace(original, content=forged_content),)
        + tuple(section for section in envelope.sections if section.name != "candidates"),
    )
    assert forged.manifest_digest == envelope.manifest_digest
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e1")

    def refuse_insert(**kwargs):
        raise AssertionError("projected before verification")

    warehouse.record_round = refuse_insert
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        project_metrics(warehouse, forged, experiment_id="e1")
    assert warehouse.get_rounds("e1") == []


def test_projection_keeps_declared_metrics_and_marks_gaps() -> None:
    """O04/AT08 partial. MAE, parameters and declared qualifiers survive; gaps are explicit."""
    row = _mae_row()
    row["metrics"] = {
        "MAE": {"value": 0.1, "unit": "z", "horizon": 96},
        "split": "val",
    }
    row["provenance"] = {"source": "declared-fixture"}
    envelope = build_archive(block=None, candidates=[row])
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e1")
    assert project_metrics(warehouse, envelope, experiment_id="e1") == 1
    stored = warehouse.get_rounds("e1")[0]
    assert stored["performance"] == 0.1
    assert json.loads(stored["parameters"]) == {"depth": 2}
    assert stored["metric_schema"] == "doin.archive_projection.v1"
    payload = json.loads(stored["metrics"])
    assert payload["chain_verified"] is False
    by_name = {item["name"]: item for item in payload["metrics"]}
    assert by_name["MAE"]["value"] == 0.1
    assert by_name["MAE"]["unit"] == "z"
    assert by_name["MAE"]["horizon"] == 96
    assert by_name["MAE"]["scale"] == "NOT_COMPARABLE"
    assert by_name["MAE"]["population"] == "NOT_COMPARABLE"
    assert by_name["MAE"]["reduction"] == "NOT_COMPARABLE"
    assert by_name["split"]["value"] == "val"
    assert by_name["split"]["unit"] == "NOT_COMPARABLE"
    assert payload["provenance"] == {"source": "declared-fixture"}
    assert "z^2" not in stored["metrics"]
    plain = build_archive(block=None, candidates=[_mae_row(record_id="r-plain")])
    other = warehouse.create_experiment(
        domain_id="test", node_id="n", hostname="", experiment_id="e-plain"
    )
    project_metrics(warehouse, plain, experiment_id=other)
    plain_metrics = json.loads(warehouse.get_rounds(other)[0]["metrics"])
    mae = next(item for item in plain_metrics["metrics"] if item["name"] == "MAE")
    assert mae["value"] == 0.1
    assert mae["scale"] == "NOT_COMPARABLE"
    assert mae["unit"] == "NOT_COMPARABLE"


def test_same_content_is_idempotent_and_any_discrepancy_conflicts() -> None:
    """O01/O05/AT03. Equality uses the columns a comparison would read."""
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e1")
    kwargs = dict(
        experiment_id="e1",
        domain_id="test",
        round_number=1,
        performance=0.1,
        parameters={"depth": 2},
        round_id="r1",
        detail_metrics={"attempt": 1, "MAE": 0.1},
        metric_schema="schema-a",
    )
    assert warehouse.record_round(**kwargs) == "r1"
    assert warehouse.record_round(**kwargs) == "r1"
    assert len(warehouse.get_rounds("e1")) == 1
    for changed in (
        {"parameters": {"depth": 3}},
        {"metric_schema": "schema-b"},
        {"performance": 0.2},
        {"round_number": 2},
        {"detail_metrics": {"attempt": 2, "MAE": 0.1}},
        {"experiment_id": "e1", "domain_id": "other"},
    ):
        with pytest.raises(ArchiveRefusal, match="CONFLICT"):
            warehouse.record_round(**{**kwargs, **changed})
    stored = warehouse.get_rounds("e1")[0]
    assert len(warehouse.get_rounds("e1")) == 1
    assert stored["performance"] == 0.1
    assert json.loads(stored["parameters"]) == {"depth": 2}
    assert stored["metric_schema"] == "schema-a"
    assert json.loads(stored["metrics"])["attempt"] == 1


def test_cross_experiment_round_reuse_is_rejected() -> None:
    """O05/AT03. The same round id with another experiment, attempt slot and performance conflicts."""
    row = _mae_row()
    envelope = build_archive(block=None, candidates=[row])
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e1")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e2")
    assert project_metrics(warehouse, envelope, experiment_id="e1") == 1
    stored = warehouse.get_rounds("e1")[0]
    with pytest.raises(ArchiveRefusal, match="CONFLICT"):
        warehouse.record_round(
            experiment_id="e2",
            domain_id="test",
            round_number=9,
            performance=123.0,
            round_id="r1",
            detail_metrics=json.loads(stored["metrics"]),
        )
    assert warehouse.get_rounds("e2") == []
    assert warehouse.get_rounds("e1")[0]["performance"] == 0.1
    assert warehouse.get_rounds("e1")[0]["experiment_id"] == "e1"


def test_mid_projection_failure_is_not_a_complete_close() -> None:
    """O05/AT03. A failure after the first record leaves no partial projection."""
    _block, envelope = _fixture()
    warehouse = DisposableWarehouse(":memory:")
    experiment_id = warehouse.create_experiment(
        domain_id="fixture-domain",
        node_id="node-fixture",
        hostname="",
    )
    original = warehouse.record_round
    calls = {"n": 0}

    def stop_on_second(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ArchiveRefusal("MID_PROJECTION")
        return original(**kwargs)

    warehouse.record_round = stop_on_second
    with pytest.raises(ArchiveRefusal, match="MID_PROJECTION"):
        project_metrics(warehouse, envelope, experiment_id=experiment_id)
    assert warehouse.get_rounds(experiment_id) == []


def test_manifest_successor_does_not_overwrite_or_become_a_retry(tmp_path) -> None:
    """O03/O05. Same body digest, different manifest digest: both remain, neither is a retry."""
    assert "does not overwrite" in MANIFEST_IDENTITY_RULE
    assert "CONFLICT" in MANIFEST_IDENTITY_RULE
    assert "not a retry" in MANIFEST_IDENTITY_RULE
    block = Block.genesis()
    first = build_archive(block=block, candidates=[_mae_row(performance=0.1)])
    second = build_archive(block=block, candidates=[_mae_row(performance=0.2)])
    assert first.body_digest == second.body_digest
    assert first.manifest_digest != second.manifest_digest
    assert first.body_digest != first.manifest_digest
    assert first.header_hash not in (first.body_digest, first.manifest_digest)
    adapter = FileArchiveAdapter(tmp_path)
    assert adapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    stored_first = adapter.put(first)
    stored_second = adapter.put(second)
    assert stored_first.verified is True
    assert stored_second.verified is True
    assert stored_second.already_stored is False
    assert adapter.read_manifest(first.manifest_digest) == first.manifest
    assert adapter.read_manifest(second.manifest_digest) == second.manifest
    assert adapter.record_count() == 2
    retry = adapter.put(first)
    assert retry.already_stored is True
    assert retry.verified is True
    assert retry.manifest_digest == first.manifest_digest
    assert adapter.record_count() == 2
    index = tmp_path / "index" / f"{second.manifest_digest}.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["body_digest"] = "ab" * 32
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchiveRefusal, match="CONFLICT"):
        adapter.put(second)
    assert adapter.read_manifest(first.manifest_digest) == first.manifest
    assert adapter.read_manifest(second.manifest_digest) == second.manifest


def test_two_writers_and_truncated_recovery_are_not_falsely_verified(tmp_path) -> None:
    """O05/AT02/AT03. Concurrent puts stay content-addressed; a short file is not verified."""
    _block, envelope = _fixture()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    receipts = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            receipts.append(FileArchiveAdapter(tmp_path).put(envelope))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(receipts) == 2
    assert all(item.verified is True for item in receipts)
    adapter = FileArchiveAdapter(tmp_path)
    assert adapter.record_count() == 1
    assert adapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    healthy = adapter.recover()
    assert healthy["verified"] is True
    body_path = tmp_path / "objects" / envelope.body_digest
    body_path.write_bytes(body_path.read_bytes()[:4])
    report = adapter.recover()
    assert report["verified"] is False
    assert report["problems"]
    assert all(item["verified"] is False for item in report["problems"])
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        adapter.read_verified(receipts[0])
    assert adapter.recover()["verified"] is False


def test_fresh_process_rebuilds_a_warehouse_from_the_file_reference(tmp_path) -> None:
    """O04/AT08 partial. The child process never receives the envelope object."""
    envelope = build_archive(block=None, candidates=[_mae_row()])
    root = tmp_path / "archive"
    adapter = FileArchiveAdapter(root)
    receipt = adapter.put(envelope)
    db_path = tmp_path / "warehouse.sqlite"
    script = """
import json, sys
root, manifest_digest, db_path, experiment_id = sys.argv[1:5]
from doin_node.archive.file_adapter import FileArchiveAdapter
from doin_node.archive.warehouse import DisposableWarehouse, project_from_reference
adapter = FileArchiveAdapter(root)
warehouse = DisposableWarehouse(db_path)
warehouse.create_experiment(
    domain_id="test", node_id="n", hostname="", experiment_id=experiment_id
)
inserted = project_from_reference(
    warehouse, adapter, manifest_digest=manifest_digest, experiment_id=experiment_id
)
rows = warehouse.get_rounds(experiment_id)
print(json.dumps({
    "inserted": inserted,
    "label": adapter.LABEL,
    "rows": rows,
}))
"""
    env = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            receipt.manifest_digest,
            str(db_path),
            "e-rebuilt",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["inserted"] == 1
    assert payload["label"] == "DISPOSABLE_FILE_NOT_LAKE"
    row = payload["rows"][0]
    metrics = json.loads(row["metrics"])
    assert metrics["chain_verified"] is False
    assert any(item["name"] == "MAE" and item["value"] == 0.1 for item in metrics["metrics"])
    assert row["performance"] == 0.1
    reopened = DisposableWarehouse(db_path)
    assert len(reopened.get_rounds("e-rebuilt")) == 1
    altered = root / "objects" / next(
        section.digest for section in envelope.sections if section.name == "candidates"
    )
    altered.write_bytes(b"truncated")
    empty = DisposableWarehouse(":memory:")
    empty.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e-bad")
    with pytest.raises(ArchiveRefusal):
        project_from_reference(
            empty,
            FileArchiveAdapter(root),
            manifest_digest=receipt.manifest_digest,
            experiment_id="e-bad",
        )
    assert empty.get_rounds("e-bad") == []


def test_governed_byte_api_refuses_a_bad_readback(tmp_path) -> None:
    """O03 partial. The adapter uses governed put/get and does not call a local directory a lake."""
    _block, envelope = _fixture()
    adapter = FileArchiveAdapter(tmp_path)

    class Store:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.calls = 0
            self.lie = False

        def put_bytes(self, content: bytes, *, grant: str) -> str:
            self.calls += 1
            if grant != "granted-test":
                raise ArchiveRefusal("GRANT_REFUSED")
            digest = digest_bytes(content)
            self.objects[digest] = content
            return digest

        def get_bytes(self, digest: str, *, grant: str) -> bytes:
            if grant != "granted-test":
                raise ArchiveRefusal("GRANT_REFUSED")
            if self.lie:
                return b"not-the-body"
            return self.objects[digest]

    store = Store()
    with pytest.raises(ArchiveRefusal, match="GRANT_REQUIRED"):
        adapter.put_governed(envelope, store, grant="")
    assert store.calls == 0
    receipt = adapter.put_governed(envelope, store, grant="granted-test")
    assert receipt.verified is True
    assert receipt.body_digest == envelope.body_digest
    assert adapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    store.lie = True
    with pytest.raises(ArchiveRefusal):
        adapter.put_governed(envelope, store, grant="granted-test")


class _ReturnedArchive:
    """Source that returns a prepared object and does not check the request."""

    def __init__(self, verified) -> None:
        self.verified = verified

    def load_verified(self, manifest_digest: str):
        del manifest_digest
        return self.verified


def _open_experiment(experiment_id: str) -> DisposableWarehouse:
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(
        domain_id="test",
        node_id="n",
        hostname="",
        experiment_id=experiment_id,
    )
    return warehouse


def _mae_qualifiers(stored: dict) -> dict:
    payload = json.loads(stored["metrics"])
    return next(item for item in payload["metrics"] if item["name"] == "MAE")


def test_correct_reference_projects_bytes_and_keeps_metric_gaps() -> None:
    """A matching manifest digest projects the bytes, not a handwritten attribute."""
    row = _mae_row(performance=0.1)
    row["metrics"] = {"MAE": {"value": 0.1, "unit": "z", "horizon": 96}}
    envelope = build_archive(block=None, candidates=[row])
    verified = verify_envelope(envelope)
    object.__setattr__(verified, "manifest_digest", "ab" * 32)
    assert digest_bytes(verified.manifest) == envelope.manifest_digest
    warehouse = _open_experiment("e-ok")
    inserted = project_from_reference(
        warehouse,
        _ReturnedArchive(verified),
        manifest_digest=envelope.manifest_digest,
        experiment_id="e-ok",
    )
    assert inserted == 1
    stored = warehouse.get_rounds("e-ok")[0]
    assert stored["performance"] == 0.1
    assert json.loads(stored["parameters"]) == {"depth": 2}
    payload = json.loads(stored["metrics"])
    assert payload["chain_verified"] is False
    assert payload["anchor"] == "UNANCHORED"
    mae = _mae_qualifiers(stored)
    assert mae["value"] == 0.1
    assert mae["unit"] == "z"
    assert mae["horizon"] == 96
    assert mae["scale"] == "NOT_COMPARABLE"
    assert mae["population"] == "NOT_COMPARABLE"
    assert mae["reduction"] == "NOT_COMPARABLE"

    bare = build_archive(block=None, candidates=[_mae_row(record_id="r-bare")])
    other = _open_experiment("e-bare")
    assert (
        project_from_reference(
            other,
            _ReturnedArchive(verify_envelope(bare)),
            manifest_digest=bare.manifest_digest,
            experiment_id="e-bare",
        )
        == 1
    )
    bare_mae = _mae_qualifiers(other.get_rounds("e-bare")[0])
    assert bare_mae["value"] == 0.1
    for qualifier in ("scale", "unit", "horizon", "population", "reduction"):
        assert bare_mae[qualifier] == "NOT_COMPARABLE"


def test_wrong_reference_is_rejected_before_insert() -> None:
    """Asking for A while the source returns B inserts nothing. The attribute is ignored."""
    asked = build_archive(block=None, candidates=[_mae_row(performance=0.1)])
    returned = build_archive(block=None, candidates=[_mae_row(performance=0.2)])
    assert asked.manifest_digest != returned.manifest_digest
    loaded = verify_envelope(returned)
    object.__setattr__(loaded, "manifest_digest", asked.manifest_digest)
    assert loaded.manifest_digest == asked.manifest_digest
    assert digest_bytes(loaded.manifest) == returned.manifest_digest
    warehouse = _open_experiment("e-wrong")

    def refuse_insert(**kwargs):
        raise AssertionError("inserted a different manifest")

    warehouse.record_round = refuse_insert
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        project_from_reference(
            warehouse,
            _ReturnedArchive(loaded),
            manifest_digest=asked.manifest_digest,
            experiment_id="e-wrong",
        )
    assert warehouse.get_rounds("e-wrong") == []

    matched = _open_experiment("e-bytes")
    inserted = project_from_reference(
        matched,
        _ReturnedArchive(loaded),
        manifest_digest=returned.manifest_digest,
        experiment_id="e-bytes",
    )
    assert inserted == 1
    assert matched.get_rounds("e-bytes")[0]["performance"] == 0.2


def test_mutated_record_dict_does_not_change_the_projected_row() -> None:
    """A post-verify edit of the record dict is not the inserted row."""
    envelope = build_archive(block=None, candidates=[_mae_row(performance=0.1)])
    verified = verify_envelope(envelope)
    verified.records[0]["performance"] = 999.0
    verified.records[0]["parameters"] = {"depth": 999}
    verified.records[0]["metrics"] = {"MAE": 999.0}
    object.__setattr__(verified, "chain_verified", True)
    warehouse = _open_experiment("e-mut")
    inserted = project_from_reference(
        warehouse,
        _ReturnedArchive(verified),
        manifest_digest=envelope.manifest_digest,
        experiment_id="e-mut",
    )
    assert inserted == 1
    stored = warehouse.get_rounds("e-mut")[0]
    assert stored["performance"] == 0.1
    assert json.loads(stored["parameters"]) == {"depth": 2}
    payload = json.loads(stored["metrics"])
    assert payload["chain_verified"] is False
    mae = _mae_qualifiers(stored)
    assert mae["value"] == 0.1
    for qualifier in ("scale", "unit", "horizon", "population", "reduction"):
        assert mae[qualifier] == "NOT_COMPARABLE"

    broken = verify_envelope(envelope)
    object.__setattr__(broken, "body", b"{}")
    broken.records[0]["performance"] = 999.0
    empty = _open_experiment("e-broken")

    def refuse_insert(**kwargs):
        raise AssertionError("inserted unverifiable bytes")

    empty.record_round = refuse_insert
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        project_from_reference(
            empty,
            _ReturnedArchive(broken),
            manifest_digest=envelope.manifest_digest,
            experiment_id="e-broken",
        )
    assert empty.get_rounds("e-broken") == []


def test_missing_reference_file_inserts_nothing(tmp_path) -> None:
    """A missing manifest file is a refusal. The directory is not a lake."""
    envelope = build_archive(block=None, candidates=[_mae_row()])
    root = tmp_path / "archive"
    adapter = FileArchiveAdapter(root)
    assert adapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    adapter.put(envelope)
    (root / "manifests" / envelope.manifest_digest).unlink()
    warehouse = _open_experiment("e-missing")

    def refuse_insert(**kwargs):
        raise AssertionError("inserted from a missing file")

    warehouse.record_round = refuse_insert
    with pytest.raises(ArchiveRefusal, match="MISSING"):
        project_from_reference(
            warehouse,
            adapter,
            manifest_digest=envelope.manifest_digest,
            experiment_id="e-missing",
        )
    assert warehouse.get_rounds("e-missing") == []


def test_reference_retry_returns_previous_and_rejects_discrepancy(tmp_path) -> None:
    """Same bytes keep the previous row. A changed compared column conflicts."""
    row = _mae_row()
    envelope = build_archive(block=None, candidates=[row])
    adapter = FileArchiveAdapter(tmp_path)
    receipt = adapter.put(envelope)
    assert adapter.LABEL == "DISPOSABLE_FILE_NOT_LAKE"
    warehouse = DisposableWarehouse(":memory:")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e1")
    warehouse.create_experiment(domain_id="test", node_id="n", hostname="", experiment_id="e2")
    assert (
        project_from_reference(
            warehouse,
            adapter,
            manifest_digest=receipt.manifest_digest,
            experiment_id="e1",
        )
        == 1
    )
    previous = warehouse.get_rounds("e1")[0]
    assert (
        project_from_reference(
            warehouse,
            adapter,
            manifest_digest=receipt.manifest_digest,
            experiment_id="e1",
        )
        == 0
    )
    replay = warehouse.get_rounds("e1")
    assert len(replay) == 1
    assert replay[0]["round_id"] == previous["round_id"] == "r1"
    assert replay[0]["experiment_id"] == "e1"
    assert replay[0]["performance"] == previous["performance"]
    assert replay[0]["parameters"] == previous["parameters"]
    assert replay[0]["metric_schema"] == previous["metric_schema"]
    assert replay[0]["metrics"] == previous["metrics"]
    with pytest.raises(ArchiveRefusal, match="CONFLICT"):
        project_from_reference(
            warehouse,
            adapter,
            manifest_digest=receipt.manifest_digest,
            experiment_id="e2",
        )
    assert warehouse.get_rounds("e2") == []

    for changes in (
        {"attempt": 2},
        {"performance": 0.2},
        {"parameters": {"depth": 9}},
        {"metrics": {"MAE": 0.9}},
    ):
        variant = build_archive(block=None, candidates=[{**row, **changes}])
        with pytest.raises(ArchiveRefusal, match="CONFLICT"):
            project_from_reference(
                warehouse,
                _ReturnedArchive(verify_envelope(variant)),
                manifest_digest=variant.manifest_digest,
                experiment_id="e1",
            )
        rows = warehouse.get_rounds("e1")
        assert len(rows) == 1
        assert rows[0]["round_id"] == previous["round_id"]
        assert rows[0]["experiment_id"] == "e1"
        assert rows[0]["performance"] == previous["performance"]
        assert rows[0]["parameters"] == previous["parameters"]
        assert rows[0]["metrics"] == previous["metrics"]
    assert warehouse.get_rounds("e2") == []


def test_reference_rollback_does_not_present_a_partial_close() -> None:
    """A failure after the first reference row leaves no partial projection."""
    envelope = build_archive(
        block=None,
        candidates=[
            _mae_row(record_id="r1", performance=0.1),
            _mae_row(record_id="r2", performance=0.3),
        ],
    )
    verified = verify_envelope(envelope)
    warehouse = _open_experiment("e-roll")
    original = warehouse.record_round
    calls = {"n": 0}

    def stop_on_second(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ArchiveRefusal("MID_PROJECTION")
        return original(**kwargs)

    warehouse.record_round = stop_on_second
    with pytest.raises(ArchiveRefusal, match="MID_PROJECTION"):
        project_from_reference(
            warehouse,
            _ReturnedArchive(verified),
            manifest_digest=envelope.manifest_digest,
            experiment_id="e-roll",
        )
    assert calls["n"] == 2
    assert warehouse.get_rounds("e-roll") == []


def test_projection_count_is_not_a_truncated_read() -> None:
    """get_rounds may stop at its limit. The projection count is every committed row."""
    total = 1001
    envelope = build_archive(
        block=None,
        candidates=[_mae_row(record_id=f"r{index}") for index in range(total)],
    )
    warehouse = _open_experiment("e-count")
    inserted = project_from_reference(
        warehouse,
        _ReturnedArchive(verify_envelope(envelope)),
        manifest_digest=envelope.manifest_digest,
        experiment_id="e-count",
    )
    assert inserted == total
    assert len(warehouse.get_rounds("e-count")) == 1000
    assert len(warehouse.get_rounds("e-count", limit=1)) == 1
    assert len(warehouse.get_rounds("e-count", limit=1)) < inserted
