"""Disposable analytical double.

The call shape matches ``OLAPDatabase.create_experiment`` and ``record_round``.
Rows stay in this process's SQLite connection. PostgreSQL is refused. The
label DISPOSABLE means these rows are not the node's warehouse and not a
Metabase database.

Projection calls ``record_round`` once per candidate and per retry, including
attempts that did not win. A repeated projection of the same record id does
not insert a second row. Archive success is not chain verification.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from doin_core.archive.body import (
    ArchiveEnvelope,
    ArchiveRefusal,
    VerifiedArchive,
    digest_bytes,
    verify_archive_bytes,
    verify_envelope,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_experiment (
    experiment_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    hostname TEXT
);
CREATE TABLE IF NOT EXISTS fact_round (
    round_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    performance REAL NOT NULL,
    parameters TEXT NOT NULL,
    metric_schema TEXT NOT NULL,
    metrics TEXT NOT NULL
);
"""

_GAP = "NOT_COMPARABLE"
_QUALIFIERS = ("scale", "unit", "horizon", "population", "reduction")


class DisposableWarehouse:
    """In-process SQLite double of the node's analytical row calls."""

    LABEL = "DISPOSABLE"

    def __init__(self, db_path: str | Path) -> None:
        text = str(db_path)
        if text.lower().startswith("postgres"):
            raise ArchiveRefusal("POSTGRES_REFUSED")
        self._db_path = text
        if text != ":memory:":
            Path(text).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._bulk = 0
        self._conn = sqlite3.connect(text, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def describe(self) -> dict[str, Any]:
        return {"label": self.LABEL, "backend": "sqlite-double", "postgres": False}

    def create_experiment(
        self,
        *,
        domain_id: str,
        node_id: str,
        hostname: str = "",
        optimizer_plugin: str = "",
        performance_metric: str = "fitness",
        metric_schema: str = "",
        higher_is_better: bool = True,
        optimization_config: dict[str, Any] | None = None,
        param_bounds: dict[str, Any] | None = None,
        target_performance: float | None = None,
        doin_version: str = "",
        experiment_id: str | None = None,
    ) -> str:
        del optimizer_plugin, performance_metric, metric_schema, higher_is_better
        del optimization_config, param_bounds, target_performance, doin_version
        eid = experiment_id or uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO dim_experiment
                   (experiment_id, domain_id, node_id, hostname)
                   VALUES (?, ?, ?, ?)""",
                (eid, domain_id, node_id, hostname),
            )
        return eid

    def record_round(
        self,
        *,
        experiment_id: str,
        domain_id: str,
        round_number: int,
        performance: float,
        best_performance: float | None = None,
        performance_delta: float | None = None,
        is_improvement: bool = False,
        parameters: dict[str, Any] | None = None,
        best_parameters: dict[str, Any] | None = None,
        wall_clock_seconds: float = 0.0,
        elapsed_seconds: float = 0.0,
        time_to_current_best_seconds: float = 0.0,
        time_to_target_seconds: float | None = None,
        chain_height: int = 0,
        peers_count: int = 0,
        block_reward_earned: float = 0.0,
        converged: bool = False,
        round_id: str | None = None,
        detail_metrics: dict[str, Any] | None = None,
        metric_schema: str = "",
    ) -> str:
        del best_performance, performance_delta, is_improvement
        del best_parameters, wall_clock_seconds, elapsed_seconds
        del time_to_current_best_seconds, time_to_target_seconds, chain_height
        del peers_count, block_reward_earned, converged
        if isinstance(performance, bool) or not isinstance(performance, (int, float)):
            raise ArchiveRefusal("PERFORMANCE_MUST_BE_A_NUMBER")
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise ArchiveRefusal("ROUND_NUMBER")
        rid = round_id or uuid.uuid4().hex
        parameters_text = _parameters_text(parameters)
        schema_text = metric_schema if isinstance(metric_schema, str) else _GAP
        metrics = _canonical_json(detail_metrics or {})
        with self._lock:
            if self._bulk == 0:
                self._conn.execute("BEGIN")
            try:
                existing = self._conn.execute(
                    """SELECT experiment_id, domain_id, round_number, performance,
                              parameters, metric_schema, metrics
                       FROM fact_round WHERE round_id=?""",
                    (rid,),
                ).fetchone()
                if existing is not None:
                    same = (
                        existing["experiment_id"] == experiment_id
                        and existing["domain_id"] == domain_id
                        and int(existing["round_number"]) == round_number
                        and existing["performance"] == performance
                        and existing["parameters"] == parameters_text
                        and existing["metric_schema"] == schema_text
                        and existing["metrics"] == metrics
                    )
                    if not same:
                        raise ArchiveRefusal("CONFLICT")
                    if self._bulk == 0:
                        self._conn.commit()
                    return rid
                self._conn.execute(
                    """INSERT INTO fact_round
                       (round_id, experiment_id, domain_id, round_number,
                        performance, parameters, metric_schema, metrics)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rid,
                        experiment_id,
                        domain_id,
                        round_number,
                        performance,
                        parameters_text,
                        schema_text,
                        metrics,
                    ),
                )
                if self._bulk == 0:
                    self._conn.commit()
            except Exception:
                if self._bulk == 0:
                    self._conn.rollback()
                raise
        return rid

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """One transaction for a projection. A failure rolls every new row back."""
        with self._lock:
            self._conn.execute("BEGIN")
            self._bulk += 1
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._bulk -= 1

    def get_rounds(self, experiment_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM fact_round
                   WHERE experiment_id=? ORDER BY round_number LIMIT ?""",
                (experiment_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _round_count(self, experiment_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM fact_round WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def project_metrics(
    warehouse: DisposableWarehouse,
    envelope: ArchiveEnvelope,
    *,
    experiment_id: str,
) -> int:
    """Project rows from a verified envelope, not from unchecked section objects.

    ``chain_verified`` stays false: a stored archive is not a consensus result.
    UNANCHORED is written as such when the round had no block.
    """
    verified = verify_envelope(envelope)
    return _project_verified(warehouse, verified, experiment_id=experiment_id)


def project_from_reference(
    warehouse: DisposableWarehouse,
    source: Any,
    *,
    manifest_digest: str,
    experiment_id: str,
) -> int:
    """Project the archive whose manifest bytes hash to ``manifest_digest``.

    Records are parsed from those bytes. A mismatch is refused before any insert.
    """
    load = getattr(source, "load_verified", None)
    if load is None:
        raise ArchiveRefusal("FILE_REFERENCE_REQUIRED")
    loaded = load(manifest_digest)
    if not isinstance(loaded, VerifiedArchive):
        raise ArchiveRefusal("FILE_REFERENCE_REQUIRED")
    verified = _rebuild_for_reference(loaded, manifest_digest)
    return _project_verified(warehouse, verified, experiment_id=experiment_id)


def _rebuild_for_reference(loaded: VerifiedArchive, manifest_digest: str) -> VerifiedArchive:
    """Re-parse body, manifest and sections. Do not read ``loaded.records``."""
    manifest = loaded.manifest
    # The requested digest is the hash of these bytes, not loaded.manifest_digest.
    if digest_bytes(manifest) != manifest_digest:
        raise ArchiveRefusal("DIGEST_MISMATCH")
    body = loaded.body
    if not isinstance(body, bytes):
        raise ArchiveRefusal("DIGEST_REQUIRES_BYTES")
    rebuilt = verify_archive_bytes(
        body=body,
        manifest=manifest,
        sections=_section_bytes(loaded),
    )
    if rebuilt.chain_verified:
        raise ArchiveRefusal("CHAIN_NOT_VERIFIED")
    if digest_bytes(rebuilt.manifest) != manifest_digest or rebuilt.manifest_digest != manifest_digest:
        raise ArchiveRefusal("DIGEST_MISMATCH")
    return rebuilt


def _section_bytes(loaded: VerifiedArchive) -> dict[str, bytes]:
    sections: dict[str, bytes] = {}
    for section in loaded.sections:
        content = getattr(section, "content", None)
        if not isinstance(content, bytes):
            raise ArchiveRefusal("DIGEST_REQUIRES_BYTES")
        digest = digest_bytes(content)
        previous = sections.get(digest)
        if previous is not None and previous != content:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        sections[digest] = content
    return sections


def metric_contract(metrics: Any) -> list[dict[str, Any]]:
    """Copy declared metric fields. Missing qualifiers stay ``NOT_COMPARABLE``."""
    if not isinstance(metrics, dict):
        raise ArchiveRefusal("METRIC_CONTRACT")
    rows: list[dict[str, Any]] = []
    for name, raw in metrics.items():
        if isinstance(raw, dict) and "value" in raw:
            row: dict[str, Any] = {"name": name, "value": raw["value"]}
            for qualifier in _QUALIFIERS:
                if qualifier in raw and raw[qualifier] is not None:
                    row[qualifier] = raw[qualifier]
                else:
                    row[qualifier] = _GAP
        else:
            row = {"name": name, "value": raw}
            for qualifier in _QUALIFIERS:
                row[qualifier] = _GAP
        rows.append(row)
    _canonical_json(rows)
    return rows


def _project_verified(
    warehouse: DisposableWarehouse,
    verified: VerifiedArchive,
    *,
    experiment_id: str,
) -> int:
    if verified.chain_verified:
        raise ArchiveRefusal("CHAIN_NOT_VERIFIED")
    chain_height = 0
    if verified.anchor == "ANCHORED":
        chain_height = int(json.loads(verified.body)["header"]["index"])
    with warehouse.bulk():
        before = warehouse._round_count(experiment_id)
        for seq, item in enumerate(verified.records):
            parameters = item["parameters"] if isinstance(item.get("parameters"), dict) else None
            provenance = item["provenance"] if "provenance" in item else _GAP
            warehouse.record_round(
                experiment_id=experiment_id,
                domain_id=item["domain_id"],
                round_number=seq,
                performance=item["performance"],
                is_improvement=False,
                parameters=parameters,
                chain_height=chain_height,
                peers_count=0,
                block_reward_earned=0.0,
                converged=False,
                round_id=item["record_id"],
                detail_metrics={
                    "anchor": verified.anchor,
                    "chain_verified": False,
                    "won": item["won"],
                    "attempt": item["attempt"],
                    "candidate_id": item["candidate_id"],
                    "kind": item["kind"],
                    "header_hash": verified.header_hash,
                    "body_digest": verified.body_digest,
                    "manifest_digest": verified.manifest_digest,
                    "metrics": metric_contract(item.get("metrics")),
                    "parameters": parameters if parameters is not None else _GAP,
                    "provenance": provenance,
                },
                metric_schema="doin.archive_projection.v1",
            )
        # get_rounds is capped. The caller needs the rows this transaction committed.
        inserted = warehouse._round_count(experiment_id) - before
    return inserted


def _parameters_text(parameters: dict[str, Any] | None) -> str:
    if parameters is None:
        return _GAP
    if not isinstance(parameters, dict):
        raise ArchiveRefusal("RECORD_MAPPING_EXPECTED")
    return _canonical_json(parameters)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArchiveRefusal(f"NOT_CANONICAL: {exc}") from exc
