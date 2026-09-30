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
from pathlib import Path
from typing import Any

from doin_core.archive.body import ArchiveEnvelope, ArchiveRefusal

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
    metrics TEXT NOT NULL
);
"""


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
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(text, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
            self._conn.commit()
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
        del best_performance, performance_delta, is_improvement, parameters
        del best_parameters, wall_clock_seconds, elapsed_seconds
        del time_to_current_best_seconds, time_to_target_seconds, chain_height
        del peers_count, block_reward_earned, converged, metric_schema
        rid = round_id or uuid.uuid4().hex
        metrics = json.dumps(
            detail_metrics or {},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            existing = self._conn.execute(
                "SELECT domain_id, metrics FROM fact_round WHERE round_id=?",
                (rid,),
            ).fetchone()
            if existing is not None:
                same = (
                    existing["domain_id"] == domain_id
                    and existing["metrics"] == metrics
                )
                if same:
                    return rid
                raise ArchiveRefusal("CONFLICT")
            self._conn.execute(
                """INSERT INTO fact_round
                   (round_id, experiment_id, domain_id, round_number,
                    performance, metrics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rid, experiment_id, domain_id, round_number, performance, metrics),
            )
            self._conn.commit()
        return rid

    def get_rounds(self, experiment_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM fact_round
                   WHERE experiment_id=? ORDER BY round_number LIMIT ?""",
                (experiment_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def project_metrics(
    warehouse: DisposableWarehouse,
    envelope: ArchiveEnvelope,
    *,
    experiment_id: str,
) -> int:
    """Project candidate and retry rows. Non-winners are included.

    ``chain_verified`` stays false: a stored archive is not a consensus result.
    UNANCHORED is written as such when the round had no block.
    """
    before = len(warehouse.get_rounds(experiment_id))
    records: list[dict[str, Any]] = []
    for section in envelope.sections:
        if section.name in ("candidates", "retries"):
            parsed = json.loads(section.content)
            if isinstance(parsed, list):
                records.extend(parsed)
    chain_height = 0
    if envelope.anchor == "ANCHORED":
        chain_height = int(json.loads(envelope.body)["header"]["index"])
    for seq, item in enumerate(records):
        warehouse.record_round(
            experiment_id=experiment_id,
            domain_id=item["domain_id"],
            round_number=seq,
            performance=item["performance"],
            is_improvement=False,
            parameters=item.get("parameters") or {},
            chain_height=chain_height,
            peers_count=0,
            block_reward_earned=0.0,
            converged=False,
            round_id=item["record_id"],
            detail_metrics={
                "anchor": envelope.anchor,
                "chain_verified": False,
                "won": item["won"],
                "attempt": item["attempt"],
                "candidate_id": item["candidate_id"],
                "kind": item["kind"],
                "header_hash": envelope.header_hash,
                "body_digest": envelope.body_digest,
                "manifest_digest": envelope.manifest_digest,
            },
            metric_schema="doin.archive_projection.v1",
        )
    return len(warehouse.get_rounds(experiment_id)) - before
