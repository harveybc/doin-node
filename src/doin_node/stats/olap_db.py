"""Thread-safe SQLite OLAP database manager for experiment tracking.

Each DOIN node keeps a local ``olap.db`` that records every optimization
round automatically.  The file can later be synced to a central PostgreSQL
instance via :mod:`doin_node.stats.olap_sync`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doin_node.stats.olap_schema import SCHEMA_SQL, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class OLAPDatabase:
    """Manage a local SQLite OLAP database.

    * WAL mode for concurrent reads while the optimizer writes.
    * All public methods are thread-safe (internal lock).
    * Schema is auto-created / migrated on first open.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ── Schema migration ─────────────────────────────────────────

    def _migrate(self) -> None:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_version'"
        )
        if cur.fetchone() is None:
            # Fresh database – apply full schema
            self._conn.executescript(SCHEMA_SQL)
            self._conn.execute(
                "INSERT INTO _schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _now_iso()),
            )
            logger.info("OLAP schema v%d created at %s", SCHEMA_VERSION, self._db_path)
            return

        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM _schema_version"
        ).fetchone()
        current = row["v"] if row else 0
        if current < SCHEMA_VERSION:
            # Future migrations go here
            self._conn.executescript(SCHEMA_SQL)  # idempotent CREATE IF NOT EXISTS
            self._conn.execute(
                "INSERT INTO _schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _now_iso()),
            )
            logger.info("OLAP schema migrated %d → %d", current, SCHEMA_VERSION)

    # ── Experiment lifecycle ─────────────────────────────────────

    def create_experiment(
        self,
        *,
        domain_id: str,
        node_id: str,
        hostname: str = "",
        optimizer_plugin: str = "",
        optimization_config: dict[str, Any] | None = None,
        param_bounds: dict[str, Any] | None = None,
        target_performance: float | None = None,
        doin_version: str = "",
        experiment_id: str | None = None,
    ) -> str:
        eid = experiment_id or uuid.uuid4().hex
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO dim_domain (domain_id) VALUES (?)",
                (domain_id,),
            )
            self._conn.execute(
                """INSERT INTO dim_experiment
                   (experiment_id, domain_id, node_id, hostname,
                    optimizer_plugin, optimization_config, param_bounds,
                    target_performance, started_at, doin_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    eid, domain_id, node_id, hostname,
                    optimizer_plugin,
                    json.dumps(optimization_config or {}),
                    json.dumps(param_bounds or {}),
                    target_performance, now, doin_version,
                ),
            )
        logger.debug("OLAP experiment created: %s domain=%s", eid[:12], domain_id)
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
    ) -> str:
        rid = round_id or uuid.uuid4().hex
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO fact_round
                   (round_id, experiment_id, domain_id, round_number, timestamp_utc,
                    performance, best_performance, performance_delta, is_improvement,
                    parameters, best_parameters,
                    wall_clock_seconds, elapsed_seconds,
                    time_to_current_best_seconds, time_to_target_seconds,
                    chain_height, peers_count, block_reward_earned, converged)
                   VALUES (?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?, ?,?,?,?)""",
                (
                    rid, experiment_id, domain_id, round_number, now,
                    performance, best_performance, performance_delta, is_improvement,
                    json.dumps(parameters) if parameters else "{}",
                    json.dumps(best_parameters) if best_parameters else "{}",
                    wall_clock_seconds, elapsed_seconds,
                    time_to_current_best_seconds, time_to_target_seconds,
                    chain_height, peers_count, block_reward_earned, converged,
                ),
            )
        return rid

    def mark_converged(self, experiment_id: str) -> None:
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE dim_experiment SET converged=TRUE, finished_at=? WHERE experiment_id=?",
                (now, experiment_id),
            )

    def finalize_experiment(self, experiment_id: str) -> None:
        """Compute and write ``fact_experiment_summary`` from recorded rounds."""
        now = _now_iso()
        with self._lock:
            # Update finished_at
            self._conn.execute(
                "UPDATE dim_experiment SET finished_at=COALESCE(finished_at, ?) WHERE experiment_id=?",
                (now, experiment_id),
            )
            exp = self._conn.execute(
                "SELECT * FROM dim_experiment WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if exp is None:
                return

            agg = self._conn.execute(
                """SELECT COUNT(*) AS total_rounds,
                          MAX(performance) AS best_perf,
                          MAX(elapsed_seconds) AS total_elapsed,
                          MAX(time_to_target_seconds) AS ttt
                   FROM fact_round WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()

            last = self._conn.execute(
                "SELECT performance FROM fact_round WHERE experiment_id=? ORDER BY round_number DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()

            converged = bool(exp["converged"])
            rounds_to_conv = None
            if converged:
                first_conv = self._conn.execute(
                    "SELECT round_number FROM fact_round WHERE experiment_id=? AND converged=TRUE ORDER BY round_number LIMIT 1",
                    (experiment_id,),
                ).fetchone()
                if first_conv:
                    rounds_to_conv = first_conv["round_number"]

            self._conn.execute(
                """INSERT OR REPLACE INTO fact_experiment_summary
                   (experiment_id, domain_id, total_rounds, final_performance,
                    best_performance, time_to_target_seconds, total_elapsed_seconds,
                    converged, rounds_to_convergence, node_id, hostname)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id, exp["domain_id"],
                    agg["total_rounds"],
                    last["performance"] if last else None,
                    agg["best_perf"],
                    agg["ttt"],
                    agg["total_elapsed"],
                    converged,
                    rounds_to_conv,
                    exp["node_id"],
                    exp["hostname"],
                ),
            )

    # ── Queries ──────────────────────────────────────────────────

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM fact_experiment_summary WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM fact_experiment_summary").fetchall()
            return [dict(r) for r in rows]

    def get_all_experiments(self) -> list[dict[str, Any]]:
        """Return all experiments from dim_experiment (even if not yet finalized)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM dim_experiment").fetchall()
            return [dict(r) for r in rows]

    def get_rounds(self, experiment_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fact_round WHERE experiment_id=? ORDER BY round_number LIMIT ?",
                (experiment_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def ingest_from_chain(self, blocks: list) -> int:
        """Walk blockchain blocks, extract experiment metrics from OPTIMAE_ACCEPTED
        transactions, and insert into fact_chain_optimae.

        Returns count of new records ingested. Idempotent — uses optimae_id as dedup key.
        """
        from doin_node.stats.chain_metrics import collect_chain_metrics
        rows = collect_chain_metrics(blocks)
        now = _now_iso()
        count = 0
        with self._lock:
            for row in rows:
                oid = row.get("optimae_id")
                if not oid:
                    continue
                try:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO fact_chain_optimae
                           (optimae_id, domain_id, optimizer_id, experiment_id,
                            round_number, parameters, reported_performance,
                            verified_performance, effective_increment,
                            time_to_this_result_seconds, optimization_config_hash,
                            data_hash, previous_best_performance, reward_fraction,
                            quorum_agree_fraction, block_height, block_timestamp,
                            ingested_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            oid,
                            row.get("domain_id", ""),
                            row.get("optimizer_id", ""),
                            row.get("experiment_id"),
                            row.get("round_number"),
                            json.dumps(row.get("parameters")) if row.get("parameters") else None,
                            row.get("reported_performance"),
                            row.get("verified_performance"),
                            row.get("effective_increment"),
                            row.get("time_to_this_result_seconds"),
                            row.get("optimization_config_hash"),
                            row.get("data_hash"),
                            row.get("previous_best_performance"),
                            row.get("reward_fraction"),
                            row.get("quorum_agree_fraction"),
                            row.get("block_height", 0),
                            row.get("block_timestamp", ""),
                            now,
                        ),
                    )
                    if self._conn.execute("SELECT changes()").fetchone()[0] > 0:
                        count += 1
                except Exception:
                    logger.exception("Failed to ingest chain optimae %s", oid)
        return count

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
