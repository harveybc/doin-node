"""Sync local SQLite OLAP → central PostgreSQL.

Usage::

    python -m doin_node.stats.olap_sync \\
        --source ./olap.db \\
        --target postgresql://user:pass@host/db

Idempotent — safe to run repeatedly.  Uses ``INSERT ... ON CONFLICT DO UPDATE``
(upsert) so rows are never duplicated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_pg_schema(pg_conn) -> None:
    """Create PostgreSQL tables if they don't exist (same schema, PG types)."""
    cur = pg_conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_domain (
        domain_id TEXT PRIMARY KEY,
        domain_type TEXT,
        description TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dim_experiment (
        experiment_id TEXT PRIMARY KEY,
        domain_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        hostname TEXT,
        optimizer_plugin TEXT,
        optimization_config JSONB,
        param_bounds JSONB,
        target_performance DOUBLE PRECISION,
        started_at TEXT,
        finished_at TEXT,
        converged BOOLEAN DEFAULT FALSE,
        doin_version TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_round (
        round_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES dim_experiment(experiment_id),
        domain_id TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        timestamp_utc TEXT NOT NULL,
        performance DOUBLE PRECISION NOT NULL,
        best_performance DOUBLE PRECISION,
        performance_delta DOUBLE PRECISION,
        is_improvement BOOLEAN,
        parameters JSONB,
        best_parameters JSONB,
        wall_clock_seconds DOUBLE PRECISION,
        elapsed_seconds DOUBLE PRECISION,
        time_to_current_best_seconds DOUBLE PRECISION,
        time_to_target_seconds DOUBLE PRECISION,
        chain_height INTEGER DEFAULT 0,
        peers_count INTEGER DEFAULT 0,
        block_reward_earned DOUBLE PRECISION DEFAULT 0.0,
        converged BOOLEAN DEFAULT FALSE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_experiment_summary (
        experiment_id TEXT PRIMARY KEY REFERENCES dim_experiment(experiment_id),
        domain_id TEXT NOT NULL,
        total_rounds INTEGER,
        final_performance DOUBLE PRECISION,
        best_performance DOUBLE PRECISION,
        time_to_target_seconds DOUBLE PRECISION,
        total_elapsed_seconds DOUBLE PRECISION,
        converged BOOLEAN,
        rounds_to_convergence INTEGER,
        node_id TEXT,
        hostname TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS _sync_state (
        source_path TEXT PRIMARY KEY,
        last_sync_utc TEXT
    )""")
    pg_conn.commit()


def _upsert_rows(pg_cur, table: str, rows: list[dict], pk: str) -> int:
    """Upsert rows into a PostgreSQL table. Returns count."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != pk)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk}) DO UPDATE SET {updates}"
    )
    count = 0
    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            # Convert JSON strings to actual JSON for JSONB columns
            if isinstance(v, str) and v.startswith("{"):
                try:
                    json.loads(v)
                    # It's valid JSON string — psycopg2 will handle it
                except (json.JSONDecodeError, ValueError):
                    pass
            vals.append(v)
        pg_cur.execute(sql, vals)
        count += 1
    return count


def sync(source_path: str, target_dsn: str) -> dict:
    """Sync SQLite OLAP → PostgreSQL. Returns stats dict."""
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return {"error": "psycopg2 not installed"}

    sqlite_conn = _open_sqlite(source_path)
    pg_conn = psycopg2.connect(target_dsn)

    _ensure_pg_schema(pg_conn)
    pg_cur = pg_conn.cursor()

    stats: dict[str, int] = {}

    # Sync dimensions first, then facts
    # Ensure fact_chain_optimae exists in PG
    pg_cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_chain_optimae (
        optimae_id TEXT PRIMARY KEY,
        domain_id TEXT NOT NULL,
        optimizer_id TEXT NOT NULL,
        experiment_id TEXT,
        round_number INTEGER,
        parameters JSONB,
        reported_performance DOUBLE PRECISION,
        verified_performance DOUBLE PRECISION,
        effective_increment DOUBLE PRECISION,
        time_to_this_result_seconds DOUBLE PRECISION,
        optimization_config_hash TEXT,
        data_hash TEXT,
        previous_best_performance DOUBLE PRECISION,
        reward_fraction DOUBLE PRECISION,
        quorum_agree_fraction DOUBLE PRECISION,
        block_height INTEGER NOT NULL,
        block_timestamp TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    )""")
    pg_conn.commit()

    for table, pk in [
        ("dim_domain", "domain_id"),
        ("dim_experiment", "experiment_id"),
        ("fact_round", "round_id"),
        ("fact_experiment_summary", "experiment_id"),
        ("fact_chain_optimae", "optimae_id"),
    ]:
        rows = [dict(r) for r in sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()]
        count = _upsert_rows(pg_cur, table, rows, pk)
        stats[table] = count
        logger.info("Synced %d rows to %s", count, table)

    # Record sync timestamp
    now = datetime.now(timezone.utc).isoformat()
    pg_cur.execute(
        "INSERT INTO _sync_state (source_path, last_sync_utc) VALUES (%s, %s) "
        "ON CONFLICT (source_path) DO UPDATE SET last_sync_utc=EXCLUDED.last_sync_utc",
        (source_path, now),
    )

    pg_conn.commit()
    pg_cur.close()
    pg_conn.close()
    sqlite_conn.close()

    stats["synced_at"] = now  # type: ignore[assignment]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local OLAP SQLite → PostgreSQL")
    parser.add_argument("--source", required=True, help="Path to local olap.db")
    parser.add_argument("--target", required=True, help="PostgreSQL DSN")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    result = sync(args.source, args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
