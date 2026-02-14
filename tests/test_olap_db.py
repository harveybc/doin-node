"""Comprehensive tests for the OLAP database layer."""

import json
import os
import tempfile
import threading
import time

import pytest

from doin_node.stats.olap_db import OLAPDatabase
from doin_node.stats.olap_schema import SCHEMA_VERSION
from doin_node.stats.experiment_tracker import ExperimentTracker


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_olap.db"
    db = OLAPDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def tmp_tracker(tmp_path):
    csv_path = tmp_path / "stats.csv"
    olap_path = tmp_path / "olap.db"
    tracker = ExperimentTracker(csv_path, node_id="node-1", olap_db_path=olap_path)
    yield tracker


# ── 1. Schema creation and migration ────────────────────────────

class TestSchemaCreation:
    def test_creates_tables(self, tmp_db):
        import sqlite3
        conn = sqlite3.connect(tmp_db.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "dim_experiment" in tables
        assert "dim_domain" in tables
        assert "fact_round" in tables
        assert "fact_experiment_summary" in tables
        assert "_schema_version" in tables
        conn.close()

    def test_schema_version(self, tmp_db):
        import sqlite3
        conn = sqlite3.connect(tmp_db.db_path)
        row = conn.execute("SELECT MAX(version) AS v FROM _schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION
        conn.close()

    def test_wal_mode(self, tmp_db):
        import sqlite3
        conn = sqlite3.connect(tmp_db.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_idempotent_migration(self, tmp_path):
        db_path = tmp_path / "idem.db"
        db1 = OLAPDatabase(db_path)
        db1.close()
        # Reopen — should not fail
        db2 = OLAPDatabase(db_path)
        db2.close()

    def test_indexes_created(self, tmp_db):
        import sqlite3
        conn = sqlite3.connect(tmp_db.db_path)
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        assert "idx_fact_round_experiment" in indexes
        assert "idx_fact_round_domain" in indexes
        assert "idx_fact_round_timestamp" in indexes
        assert "idx_fact_round_performance" in indexes
        conn.close()


# ── 2. Experiment lifecycle ──────────────────────────────────────

class TestExperimentLifecycle:
    def test_create_experiment(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="quadratic",
            node_id="node-1",
            hostname="test-host",
            optimizer_plugin="bayesian",
            optimization_config={"n_iter": 100},
            param_bounds={"x": [-5, 5]},
            target_performance=0.99,
            doin_version="0.1.0",
        )
        assert len(eid) == 32  # hex UUID

    def test_create_with_custom_id(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="test",
            node_id="n1",
            experiment_id="custom-id-123",
        )
        assert eid == "custom-id-123"

    def test_full_lifecycle(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="quadratic",
            node_id="node-1",
            target_performance=0.95,
        )

        # Record rounds
        for i in range(1, 6):
            perf = i * 0.2
            tmp_db.record_round(
                experiment_id=eid,
                domain_id="quadratic",
                round_number=i,
                performance=perf,
                best_performance=perf,
                is_improvement=True,
                parameters={"x": [float(i)]},
                elapsed_seconds=float(i),
                converged=(perf >= 0.95),
                time_to_target_seconds=float(i) if perf >= 0.95 else None,
            )

        # Mark converged
        tmp_db.mark_converged(eid)

        # Finalize
        tmp_db.finalize_experiment(eid)

        summary = tmp_db.get_experiment_summary(eid)
        assert summary is not None
        assert summary["total_rounds"] == 5
        assert summary["best_performance"] == 1.0
        assert summary["converged"] == 1  # SQLite stores as int


# ── 3. Round recording ──────────────────────────────────────────

class TestRoundRecording:
    def test_record_all_fields(self, tmp_db):
        eid = tmp_db.create_experiment(domain_id="d1", node_id="n1")
        rid = tmp_db.record_round(
            experiment_id=eid,
            domain_id="d1",
            round_number=1,
            performance=-3.5,
            best_performance=-3.5,
            performance_delta=0.5,
            is_improvement=True,
            parameters={"x": [1.0, 2.0], "y": 3.0},
            best_parameters={"x": [1.0, 2.0], "y": 3.0},
            wall_clock_seconds=0.42,
            elapsed_seconds=0.42,
            time_to_current_best_seconds=0.42,
            time_to_target_seconds=None,
            chain_height=100,
            peers_count=5,
            block_reward_earned=12.5,
            converged=False,
        )
        assert len(rid) == 32

        rounds = tmp_db.get_rounds(eid)
        assert len(rounds) == 1
        r = rounds[0]
        assert r["performance"] == -3.5
        assert r["chain_height"] == 100
        assert r["peers_count"] == 5
        assert json.loads(r["parameters"])["y"] == 3.0

    def test_multiple_rounds(self, tmp_db):
        eid = tmp_db.create_experiment(domain_id="d1", node_id="n1")
        for i in range(10):
            tmp_db.record_round(
                experiment_id=eid, domain_id="d1",
                round_number=i + 1, performance=float(i),
            )
        rounds = tmp_db.get_rounds(eid)
        assert len(rounds) == 10
        assert rounds[0]["round_number"] == 1
        assert rounds[9]["round_number"] == 10

    def test_limit(self, tmp_db):
        eid = tmp_db.create_experiment(domain_id="d1", node_id="n1")
        for i in range(20):
            tmp_db.record_round(
                experiment_id=eid, domain_id="d1",
                round_number=i + 1, performance=float(i),
            )
        rounds = tmp_db.get_rounds(eid, limit=5)
        assert len(rounds) == 5


# ── 4. Summary generation ───────────────────────────────────────

class TestSummaryGeneration:
    def test_finalize_creates_summary(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="d1", node_id="n1", hostname="h1",
        )
        for i in range(1, 4):
            tmp_db.record_round(
                experiment_id=eid, domain_id="d1",
                round_number=i, performance=float(i),
                elapsed_seconds=float(i) * 10,
            )
        tmp_db.finalize_experiment(eid)

        s = tmp_db.get_experiment_summary(eid)
        assert s["total_rounds"] == 3
        assert s["final_performance"] == 3.0
        assert s["best_performance"] == 3.0
        assert s["node_id"] == "n1"
        assert s["hostname"] == "h1"

    def test_get_all_summaries(self, tmp_db):
        for dom in ["a", "b", "c"]:
            eid = tmp_db.create_experiment(domain_id=dom, node_id="n1")
            tmp_db.record_round(
                experiment_id=eid, domain_id=dom,
                round_number=1, performance=1.0,
            )
            tmp_db.finalize_experiment(eid)
        summaries = tmp_db.get_all_summaries()
        assert len(summaries) == 3

    def test_summary_with_convergence(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="d1", node_id="n1", target_performance=5.0,
        )
        for i in range(1, 8):
            converged = i >= 5
            tmp_db.record_round(
                experiment_id=eid, domain_id="d1",
                round_number=i, performance=float(i),
                converged=converged,
                time_to_target_seconds=float(i) * 10 if converged else None,
            )
        tmp_db.mark_converged(eid)
        tmp_db.finalize_experiment(eid)

        s = tmp_db.get_experiment_summary(eid)
        assert s["converged"] == 1
        assert s["rounds_to_convergence"] == 5


# ── 5. Thread safety ────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes(self, tmp_db):
        eid = tmp_db.create_experiment(domain_id="d1", node_id="n1")
        errors = []

        def writer(start_round: int):
            try:
                for i in range(start_round, start_round + 50):
                    tmp_db.record_round(
                        experiment_id=eid, domain_id="d1",
                        round_number=i, performance=float(i),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 50 + 1,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        rounds = tmp_db.get_rounds(eid, limit=10000)
        assert len(rounds) == 200


# ── 6. Convergence timing ───────────────────────────────────────

class TestConvergenceTiming:
    def test_time_to_target_recorded(self, tmp_db):
        eid = tmp_db.create_experiment(
            domain_id="d1", node_id="n1", target_performance=5.0,
        )
        # Not converged yet
        tmp_db.record_round(
            experiment_id=eid, domain_id="d1",
            round_number=1, performance=3.0,
            elapsed_seconds=10.0,
        )
        # Now converged
        tmp_db.record_round(
            experiment_id=eid, domain_id="d1",
            round_number=2, performance=6.0,
            elapsed_seconds=25.0,
            converged=True,
            time_to_target_seconds=25.0,
        )
        tmp_db.finalize_experiment(eid)

        s = tmp_db.get_experiment_summary(eid)
        assert s["time_to_target_seconds"] == 25.0


# ── 7. Export / query methods ────────────────────────────────────

class TestQueryMethods:
    def test_get_all_experiments(self, tmp_db):
        for i in range(3):
            tmp_db.create_experiment(domain_id=f"d{i}", node_id="n1")
        exps = tmp_db.get_all_experiments()
        assert len(exps) == 3

    def test_get_experiment_summary_not_found(self, tmp_db):
        assert tmp_db.get_experiment_summary("nonexistent") is None

    def test_get_rounds_empty(self, tmp_db):
        assert tmp_db.get_rounds("nonexistent") == []

    def test_db_path_property(self, tmp_db):
        assert tmp_db.db_path.endswith("test_olap.db")


# ── 8. Integration with ExperimentTracker ────────────────────────

class TestTrackerIntegration:
    def test_dual_write_csv_and_olap(self, tmp_tracker):
        tmp_tracker.start_experiment(
            "quad",
            optimization_config={"n": 10},
            param_bounds={"x": [-5, 5]},
            target_performance=0.9,
            optimizer_plugin="bayes",
        )
        tmp_tracker.record_round(
            "quad", performance=0.5, parameters={"x": [1.0]},
            wall_clock_seconds=0.1, chain_height=10, peers_count=2,
        )
        tmp_tracker.record_round(
            "quad", performance=0.95, parameters={"x": [2.0]},
            wall_clock_seconds=0.2, chain_height=11, peers_count=3,
        )

        # CSV should have data
        assert tmp_tracker._csv_path.stat().st_size > 0

        # OLAP should have data
        rounds = tmp_tracker.get_olap_rounds(
            tmp_tracker._experiments["quad"].experiment_id
        )
        assert rounds is not None
        assert len(rounds) == 2
        assert rounds[0]["performance"] == 0.5
        assert rounds[1]["performance"] == 0.95

    def test_finalize_writes_olap_summary(self, tmp_tracker):
        eid = tmp_tracker.start_experiment("d1", target_performance=10.0)
        tmp_tracker.record_round("d1", performance=5.0, parameters={"x": [1]})
        tmp_tracker.finalize()

        summary = tmp_tracker.get_olap_summary(eid)
        assert summary is not None
        assert summary["total_rounds"] == 1

    def test_mark_converged_updates_olap(self, tmp_tracker):
        eid = tmp_tracker.start_experiment("d1", target_performance=0.5)
        tmp_tracker.record_round("d1", performance=0.6, parameters={"x": [1]})
        tmp_tracker.mark_converged("d1")
        tmp_tracker.finalize()

        summary = tmp_tracker.get_olap_summary(eid)
        assert summary is not None
        assert summary["converged"] == 1

    def test_olap_disabled_returns_none(self, tmp_path):
        csv_only = ExperimentTracker(tmp_path / "stats.csv", node_id="n1")
        csv_only.start_experiment("d1")
        csv_only.record_round("d1", performance=1.0)
        assert csv_only.get_olap_summary() is None
        assert csv_only.get_olap_rounds("whatever") is None

    def test_get_olap_summary_all(self, tmp_tracker):
        tmp_tracker.start_experiment("a")
        tmp_tracker.record_round("a", performance=1.0)
        tmp_tracker.start_experiment("b")
        tmp_tracker.record_round("b", performance=2.0)
        tmp_tracker.finalize()

        all_summaries = tmp_tracker.get_olap_summary()
        assert isinstance(all_summaries, list)
        assert len(all_summaries) == 2
