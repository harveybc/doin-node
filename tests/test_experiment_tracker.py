"""Tests for the experiment stats tracker."""

import csv
import json
import threading
from pathlib import Path

import pytest

from doin_node.stats.experiment_tracker import ExperimentTracker, COLUMNS


@pytest.fixture
def tracker(tmp_path):
    csv_path = tmp_path / "stats.csv"
    t = ExperimentTracker(csv_path, node_id="test-node-123", doin_version="0.1.0")
    t.start_experiment(
        "quadratic",
        optimization_config={"n_params": 10},
        param_bounds={"x": [-20.0, 20.0]},
        target_performance=-1.0,
        optimizer_plugin="simple_quadratic",
    )
    return t


@pytest.fixture
def csv_path(tracker):
    return tracker._csv_path


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


class TestCSVCreation:
    def test_creates_csv_with_headers(self, csv_path):
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == COLUMNS

    def test_records_appended(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-5.0, parameters={"x": [1.0]})
        tracker.record_round("quadratic", performance=-3.0, parameters={"x": [2.0]})
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert float(rows[0]["performance"]) == -5.0
        assert float(rows[1]["performance"]) == -3.0

    def test_round_numbers_increment(self, tracker, csv_path):
        for i in range(5):
            tracker.record_round("quadratic", performance=float(-10 + i))
        rows = _read_csv(csv_path)
        assert [r["round_number"] for r in rows] == ["1", "2", "3", "4", "5"]


class TestPerformanceTracking:
    def test_best_performance_tracked(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-5.0)
        tracker.record_round("quadratic", performance=-3.0)
        tracker.record_round("quadratic", performance=-4.0)  # worse
        rows = _read_csv(csv_path)
        assert float(rows[2]["best_performance"]) == -3.0

    def test_performance_delta(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-5.0)
        tracker.record_round("quadratic", performance=-3.0)  # improvement of 2.0
        tracker.record_round("quadratic", performance=-4.0)  # no improvement
        rows = _read_csv(csv_path)
        assert float(rows[1]["performance_delta"]) == 2.0
        assert float(rows[2]["performance_delta"]) == 0.0


class TestConvergence:
    def test_time_to_target_on_convergence(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-5.0)
        tracker.record_round("quadratic", performance=-0.5)  # >= -1.0 target
        rows = _read_csv(csv_path)
        assert rows[0]["time_to_target_seconds"] == ""  # not reached yet
        assert rows[1]["converged"] == "True"
        assert float(rows[1]["time_to_target_seconds"]) >= 0

    def test_mark_converged(self, tracker):
        tracker.record_round("quadratic", performance=-5.0)
        tracker.mark_converged("quadratic")
        summary = tracker.get_summary("quadratic")
        assert summary["converged"] is True
        assert summary["time_to_target_seconds"] is not None


class TestSummary:
    def test_summary_json_on_finalize(self, tracker):
        tracker.record_round("quadratic", performance=-3.0)
        tracker.finalize()
        summary_path = tracker._summary_path
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert "quadratic" in data
        assert data["quadratic"]["rounds"] == 1

    def test_get_summary(self, tracker):
        tracker.record_round("quadratic", performance=-2.0)
        s = tracker.get_summary("quadratic")
        assert s["rounds"] == 1
        assert s["best_performance"] == -2.0


class TestThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        csv_path = tmp_path / "concurrent.csv"
        tracker = ExperimentTracker(csv_path, node_id="thread-test")
        tracker.start_experiment("d1")
        tracker.start_experiment("d2")

        errors = []

        def write_rounds(domain, n):
            try:
                for i in range(n):
                    tracker.record_round(domain, performance=float(i))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=write_rounds, args=("d1", 50))
        t2 = threading.Thread(target=write_rounds, args=("d2", 50))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        rows = _read_csv(csv_path)
        assert len(rows) == 100


class TestJSONFields:
    def test_parameters_stored_as_json(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-3.0, parameters={"x": [1.0, 2.0]})
        rows = _read_csv(csv_path)
        params = json.loads(rows[0]["parameters"])
        assert params == {"x": [1.0, 2.0]}

    def test_config_stored_as_json(self, tracker, csv_path):
        tracker.record_round("quadratic", performance=-3.0)
        rows = _read_csv(csv_path)
        config = json.loads(rows[0]["optimization_config"])
        assert config == {"n_params": 10}


class TestAutoStart:
    def test_auto_starts_experiment(self, tmp_path):
        csv_path = tmp_path / "auto.csv"
        tracker = ExperimentTracker(csv_path, node_id="auto")
        # No explicit start_experiment call
        tracker.record_round("unknown_domain", performance=1.0)
        rows = _read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["domain_id"] == "unknown_domain"
