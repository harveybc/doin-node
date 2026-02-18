"""Comprehensive experiment tracking for OLAP cube analysis.

Records every optimization round as a flat, denormalized CSV row.
Complex fields (parameters, config, bounds) are stored as JSON strings.
Append-only — never rewrites the CSV file.
Thread-safe via a lock (the optimizer loop is async but plugin calls
are dispatched to a thread executor).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical column order — OLAP-friendly flat schema
COLUMNS = [
    "experiment_id",
    "round_id",
    "domain_id",
    "node_id",
    "timestamp_utc",
    "wall_clock_seconds",
    # Timing
    "experiment_start_utc",
    "time_to_current_best_seconds",
    "time_to_target_seconds",
    "total_rounds",
    "round_number",
    # Performance
    "performance",
    "best_performance",
    "target_performance",
    "performance_delta",
    "converged",
    # Parameters (JSON strings)
    "parameters",
    "best_parameters",
    # Config (JSON strings, denormalized per row)
    "optimization_config",
    "param_bounds",
    # Node / environment
    "hostname",
    "optimizer_plugin",
    "doin_version",
    # Network context
    "chain_height",
    "peers_count",
    "block_reward_earned",
]


class ExperimentTracker:
    """Records every optimization round to a CSV file for OLAP analysis.

    Usage::

        tracker = ExperimentTracker("experiment_stats.csv", node_id="abc123")
        tracker.start_experiment("quadratic", config={...}, bounds={...}, ...)

        # After each round:
        tracker.record_round(
            domain_id="quadratic",
            performance=-3.5,
            parameters={"x": [1.2, 3.4]},
            wall_clock_seconds=0.42,
            chain_height=5,
            peers_count=1,
        )

        # When target is reached:
        tracker.mark_converged("quadratic")

        # On shutdown:
        tracker.finalize()
    """

    def __init__(
        self,
        csv_path: str | Path,
        node_id: str = "",
        doin_version: str = "0.1.0",
        olap_db_path: str | Path | None = None,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._summary_path = Path(f"{csv_path}.summary.json")
        self._node_id = node_id
        self._doin_version = doin_version
        self._hostname = platform.node()
        self._lock = threading.Lock()

        # Per-domain experiment state
        self._experiments: dict[str, _DomainExperiment] = {}

        # OLAP database (optional — when provided, dual-writes CSV + SQLite)
        self._olap: Any | None = None
        if olap_db_path:
            from doin_node.stats.olap_db import OLAPDatabase
            self._olap = OLAPDatabase(olap_db_path)

        # Ensure parent directory exists
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV header if file doesn't exist or is empty
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            self._write_header()

    # ── Public API ───────────────────────────────────────────────

    def start_experiment(
        self,
        domain_id: str,
        *,
        optimization_config: dict[str, Any] | None = None,
        param_bounds: dict[str, Any] | None = None,
        target_performance: float | None = None,
        optimizer_plugin: str = "",
        experiment_id: str | None = None,
    ) -> str:
        """Start tracking a new experiment for a domain. Returns experiment_id."""
        exp_id = experiment_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        with self._lock:
            self._experiments[domain_id] = _DomainExperiment(
                experiment_id=exp_id,
                start_utc=now,
                start_mono=time.monotonic(),
                optimization_config=optimization_config or {},
                param_bounds=param_bounds or {},
                target_performance=target_performance,
                optimizer_plugin=optimizer_plugin,
            )
        # Write to OLAP database
        if self._olap:
            try:
                self._olap.create_experiment(
                    domain_id=domain_id,
                    node_id=self._node_id,
                    hostname=self._hostname,
                    optimizer_plugin=optimizer_plugin,
                    optimization_config=optimization_config,
                    param_bounds=param_bounds,
                    target_performance=target_performance,
                    doin_version=self._doin_version,
                    experiment_id=exp_id,
                )
            except Exception:
                logger.exception("OLAP: failed to create experiment")

        logger.info("Experiment started: domain=%s id=%s", domain_id, exp_id[:12])
        return exp_id

    def record_round(
        self,
        domain_id: str,
        *,
        performance: float,
        parameters: dict[str, Any] | None = None,
        wall_clock_seconds: float = 0.0,
        chain_height: int = 0,
        peers_count: int = 0,
        block_reward_earned: float = 0.0,
        detail_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one optimization round. Returns the row dict."""
        with self._lock:
            exp = self._experiments.get(domain_id)
            if exp is None:
                # Auto-start experiment if not explicitly started
                self._experiments[domain_id] = _DomainExperiment(
                    experiment_id=uuid.uuid4().hex,
                    start_utc=datetime.now(timezone.utc),
                    start_mono=time.monotonic(),
                )
                exp = self._experiments[domain_id]

            exp.round_count += 1
            now_utc = datetime.now(timezone.utc)
            now_mono = time.monotonic()

            # Performance tracking
            improved = False
            perf_delta = 0.0
            if exp.best_performance is None or performance > exp.best_performance:
                perf_delta = performance - (exp.best_performance or 0.0)
                if exp.best_performance is not None:
                    perf_delta = performance - exp.best_performance
                exp.best_performance = performance
                exp.best_parameters = parameters
                exp.time_to_best_mono = now_mono
                improved = True

            # Check convergence
            converged = exp.converged
            if (
                not converged
                and exp.target_performance is not None
                and performance >= exp.target_performance
            ):
                exp.converged = True
                exp.time_to_target_seconds = now_mono - exp.start_mono
                converged = True
                # Update OLAP dimension
                if self._olap:
                    try:
                        self._olap.mark_converged(exp.experiment_id)
                    except Exception:
                        logger.exception("OLAP: failed to mark converged in record_round")

            time_to_best = (
                (exp.time_to_best_mono - exp.start_mono)
                if exp.time_to_best_mono is not None
                else 0.0
            )

            row = {
                "experiment_id": exp.experiment_id,
                "round_id": uuid.uuid4().hex,
                "domain_id": domain_id,
                "node_id": self._node_id,
                "timestamp_utc": now_utc.isoformat(),
                "wall_clock_seconds": round(wall_clock_seconds, 6),
                "experiment_start_utc": exp.start_utc.isoformat(),
                "time_to_current_best_seconds": round(time_to_best, 6),
                "time_to_target_seconds": (
                    round(exp.time_to_target_seconds, 6)
                    if exp.time_to_target_seconds is not None
                    else ""
                ),
                "total_rounds": exp.round_count,
                "round_number": exp.round_count,
                "performance": performance,
                "best_performance": exp.best_performance if exp.best_performance is not None else "",
                "target_performance": exp.target_performance if exp.target_performance is not None else "",
                "performance_delta": round(perf_delta, 6) if improved else 0.0,
                "converged": converged,
                "parameters": json.dumps(parameters) if parameters else "{}",
                "best_parameters": json.dumps(exp.best_parameters) if exp.best_parameters else "{}",
                "optimization_config": json.dumps(exp.optimization_config),
                "param_bounds": json.dumps(exp.param_bounds),
                "hostname": self._hostname,
                "optimizer_plugin": exp.optimizer_plugin,
                "doin_version": self._doin_version,
                "chain_height": chain_height,
                "peers_count": peers_count,
                "block_reward_earned": block_reward_earned,
            }

            # Add detailed metrics (MAE breakdowns) if provided
            dm = detail_metrics or {}
            for key in ("train_mae", "train_naive_mae", "val_mae", "val_naive_mae",
                        "test_mae", "test_naive_mae"):
                row[key] = dm.get(key, "")

            self._append_row(row)

            # Dual-write to OLAP
            if self._olap:
                try:
                    elapsed = time.monotonic() - exp.start_mono
                    self._olap.record_round(
                        experiment_id=exp.experiment_id,
                        domain_id=domain_id,
                        round_number=exp.round_count,
                        performance=performance,
                        best_performance=exp.best_performance,
                        performance_delta=perf_delta if improved else 0.0,
                        is_improvement=improved,
                        parameters=parameters,
                        best_parameters=exp.best_parameters,
                        wall_clock_seconds=wall_clock_seconds,
                        elapsed_seconds=elapsed,
                        time_to_current_best_seconds=time_to_best,
                        time_to_target_seconds=exp.time_to_target_seconds,
                        chain_height=chain_height,
                        peers_count=peers_count,
                        block_reward_earned=block_reward_earned,
                        converged=converged,
                        round_id=row["round_id"],
                        detail_metrics=detail_metrics,
                    )
                except Exception:
                    logger.exception("OLAP: failed to record round")

            return row

    def mark_converged(self, domain_id: str) -> None:
        """Explicitly mark a domain as converged (finalizes time_to_target)."""
        with self._lock:
            exp = self._experiments.get(domain_id)
            if exp and not exp.converged:
                exp.converged = True
                exp.time_to_target_seconds = time.monotonic() - exp.start_mono
                if self._olap:
                    try:
                        self._olap.mark_converged(exp.experiment_id)
                    except Exception:
                        logger.exception("OLAP: failed to mark converged")

    def get_summary(self, domain_id: str | None = None) -> dict[str, Any]:
        """Get current experiment summary (for HTTP endpoint)."""
        with self._lock:
            if domain_id:
                exp = self._experiments.get(domain_id)
                if exp is None:
                    return {}
                return self._exp_summary(domain_id, exp)
            return {
                d: self._exp_summary(d, e)
                for d, e in self._experiments.items()
            }

    def get_experiment_state(self, domain_id: str) -> dict[str, Any] | None:
        """Return current experiment state for a domain (for on-chain metrics).

        Returns a dict with experiment_id, round_count, start_mono,
        best_performance, optimization_config, optimizer_plugin — or None
        if no experiment is tracked for *domain_id*.
        """
        with self._lock:
            exp = self._experiments.get(domain_id)
            if exp is None:
                return None
            return {
                "experiment_id": exp.experiment_id,
                "round_count": exp.round_count,
                "start_mono": exp.start_mono,
                "best_performance": exp.best_performance,
                "optimization_config": exp.optimization_config,
                "optimizer_plugin": exp.optimizer_plugin,
            }

    def finalize(self) -> None:
        """Write summary JSON file. Call on node shutdown."""
        summary = self.get_summary()
        summary["finalized_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Experiment summary written: %s", self._summary_path)
        except Exception:
            logger.exception("Failed to write experiment summary")

        # Finalize OLAP experiments
        if self._olap:
            for exp in self._experiments.values():
                try:
                    self._olap.finalize_experiment(exp.experiment_id)
                except Exception:
                    logger.exception("OLAP: failed to finalize %s", exp.experiment_id[:12])

    # ── OLAP query helpers ───────────────────────────────────────

    def get_olap_summary(self, experiment_id: str | None = None) -> Any:
        """Return OLAP summary data (requires olap_db_path)."""
        if not self._olap:
            return None
        if experiment_id:
            return self._olap.get_experiment_summary(experiment_id)
        return self._olap.get_all_summaries()

    def get_olap_rounds(self, experiment_id: str, limit: int = 1000) -> list[dict[str, Any]] | None:
        """Return OLAP round data (requires olap_db_path)."""
        if not self._olap:
            return None
        return self._olap.get_rounds(experiment_id, limit)

    # ── Internal ─────────────────────────────────────────────────

    def _write_header(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()

    def _append_row(self, row: dict[str, Any]) -> None:
        """Append a single row to the CSV. Caller must hold _lock."""
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)

    def _exp_summary(self, domain_id: str, exp: _DomainExperiment) -> dict[str, Any]:
        elapsed = time.monotonic() - exp.start_mono
        return {
            "experiment_id": exp.experiment_id,
            "domain_id": domain_id,
            "rounds": exp.round_count,
            "best_performance": exp.best_performance,
            "target_performance": exp.target_performance,
            "converged": exp.converged,
            "time_to_target_seconds": exp.time_to_target_seconds,
            "elapsed_seconds": round(elapsed, 2),
            "optimizer_plugin": exp.optimizer_plugin,
        }


class _DomainExperiment:
    """Mutable state for one domain's experiment run."""

    __slots__ = (
        "experiment_id",
        "start_utc",
        "start_mono",
        "optimization_config",
        "param_bounds",
        "target_performance",
        "optimizer_plugin",
        "round_count",
        "best_performance",
        "best_parameters",
        "time_to_best_mono",
        "converged",
        "time_to_target_seconds",
    )

    def __init__(
        self,
        experiment_id: str = "",
        start_utc: datetime | None = None,
        start_mono: float = 0.0,
        optimization_config: dict[str, Any] | None = None,
        param_bounds: dict[str, Any] | None = None,
        target_performance: float | None = None,
        optimizer_plugin: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.start_utc = start_utc or datetime.now(timezone.utc)
        self.start_mono = start_mono or time.monotonic()
        self.optimization_config = optimization_config or {}
        self.param_bounds = param_bounds or {}
        self.target_performance = target_performance
        self.optimizer_plugin = optimizer_plugin
        self.round_count = 0
        self.best_performance: float | None = None
        self.best_parameters: dict[str, Any] | None = None
        self.time_to_best_mono: float | None = None
        self.converged = False
        self.time_to_target_seconds: float | None = None
