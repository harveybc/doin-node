"""Dashboard API routes — register on the existing aiohttp app.

Provides real-time monitoring of:
  - Node status, peers, plugin capabilities
  - Optimization progress (rounds, stages, epochs)
  - Evaluation requests (pending, served)
  - Chain activity
  - Training metrics (champion + current candidate)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(obj):
    return json.dumps(obj, default=_json_default)
from pathlib import Path
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Shared mutable state for live training progress (updated by callbacks)
_training_state: dict[str, Any] = {
    "active": False,
    "domain_id": "",
    "round": 0,
    "stage": 0,
    "total_stages": 0,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": None,
    "val_loss": None,
    "train_mae": None,
    "val_mae": None,
    "started_at": 0,
    "candidate_params": None,
}


def get_training_state() -> dict[str, Any]:
    """Public accessor for training state (used by optimizer callback)."""
    return _training_state


def update_training_state(**kwargs: Any) -> None:
    """Update training state from optimizer/training callback."""
    _training_state.update(kwargs)


def setup_dashboard(app: web.Application, node: Any) -> None:
    """Register dashboard routes on *app*."""
    app["_doin_node"] = node
    # Pages
    app.router.add_get("/dashboard", _dashboard_page)
    # API endpoints
    app.router.add_get("/api/node", _api_node)
    app.router.add_get("/api/peers", _api_peers)
    app.router.add_get("/api/optimization", _api_optimization)
    app.router.add_get("/api/training", _api_training)
    app.router.add_get("/api/evaluations", _api_evaluations)
    app.router.add_get("/api/metrics", _api_metrics)
    app.router.add_get("/api/plugins", _api_plugins)
    app.router.add_get("/api/chain", _api_chain)
    app.router.add_get("/api/events", _api_events)
    logger.info("Dashboard enabled at /dashboard")


async def _dashboard_page(request: web.Request) -> web.Response:
    html = (_TEMPLATE_DIR / "dashboard.html").read_text()
    return web.Response(text=html, content_type="text/html")


# ── Node Info ────────────────────────────────────────────────

async def _api_node(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    uptime = time.time() - node._start_time
    domains = []
    for did, dr in node._domain_roles.items():
        domains.append({
            "domain_id": dr.domain_id,
            "optimize": dr.optimize,
            "evaluate": dr.evaluate,
            "synthetic": did in node._synthetic_plugins,
        })
    return web.json_response({
        "peer_id": node.identity.peer_id if node.identity else "unknown",
        "port": node.config.port,
        "uptime_s": round(uptime, 1),
        "domains": domains,
        "chain_height": node.chaindb.height if node.chaindb else 0,
        "discovery_enabled": node.config.discovery_enabled,
        "dashboard_enabled": node.config.dashboard_enabled,
    }, dumps=_dumps)


# ── Peers ────────────────────────────────────────────────────

async def _api_peers(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    peers = []
    for ep, p in node._peers.items():
        peers.append({
            "endpoint": ep,
            "peer_id": p.peer_id,
            "connected": True,
        })
    if node.discovery:
        for ep, dp in node.discovery._known_peers.items():
            if ep not in node._peers:
                peers.append({
                    "endpoint": ep,
                    "peer_id": dp.peer_id,
                    "connected": False,
                    "source": dp.source,
                    "last_seen": round(dp.last_seen, 1),
                    "failures": dp.connection_failures,
                    "domains": dp.domains,
                })
    return web.json_response({
        "connected_count": len(node._peers),
        "known_count": node.discovery.known_count if node.discovery else 0,
        "peers": peers,
    }, dumps=_dumps)


# ── Optimization Status ─────────────────────────────────────

async def _api_optimization(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    domains = []
    for domain_id, dr in node._domain_roles.items():
        info: dict[str, Any] = {
            "domain_id": domain_id,
            "round": node._domain_round_count.get(domain_id, 0),
            "best_performance": None,
            "best_params": None,
            "target_performance": None,
            "converged": domain_id in node._domain_converged,
        }
        best = node._domain_best.get(domain_id, (None, None))
        if best[1] is not None:
            info["best_performance"] = best[1]
            # Don't send full params (can be >1MB with model weights)
            info["best_params"] = {"_param_count": len(best[0]) if isinstance(best[0], dict) else None}
        for dcfg in node.config.domains:
            did = getattr(dcfg, "domain_id", None) or getattr(dcfg, "id", None)
            if did == domain_id:
                info["target_performance"] = getattr(dcfg, "target_performance", None)
                break
        # Champion detailed metrics (MAE breakdowns)
        champion = node._domain_champion_metrics.get(domain_id)
        if champion:
            info["champion"] = {
                "round": champion.get("round"),
                "train_mae": champion.get("train_mae"),
                "train_naive_mae": champion.get("train_naive_mae"),
                "val_mae": champion.get("val_mae"),
                "val_naive_mae": champion.get("val_naive_mae"),
                "test_mae": champion.get("test_mae"),
                "test_naive_mae": champion.get("test_naive_mae"),
            }
        domains.append(info)

    # Tolerance config
    tolerance = None
    for dcfg in node.config.domains:
        ic = getattr(dcfg, "incentive_config", None)
        if ic:
            tolerance = getattr(ic, "tolerance_margin", None)
            break

    # higher_is_better from first domain config
    hib = False
    for dcfg in node.config.domains:
        hib = getattr(dcfg, "higher_is_better", False)
        break

    return web.json_response({"domains": domains, "tolerance_margin": tolerance, "higher_is_better": hib}, dumps=_dumps)


# ── Live Training State ──────────────────────────────────────

async def _api_training(request: web.Request) -> web.Response:
    return web.json_response(_training_state, dumps=_dumps)


# ── Evaluation Requests ──────────────────────────────────────

async def _api_evaluations(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "50"))

    pending = []
    completed = []
    for tid, task in list(node.task_queue.tasks.items())[:limit]:
        entry = {
            "task_id": tid[:12],
            "domain_id": getattr(task, "domain_id", ""),
            "status": getattr(task, "status", "unknown"),
            "submitter": str(getattr(task, "submitter", ""))[:12],
            "created_at": getattr(task, "created_at", 0),
            "claimed_by": str(getattr(task, "claimed_by", ""))[:12],
            "performance": getattr(task, "verified_performance", None),
        }
        if entry["status"] in ("pending", "claimed"):
            pending.append(entry)
        else:
            completed.append(entry)

    return web.json_response({
        "pending_count": len(pending),
        "completed_count": len(completed),
        "pending": pending,
        "completed": completed[-limit:],
    }, dumps=_dumps)


# ── Metrics Time Series ─────────────────────────────────────

async def _api_metrics(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "500"))
    metrics: list[dict] = []

    # OLAP DB — use tracker's existing connection to see WAL data
    tracker = getattr(node, "experiment_tracker", None)
    olap = getattr(tracker, "_olap", None) if tracker else None
    olap_conn = getattr(olap, "_conn", None) if olap else None
    if olap_conn:
        try:
            cur = olap_conn.cursor()
            # Get current experiment_id if available
            exp_id = None
            if tracker and hasattr(tracker, "_experiments"):
                for de in tracker._experiments.values():
                    exp_id = getattr(de, "experiment_id", None)
                    break
            if exp_id:
                cur.execute(
                    "SELECT round_number as round_num, performance, "
                    "best_performance, timestamp_utc as timestamp, "
                    "domain_id, is_improvement "
                    "FROM fact_round WHERE experiment_id = ? "
                    "ORDER BY round_number ASC LIMIT ?",
                    (exp_id, limit))
            else:
                # Fallback: all data from latest experiment
                cur.execute(
                    "SELECT round_number as round_num, performance, "
                    "best_performance, timestamp_utc as timestamp, "
                    "domain_id, is_improvement "
                    "FROM fact_round WHERE experiment_id = ("
                    "SELECT experiment_id FROM fact_round ORDER BY rowid DESC LIMIT 1"
                    ") ORDER BY round_number ASC LIMIT ?",
                    (limit,))
            for row in cur.fetchall():
                metrics.append(dict(row))
        except Exception as e:
            import traceback
            print(f"[OLAP ERROR] {e}\n{traceback.format_exc()}", flush=True)

    # No CSV fallback — OLAP is the single source of truth
    return web.json_response({"metrics": metrics, "count": len(metrics)}, dumps=_dumps)


# ── Plugins ──────────────────────────────────────────────────

async def _api_plugins(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    plugins = []
    for domain_id in node._domain_roles:
        entry: dict[str, Any] = {"domain_id": domain_id}
        for role, store in [
            ("optimizer", node._optimizer_plugins),
            ("evaluator", node._evaluator_plugins),
            ("synthetic", node._synthetic_plugins),
        ]:
            p = store.get(domain_id)
            entry[role] = {"name": type(p).__name__, "module": type(p).__module__} if p else None
        plugins.append(entry)
    return web.json_response({"plugins": plugins}, dumps=_dumps)


# ── Chain ────────────────────────────────────────────────────

async def _api_chain(request: web.Request) -> web.Response:
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "30"))
    blocks = []
    if node.chaindb:
        height = node.chaindb.height
        chain_blocks = node.chaindb.get_blocks_range(max(0, height - limit), height)
        for block in chain_blocks:
            b = block if isinstance(block, dict) else (block.__dict__ if hasattr(block, "__dict__") else {})
            blocks.append({
                "index": b.get("index", b.get("block_index", 0)),
                "hash": str(b.get("hash", b.get("block_hash", "")))[:16],
                "type": b.get("payload_type", b.get("type", "")),
                "timestamp": b.get("timestamp", 0),
                "submitter": str(b.get("submitter", b.get("peer_id", "")))[:12],
            })
    return web.json_response({
        "height": node.chaindb.height if node.chaindb else 0,
        "blocks": blocks,
    }, dumps=_dumps)


# ── Optimizer Events Log ─────────────────────────────────────

async def _api_events(request: web.Request) -> web.Response:
    """Return live event log — all events captured in memory (most recent first)."""
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "200"))

    events = list(reversed(node._live_events[-limit:]))
    return web.json_response({"events": events, "count": len(events)}, dumps=_dumps)
