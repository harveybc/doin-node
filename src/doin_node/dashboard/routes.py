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
import subprocess
import time
from datetime import datetime


import math


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_for_json(obj):
    """Recursively replace Infinity/NaN floats with None for valid JSON."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _dumps(obj):
    return json.dumps(_sanitize_for_json(obj), default=_json_default)
from pathlib import Path
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ── Package versions (computed once at import time) ──────────

def _get_git_short_hash(package_path: str) -> str:
    """Return the 7-char git short hash for a repo, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=package_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_versions() -> dict[str, str]:
    """Compute git short hashes for all DOIN packages.

    Strategy:
      1. Try importlib to find the package source (works for editable installs).
      2. Try well-known repo paths (~/Documents/GitHub/<name>, ~/doin-plugins, etc.).
      3. Fall back to pip metadata version string.
    """
    import importlib.metadata
    import importlib.util

    versions = {}
    # label → (import_name, pip_dist_name, candidate_dirs)
    pkg_map = {
        "doin-core": ("doin_core", "doin-core", [
            Path.home() / "Documents" / "GitHub" / "doin-core",
            Path.home() / "doin-core",
        ]),
        "doin-node": ("doin_node", "doin-node", [
            Path.home() / "Documents" / "GitHub" / "doin-node",
            Path.home() / "doin-node",
        ]),
        "doin-plugins": ("doin_plugins", "doin-plugins", [
            Path.home() / "doin-plugins",
            Path.home() / "Documents" / "GitHub" / "doin-plugins",
        ]),
        "predictor": ("predictor", "predictor", [
            Path.home() / "Documents" / "GitHub" / "predictor",
            Path.home() / "predictor",
        ]),
    }
    for label, (import_name, dist_name, candidates) in pkg_map.items():
        found = False
        # 1. importlib spec → walk up to .git
        try:
            spec = importlib.util.find_spec(import_name)
            if spec and spec.origin:
                pkg_dir = Path(spec.origin).resolve().parent
                for parent in [pkg_dir] + list(pkg_dir.parents):
                    if (parent / ".git").exists():
                        versions[label] = _get_git_short_hash(str(parent))
                        found = True
                        break
        except Exception:
            pass
        if found:
            continue

        # 2. Try well-known repo directories
        for cand in candidates:
            if cand.exists() and (cand / ".git").exists():
                versions[label] = _get_git_short_hash(str(cand))
                found = True
                break
        if found:
            continue

        # 3. Fall back to pip metadata version
        try:
            ver = importlib.metadata.version(dist_name)
            versions[label] = f"v{ver}"
        except Exception:
            versions[label] = "?"

    return versions


_PACKAGE_VERSIONS: dict[str, str] = _compute_versions()

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
    app.router.add_get("/api/candidate", _api_candidate)
    app.router.add_get("/api/alerts", _api_alerts)
    app.router.add_post("/api/alerts/ack", _api_alerts_ack)
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
        oc = dr.optimization_config or {}
        domains.append({
            "domain_id": dr.domain_id,
            "optimize": dr.optimize,
            "evaluate": dr.evaluate,
            "synthetic": did in node._synthetic_plugins,
            "predictor_plugin": oc.get("predictor_plugin", ""),
            "optimizer_plugin": oc.get("optimizer_plugin", ""),
            "pipeline_plugin": oc.get("pipeline_plugin", ""),
            "preprocessor_plugin": oc.get("preprocessor_plugin", ""),
        })
    return web.json_response({
        "peer_id": node.identity.peer_id if node.identity else "unknown",
        "port": node.config.port,
        "uptime_s": round(uptime, 1),
        "domains": domains,
        "chain_height": node.chaindb.height if node.chaindb else 0,
        "discovery_enabled": node.config.discovery_enabled,
        "dashboard_enabled": node.config.dashboard_enabled,
        "versions": _PACKAGE_VERSIONS,
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

    # ── Derive champion data from blockchain (single source of truth) ──
    chain_champions: dict[str, dict[str, Any]] = {}
    if node.chaindb:
        height = node.chaindb.height
        for i in range(height):
            block = node.chaindb.get_block(i)
            if block is None:
                continue
            for tx in block.transactions:
                tx_type_val = tx.tx_type.value if hasattr(tx.tx_type, "value") else str(tx.tx_type)
                if tx_type_val == "optimae_accepted":
                    domain_id = tx.domain_id
                    verified = tx.payload.get("verified_performance")
                    parameters = tx.payload.get("parameters")
                    if verified is not None:
                        prev = chain_champions.get(domain_id)
                        # Determine if this is better
                        hib_domain = False
                        for dcfg in node.config.domains:
                            did = getattr(dcfg, "domain_id", None) or getattr(dcfg, "id", None)
                            if did == domain_id:
                                hib_domain = getattr(dcfg, "higher_is_better", False)
                                break
                        is_better = (
                            prev is None
                            or (hib_domain and verified > prev["performance"])
                            or (not hib_domain and verified < prev["performance"])
                        )
                        if is_better:
                            chain_champions[domain_id] = {
                                "performance": verified,
                                "parameters": parameters,
                                "block_index": block.header.index,
                                "peer_id": tx.peer_id,
                                "val_mae": tx.payload.get("val_mae"),
                                "train_mae": tx.payload.get("train_mae"),
                                "val_naive_mae": tx.payload.get("val_naive_mae"),
                                "train_naive_mae": tx.payload.get("train_naive_mae"),
                                "test_mae": tx.payload.get("test_mae"),
                                "test_naive_mae": tx.payload.get("test_naive_mae"),
                                "effective_increment": tx.payload.get("effective_increment"),
                            }

    for domain_id, dr in node._domain_roles.items():
        # Derive generation/stage from chain shared-pop state (authoritative)
        _chain_gen = node._domain_round_count.get(domain_id, 0)
        _chain_stage = None
        _chain_stage_name = None
        _chain_no_improve = None
        pop_st = node._shared_pop_state.get(domain_id)
        if pop_st:
            _chain_gen = pop_st.get("generation", _chain_gen)
            _chain_stage = pop_st.get("stage_idx")
            _sched = pop_st.get("stage_schedule", [])
            if _chain_stage is not None and _sched and _chain_stage < len(_sched):
                _chain_stage_name = _sched[_chain_stage].get("name", "")
            _chain_no_improve = pop_st.get("no_improve_count")

        info: dict[str, Any] = {
            "domain_id": domain_id,
            "round": _chain_gen,
            "best_performance": None,
            "best_params": None,
            "target_performance": None,
            "converged": domain_id in node._domain_converged,
        }

        # Use chain-derived champion as primary source
        chain_champ = chain_champions.get(domain_id)
        if chain_champ:
            info["best_performance"] = chain_champ["performance"]
            params = chain_champ.get("parameters")
            info["best_params"] = {"_param_count": len(params) if isinstance(params, dict) else None}
            info["champion"] = {
                "source": "blockchain",
                "block_index": chain_champ.get("block_index"),
                "train_mae": chain_champ.get("train_mae"),
                "train_naive_mae": chain_champ.get("train_naive_mae"),
                "val_mae": chain_champ.get("val_mae"),
                "val_naive_mae": chain_champ.get("val_naive_mae"),
                "test_mae": chain_champ.get("test_mae"),
                "test_naive_mae": chain_champ.get("test_naive_mae"),
            }
        else:
            # Fallback to local state only if chain has no data yet
            best = node._domain_best.get(domain_id, (None, None))
            if best[1] is not None:
                info["best_performance"] = best[1]
                info["best_params"] = {"_param_count": len(best[0]) if isinstance(best[0], dict) else None}
            champion = node._domain_champion_metrics.get(domain_id)
            if champion:
                info["champion"] = {
                    "source": "local",
                    "round": champion.get("round"),
                    "train_mae": champion.get("train_mae"),
                    "train_naive_mae": champion.get("train_naive_mae"),
                    "val_mae": champion.get("val_mae"),
                    "val_naive_mae": champion.get("val_naive_mae"),
                    "test_mae": champion.get("test_mae"),
                    "test_naive_mae": champion.get("test_naive_mae"),
                }

        for dcfg in node.config.domains:
            did = getattr(dcfg, "domain_id", None) or getattr(dcfg, "id", None)
            if did == domain_id:
                info["target_performance"] = getattr(dcfg, "target_performance", None)
                break

        # Expose optimization config fields for dashboard defaults
        opt_cfg = dr.optimization_config if dr else {}
        stages = opt_cfg.get("optimization_stages", opt_cfg.get("staged_stages", []))
        # Derive n_generations from sum of per-stage generations, fallback to config
        _total_gens = sum(s.get("generations", 0) for s in stages) if stages else 0
        info["n_generations"] = _total_gens or opt_cfg.get("n_generations", 20)
        info["optimization_patience"] = opt_cfg.get("optimization_patience", 0)
        info["population_size"] = opt_cfg.get("population_size", 20)
        info["total_stages"] = len(stages) if stages else 1
        info["n_generations_stage"] = stages[0].get("generations", info["n_generations"]) if stages else info["n_generations"]
        info["metric_type"] = opt_cfg.get("metric_type", "regression")
        if stages:
            # Priority: chain shared-pop state > live candidate > fallback
            if _chain_stage is not None:
                info["current_stage"] = _chain_stage + 1
                info["current_stage_name"] = _chain_stage_name or ""
                if _chain_stage < len(stages):
                    info["n_generations_stage"] = stages[_chain_stage].get("generations", info["n_generations"])
            elif (cc := (node._current_candidate if hasattr(node, "_current_candidate") else {})) and cc.get("domain_id") == domain_id:
                info["current_stage"] = cc.get("stage", 1)
                info["current_stage_name"] = cc.get("stage_name", stages[0].get("name", "") if stages else "")
                info["n_generations_stage"] = cc.get("n_generations_stage", info["n_generations"])
            else:
                # Fallback: derive from round count
                cur_round = _chain_gen
                cumulative = 0
                cur_stage_idx = 0
                cur_stage_name = stages[0].get("name", "") if stages else ""
                for si, sd in enumerate(stages):
                    cumulative += sd.get("generations", 0)
                    if cur_round < cumulative:
                        cur_stage_idx = si
                        cur_stage_name = sd.get("name", "")
                        break
                else:
                    cur_stage_idx = len(stages) - 1
                    cur_stage_name = stages[-1].get("name", "") if stages else ""
                info["current_stage"] = cur_stage_idx + 1
                info["current_stage_name"] = cur_stage_name
                info["n_generations_stage"] = stages[cur_stage_idx].get("generations", info["n_generations"])

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
    """Build metrics from blockchain — identical on every synced node."""
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "500"))
    metrics: list[dict] = []

    # Determine higher_is_better per domain
    hib_map: dict[str, bool] = {}
    for dcfg in node.config.domains:
        did = getattr(dcfg, "domain_id", None) or getattr(dcfg, "id", None)
        if did:
            hib_map[did] = getattr(dcfg, "higher_is_better", False)

    if node.chaindb:
        height = node.chaindb.height
        # Track running best per domain
        running_best: dict[str, float] = {}
        champion_index: dict[str, int] = {}  # sequential champion number per domain

        for i in range(height):
            block = node.chaindb.get_block(i)
            if block is None:
                continue
            for tx in block.transactions:
                tx_type_val = tx.tx_type.value if hasattr(tx.tx_type, "value") else str(tx.tx_type)
                if tx_type_val != "optimae_accepted":
                    continue

                domain_id = tx.domain_id
                perf = tx.payload.get("verified_performance")
                if perf is None:
                    continue

                hib = hib_map.get(domain_id, False)
                prev = running_best.get(domain_id)
                is_improvement = (
                    prev is None
                    or (hib and perf > prev)
                    or (not hib and perf < prev)
                )
                if is_improvement:
                    running_best[domain_id] = perf

                seq = champion_index.get(domain_id, 0) + 1
                champion_index[domain_id] = seq

                # Block timestamp
                ts = block.header.timestamp
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

                metrics.append({
                    "champion_num": seq,
                    "block_index": block.header.index,
                    "domain_id": domain_id,
                    "performance": perf,
                    "best_performance": running_best[domain_id],
                    "is_improvement": is_improvement,
                    "peer_id": str(tx.peer_id)[:12],
                    "timestamp": ts_str,
                    "val_mae": tx.payload.get("val_mae"),
                    "train_mae": tx.payload.get("train_mae"),
                    "val_naive_mae": tx.payload.get("val_naive_mae"),
                    "train_naive_mae": tx.payload.get("train_naive_mae"),
                    "test_mae": tx.payload.get("test_mae"),
                    "test_naive_mae": tx.payload.get("test_naive_mae"),
                })

        # Apply limit (keep latest)
        if len(metrics) > limit:
            metrics = metrics[-limit:]

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
            txns = []
            for tx in block.transactions:
                txns.append({
                    "id": tx.id[:16] if tx.id else "",
                    "tx_type": tx.tx_type.value if hasattr(tx.tx_type, "value") else str(tx.tx_type),
                    "domain_id": tx.domain_id,
                    "peer_id": str(tx.peer_id)[:12],
                    "timestamp": tx.timestamp.isoformat() if hasattr(tx.timestamp, "isoformat") else str(tx.timestamp),
                    "payload": tx.payload,
                })
            blocks.append({
                "index": block.header.index,
                "hash": str(block.hash)[:16],
                "previous_hash": str(block.header.previous_hash)[:16],
                "timestamp": block.header.timestamp.isoformat() if hasattr(block.header.timestamp, "isoformat") else str(block.header.timestamp),
                "generator_id": str(block.header.generator_id)[:12],
                "weighted_sum": block.header.weighted_performance_sum,
                "threshold": block.header.threshold,
                "tx_count": len(block.transactions),
                "transactions": txns,
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


# ── Current Candidate (local, per-machine) ───────────────────

async def _api_candidate(request: web.Request) -> web.Response:
    """Return current candidate being evaluated (local state, not on blockchain)."""
    node = request.app["_doin_node"]
    return web.json_response(node._current_candidate or {}, dumps=_dumps)


# ── Alerts ───────────────────────────────────────────────────

async def _api_alerts(request: web.Request) -> web.Response:
    """Return accumulated alerts (version mismatches, anomalies, etc.)."""
    node = request.app["_doin_node"]
    limit = int(request.query.get("limit", "200"))
    alerts = list(reversed(node._alerts[-limit:]))
    return web.json_response({
        "alerts": alerts,
        "count": len(alerts),
        "unseen": node._alerts_unseen,
    }, dumps=_dumps)


async def _api_alerts_ack(request: web.Request) -> web.Response:
    """Reset the unseen alerts counter (user acknowledged them)."""
    node = request.app["_doin_node"]
    node._alerts_unseen = 0
    return web.json_response({"status": "ok"})
