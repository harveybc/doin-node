"""Unified DON Node — single process with configurable roles.

Like a Bitcoin node that can mine + validate + relay, a DON unified node
can optimize + evaluate + relay — all configurable per domain via JSON.

Roles:
  - relay:     Always on. Forwards messages, maintains chain, serves tasks.
  - optimizer: Produces optimae (optimization results) for configured domains.
  - evaluator: Verifies optimae by re-running optimization with synthetic data.

Security systems wired in:
  1. Commit-reveal for optimae                    (prevents front-running)
  2. Random quorum selection for verification     (prevents collusion)
  3. Asymmetric reputation penalties              (prevents rubber-stamping)
  4. Resource limits + bounds validation          (prevents DoS)
  5. Finality checkpoints                         (prevents long-range attacks)
  6. Reputation decay (EMA)                       (prevents reputation farming)
  7. Min reputation threshold for consensus       (prevents sybils)
  8. External checkpoint anchoring                (external validation)
  9. Fork choice rule                             (prevents selfish mining)
  10. Deterministic seed requirement              (prevents hidden randomness)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from doin_core.consensus import (
    DeterministicSeedPolicy,
    DifficultyController,
    ExternalAnchorManager,
    FinalityManager,
    ForkChoiceRule,
    IncentiveConfig,
    ProofOfOptimization,
    VerifiedUtilityWeights,
    evaluate_verification_incentive,
)
from doin_core.crypto.identity import PeerIdentity
from doin_core.models import (
    BalanceTracker,
    Block,
    BoundsValidator,
    CoinbaseTransaction,
    Commitment,
    CommitRevealManager,
    ContributorWork,
    Domain,
    Optimae,
    QuorumConfig,
    QuorumManager,
    ReputationTracker,
    ResourceLimits,
    Reveal,
    Task,
    TaskQueue,
    TaskStatus,
    TaskType,
    Transaction,
    TransactionType,
    compute_commitment,
    distribute_block_reward,
)
from doin_core.protocol.messages import (
    BlockAnnouncement,
    ChainStatus,
    Message,
    MessageType,
    OptimaeCommit,
    OptimaeReveal,
    TaskClaimed,
    TaskCompleted,
    TaskCreated,
)

from doin_core.models.fee_market import FeeConfig, FeeMarket

from doin_node.blockchain.chain import Chain
from doin_node.network.discovery import PeerDiscovery, DiscoveredPeer
from doin_node.network.flooding import FloodingConfig, FloodingProtocol
from doin_node.network.gossip import GossipSub
from doin_node.network.peer import Peer, PeerState
from doin_node.network.sync import SyncManager, fetch_blocks, fetch_chain_status
from doin_node.network.transport import Transport
from doin_node.stats.experiment_tracker import ExperimentTracker
from doin_node.storage.chaindb import ChainDB

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class DomainRole:
    """Per-domain role configuration."""

    domain_id: str
    optimize: bool = False
    evaluate: bool = False

    # Optimizer settings
    optimization_plugin: str = ""  # Entry point name
    optimization_config: dict[str, Any] = field(default_factory=dict)
    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Evaluator settings
    inference_plugin: str = ""
    synthetic_data_plugin: str = ""  # MANDATORY for verification trust
    has_synthetic_data: bool = False
    synthetic_data_validation: bool = True  # When False, skip quorum verification — auto-accept if better
    higher_is_better: bool = True  # False for predictor (lower fitness = better)

    # Stop criteria — optimization stops when performance >= this value (per-model)
    target_performance: float | None = None

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Incentive configuration (how verification rewards scale)
    incentive_config: IncentiveConfig = field(default_factory=IncentiveConfig)


@dataclass
class UnifiedNodeConfig:
    """Full configuration for a unified DON node."""

    host: str = "0.0.0.0"
    port: int = 8470
    data_dir: str = "./doin-data"
    identity_file: str = ""  # Auto-derived from data_dir if empty
    bootstrap_peers: list[str] = field(default_factory=list)

    # Domain roles
    domains: list[DomainRole] = field(default_factory=list)

    # Consensus
    target_block_time: float = 600.0
    initial_threshold: float = 1.0
    acceptance_tolerance: float = 1e-5  # Min improvement required to accept a new champion

    # Security
    quorum_min_evaluators: int = 3
    quorum_fraction: float = 0.67
    quorum_tolerance: float = 0.05
    commit_reveal_max_age: float = 600.0
    finality_confirmation_depth: int = 6
    external_anchor_interval: int = 100
    require_deterministic_seed: bool = True

    # Evaluator polling (when this node evaluates)
    eval_poll_interval: float = 10.0
    eval_max_concurrent: int = 3

    # Optimizer loop (when this node optimizes)
    optimizer_loop_interval: float = 30.0

    # Storage backend: "sqlite" (production) or "json" (legacy/testing)
    storage_backend: str = "sqlite"
    db_path: str = ""  # Auto-derived from data_dir if empty
    snapshot_interval: int = 100  # Save state snapshot every N blocks
    prune_keep_blocks: int = 10000  # Keep tx bodies for last N blocks

    # Network: "gossipsub" (production) or "flooding" (legacy/testing)
    network_protocol: str = "gossipsub"
    gossip_heartbeat_interval: float = 1.0

    # Peer discovery
    discovery_enabled: bool = True
    discovery_interval: float = 60.0  # PEX + random walk interval

    # Web dashboard (AdminLTE — served on the same port at /dashboard)
    dashboard_enabled: bool = True

    # Experiment stats tracking (CSV for OLAP)
    experiment_stats_file: str = ""  # Auto-derived from data_dir if empty

    # OLAP SQLite database (auto-saves every round — no manual ETL)
    olap_db_path: str = ""  # Auto-derived from data_dir if empty

    # Reset chain on startup (auto-clean blockchain and OLAP data)
    reset_chain: bool = False

    # Fee market
    fee_market_enabled: bool = True
    fee_config: FeeConfig = field(default_factory=FeeConfig)


# ── Unified Node ─────────────────────────────────────────────────────

class UnifiedNode:
    """A single configurable DON node — optimizer, evaluator, and relay.

    This is the main entry point for running a DON node. Configure it
    via JSON (see UnifiedNodeConfig) to control which roles it performs
    for which domains.
    """

    def __init__(
        self,
        config: UnifiedNodeConfig,
        identity: PeerIdentity | None = None,
    ) -> None:
        self.config = config

        # Identity: use provided, or load from file (persists across restarts)
        if identity:
            self.identity = identity
        else:
            id_path = config.identity_file or str(Path(config.data_dir) / "identity.pem")
            self.identity = PeerIdentity.load_or_generate(id_path)
            logger.info("Peer identity: %s (from %s)", self.identity.peer_id[:12], id_path)

        # ── Core components ──
        self.transport = Transport(host=config.host, port=config.port)
        self.transport._peer_id = self.identity.peer_id
        self.transport._get_peers_fn = self._get_peers_for_discovery

        # Network protocol: GossipSub (production) or Flooding (legacy)
        self.flooding = FloodingProtocol(FloodingConfig())
        self.gossip: GossipSub | None = None
        if config.network_protocol == "gossipsub":
            self.gossip = GossipSub(peer_id=self.identity.peer_id)
            self.gossip.subscribe_all()

        # Storage backend: SQLite (production) or JSON Chain (legacy)
        self.chain = Chain(data_dir=Path(config.data_dir))
        self.chaindb: ChainDB | None = None
        if config.storage_backend == "sqlite":
            db_path = config.db_path or str(Path(config.data_dir) / "chain.db")
            self.chaindb = ChainDB(db_path)

        # ── Component versions (computed once, used for peer version handshake) ──
        self._component_versions: dict[str, str] = self._compute_component_versions()
        logger.info("Component versions: %s", self._component_versions)

        # ── Plugin configs (computed after domain registration, used for peer plugin handshake) ──
        self._plugin_configs: dict[str, str] = {}

        # Peer discovery
        self.discovery: PeerDiscovery | None = None
        if config.discovery_enabled:
            self.discovery = PeerDiscovery(
                our_peer_id=self.identity.peer_id,
                our_port=config.port,
                bootstrap_nodes=config.bootstrap_peers,
                required_versions=self._component_versions,
            )

        # Fee market
        self.fee_market: FeeMarket | None = None
        if config.fee_market_enabled:
            self.fee_market = FeeMarket(config.fee_config)

        self.consensus = ProofOfOptimization(
            target_block_time=config.target_block_time,
            initial_threshold=config.initial_threshold,
        )
        self.task_queue = TaskQueue()

        # ── Security systems ──
        self.reputation = ReputationTracker()
        self.quorum = QuorumManager(QuorumConfig(
            min_evaluators=config.quorum_min_evaluators,
            quorum_fraction=config.quorum_fraction,
            tolerance=config.quorum_tolerance,
        ))
        self.commit_reveal = CommitRevealManager(
            max_commit_age=config.commit_reveal_max_age,
        )
        self.finality = FinalityManager(
            confirmation_depth=config.finality_confirmation_depth,
        )
        self.anchor_manager = ExternalAnchorManager(
            anchor_interval_blocks=config.external_anchor_interval,
        )
        self.fork_choice = ForkChoiceRule()
        self.seed_policy = DeterministicSeedPolicy(
            require_seed=config.require_deterministic_seed,
        )
        self.vuw = VerifiedUtilityWeights()
        self.difficulty = DifficultyController(
            target_block_time=config.target_block_time,
            initial_threshold=config.initial_threshold,
        )
        self.balance_tracker = BalanceTracker()
        self.sync_manager = SyncManager()
        self._block_contributors: list[ContributorWork] = []  # Contributors for current block

        # Own network addresses (all local IPs) — populated on start() to prevent
        # self-registration in the peer table when messages bounce back via multi-homed routes.
        self._own_addresses: set[str] = set()

        # ── Per-domain state ──
        self._domains: dict[str, Domain] = {}
        self._domain_roles: dict[str, DomainRole] = {}
        self._bounds_validators: dict[str, BoundsValidator] = {}
        self._peers: dict[str, Peer] = {}
        self._running = False
        self._optimizer_plugins: dict[str, Any] = {}  # domain_id → plugin instance
        self._evaluator_plugins: dict[str, Any] = {}  # domain_id → plugin instance
        self._synthetic_plugins: dict[str, Any] = {}  # domain_id → plugin instance
        self._background_tasks: list[asyncio.Task] = []
        self._domain_best: dict[str, tuple[dict[str, Any] | None, float | None]] = {}  # domain_id → (best_params, best_perf)
        self._domain_converged: set[str] = set()  # domains that reached target_performance
        self._domain_round_count: dict[str, int] = {}  # domain_id → round number
        self._domain_champion_metrics: dict[str, dict[str, Any]] = {}  # domain_id → champion detail metrics
        self._domain_stage_start_fitness: dict[str, float] = {}  # domain_id → fitness at start of current stage
        self._domain_patience: dict[str, tuple[int, int]] = {}  # domain_id → (no_improve_counter, patience_max)
        self._current_candidate: dict[str, Any] = {}  # current candidate being evaluated (local, per-machine)
        self._seen_candidate_tx_ids: set[str] = set()  # dedup candidate evaluation transactions
        self._start_time: float = time.time()

        # ── Live event log (for dashboard) ──
        self._live_events: list[dict[str, Any]] = []  # Most recent events (capped at 500)
        self._live_events_max = 500

        # ── Experiment tracker ──
        stats_file = config.experiment_stats_file or str(
            Path(config.data_dir) / "experiment_stats.csv"
        )
        olap_path = config.olap_db_path or str(
            Path(config.data_dir) / "olap.db"
        )
        self._olap_db_path = str(Path(olap_path).resolve())  # Expose for dashboard (absolute)
        self.experiment_tracker = ExperimentTracker(
            csv_path=stats_file,
            node_id=self.identity.peer_id,
            doin_version="0.1.0",
            olap_db_path=olap_path,
        )

        # ── Wire up message handlers ──
        # Register on both flooding (legacy) and gossip (production)
        all_handlers = [
            (MessageType.OPTIMAE_COMMIT, self._handle_optimae_commit),
            (MessageType.OPTIMAE_REVEAL, self._handle_optimae_reveal),
            (MessageType.OPTIMAE_ANNOUNCEMENT, self._handle_optimae_announcement),
            (MessageType.BLOCK_ANNOUNCEMENT, self._handle_block_announcement),
            (MessageType.TASK_CREATED, self._handle_task_created),
            (MessageType.TASK_CLAIMED, self._handle_task_claimed),
            (MessageType.TASK_COMPLETED, self._handle_task_completed),
            (MessageType.CHAMPION_REQUEST, self._handle_champion_request),
            (MessageType.CHAMPION_RESPONSE, self._handle_champion_response),
            (MessageType.STAGE_COMPLETE, self._handle_stage_complete),
            (MessageType.CANDIDATE_EVALUATION, self._handle_candidate_evaluation),
        ]
        for msg_type, handler in all_handlers:
            self.flooding.on_message(msg_type, handler)
            if self.gossip:
                self.gossip.on_message(msg_type, handler)

        self.transport.on_message(self._on_transport_message)

        # ── Register domains ──
        for domain_role in config.domains:
            self._register_domain(domain_role)

        # ── Populate plugin configs for handshake verification ──
        for did, dr in self._domain_roles.items():
            oc = dr.optimization_config or {}
            for key in ("predictor_plugin", "optimizer_plugin"):
                if oc.get(key):
                    self._plugin_configs[f"{did}:{key}"] = oc[key]
        if self._plugin_configs:
            logger.info("Plugin configs: %s", self._plugin_configs)

    # ================================================================
    # Properties
    # ================================================================

    @property
    def peer_id(self) -> str:
        return self.identity.peer_id

    @property
    def optimizer_domains(self) -> list[str]:
        return [d for d, r in self._domain_roles.items() if r.optimize]

    @property
    def evaluator_domains(self) -> list[str]:
        return [d for d, r in self._domain_roles.items() if r.evaluate]

    # ================================================================
    # Setup
    # ================================================================

    def _register_domain(self, role: DomainRole) -> None:
        """Register a domain and its role configuration."""
        from doin_core.models.domain import DomainConfig
        domain = Domain(
            id=role.domain_id,
            name=role.domain_id,
            performance_metric="fitness",
            config=DomainConfig(
                optimization_plugin=role.optimization_plugin or "default",
                inference_plugin=role.inference_plugin or "default",
                synthetic_data_plugin=role.synthetic_data_plugin or None,
            ),
        )
        self._domains[role.domain_id] = domain
        self._domain_roles[role.domain_id] = role
        self.consensus.register_domain(domain)

        # Bounds validator
        if role.param_bounds:
            validator = BoundsValidator(role.param_bounds)
        else:
            validator = BoundsValidator()
        self._bounds_validators[role.domain_id] = validator

        # VUW registration
        self.vuw.register_domain(
            role.domain_id,
            base_weight=1.0,
            has_synthetic_data=role.has_synthetic_data,
        )

        logger.info(
            "Domain %s registered (optimize=%s, evaluate=%s, synthetic=%s)",
            role.domain_id, role.optimize, role.evaluate, role.has_synthetic_data,
        )

    def register_optimizer_plugin(self, domain_id: str, plugin: Any) -> None:
        """Register an optimizer plugin for a domain."""
        self._optimizer_plugins[domain_id] = plugin

    def register_evaluator_plugin(self, domain_id: str, plugin: Any) -> None:
        """Register an evaluator/inferencer plugin for a domain."""
        self._evaluator_plugins[domain_id] = plugin

    def register_synthetic_plugin(self, domain_id: str, plugin: Any) -> None:
        """Register a synthetic data generator plugin for a domain."""
        self._synthetic_plugins[domain_id] = plugin

    @staticmethod
    def _compute_component_versions() -> dict[str, str]:
        """Compute git short hashes for DOIN component packages.

        Used for the version handshake protocol: peers must have identical
        component versions to be accepted into the network.
        """
        import importlib.metadata
        import importlib.util
        import subprocess as _sp

        def _git_hash(path: str) -> str:
            try:
                r = _sp.run(
                    ["git", "rev-parse", "--short=7", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd=path,
                )
                if r.returncode == 0:
                    return r.stdout.strip()
            except Exception:
                pass
            return "unknown"

        versions: dict[str, str] = {}
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
            try:
                spec = importlib.util.find_spec(import_name)
                if spec and spec.origin:
                    pkg_dir = Path(spec.origin).resolve().parent
                    for parent in [pkg_dir] + list(pkg_dir.parents):
                        if (parent / ".git").exists():
                            versions[label] = _git_hash(str(parent))
                            found = True
                            break
            except Exception:
                pass
            if found:
                continue
            for cand in candidates:
                if cand.exists() and (cand / ".git").exists():
                    versions[label] = _git_hash(str(cand))
                    found = True
                    break
            if found:
                continue
            try:
                ver = importlib.metadata.version(dist_name)
                versions[label] = f"v{ver}"
            except Exception:
                versions[label] = "?"
        return versions

    def _peer_id_exists(self, peer_id: str) -> bool:
        """Check if a peer with this ID already exists (on any endpoint)."""
        return any(p.peer_id == peer_id for p in self._peers.values())

    def _is_better(self, domain_id: str, new_perf: float, old_perf: float | None) -> bool:
        """Compare performances respecting higher_is_better setting and acceptance_tolerance."""
        if old_perf is None:
            return True
        tol = self.config.acceptance_tolerance
        role = self._domain_roles.get(domain_id)
        if role and not role.higher_is_better:
            return new_perf < old_perf - tol  # Lower = better (e.g. predictor fitness)
        return new_perf > old_perf + tol  # Higher = better (default)

    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        """Add an event to the live event log (for dashboard display)."""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._live_events.append(event)
        if len(self._live_events) > self._live_events_max:
            self._live_events = self._live_events[-self._live_events_max:]

    def _get_peers_for_discovery(self) -> list[dict]:
        """Return known peers for the /peers endpoint."""
        return [
            {"peer_id": p.peer_id, "address": p.address, "port": p.port}
            for p in self._peers.values()
            if p.peer_id != self.peer_id
        ]

    def add_peer(self, address: str, port: int, peer_id: str = "") -> Peer:
        pid = peer_id or f"{address}:{port}"
        endpoint = f"{address}:{port}"
        # If we already know this endpoint, just return it.
        if endpoint in self._peers:
            existing = self._peers[endpoint]
            # Update placeholder with real peer_id if we now know it.
            if peer_id and existing.peer_id != peer_id and existing.peer_id == endpoint:
                old_id = existing.peer_id
                existing.peer_id = peer_id
                if self.gossip:
                    self.gossip.rename_peer(old_id, peer_id)
            return existing
        peer = Peer(
            peer_id=pid,
            address=address,
            port=port,
            state=PeerState.DISCOVERED,
        )
        self._peers[peer.endpoint] = peer
        return peer

    async def _check_port_reachable(self) -> None:
        """Self-check: verify our listening port is reachable from localhost."""
        import aiohttp as _aio
        url = f"http://127.0.0.1:{self.config.port}/chain/status"
        try:
            async with _aio.ClientSession() as s:
                async with s.get(url, timeout=_aio.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        logger.info("✅ Port %d self-check passed", self.config.port)
                    else:
                        logger.warning("⚠️  Port %d self-check: HTTP %d", self.config.port, r.status)
        except Exception:
            logger.warning(
                "⚠️  Port %d may be blocked! Peers won't be able to reach this node. "
                "Check firewall: sudo ufw allow %d/tcp",
                self.config.port, self.config.port,
            )

    # ================================================================
    # Lifecycle
    # ================================================================

    async def start(self) -> None:
        """Start the unified node."""
        self._running = True

        # Discover own local IP addresses so we never register ourselves as a peer.
        # Three independent methods tried in order; results are merged so that
        # failure of one method doesn't leave us blind.
        import socket
        import subprocess as _sp
        self._own_addresses = set()

        # Method 1 — psutil (most reliable, enumerates every interface)
        try:
            import psutil as _psutil
            for _addrs in _psutil.net_if_addrs().values():
                for _a in _addrs:
                    if _a.family in (socket.AF_INET, socket.AF_INET6):
                        self._own_addresses.add(_a.address.split('%')[0])
        except Exception:
            pass

        # Method 2 — `ip -4 -o addr` (Linux, reliable even without psutil)
        try:
            _res = _sp.run(["ip", "-4", "-o", "addr"],
                           capture_output=True, text=True, timeout=3)
            for _line in _res.stdout.strip().split("\n"):
                _parts = _line.split()
                if len(_parts) >= 4 and _parts[1] != "lo":
                    self._own_addresses.add(_parts[3].split("/")[0])
        except Exception:
            pass

        # Method 3 — UDP connect trick (gets the primary outbound IP)
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))
            self._own_addresses.add(_s.getsockname()[0])
            _s.close()
        except Exception:
            pass

        # Always include loopback variants
        self._own_addresses.update({"127.0.0.1", "127.0.1.1", "::1", "localhost", "unknown"})
        logger.info("Own addresses: %s", self._own_addresses)

        # Initialize storage
        if self.chaindb:
            self.chaindb.open()
            if self.chaindb.height == 0:
                self.chaindb.initialize("genesis")
            logger.info("SQLite storage: height=%d", self.chaindb.height)
        else:
            chain_path = Path(self.config.data_dir) / "chain.json"
            if chain_path.exists():
                self.chain.load()
            else:
                self.chain.initialize("genesis")
                self.chain.save()

        # Wire gossip send function to transport
        if self.gossip:
            self.gossip.set_send_fn(self._gossip_send)

        # Start experiment tracking for optimizer domains
        for domain_id in self.optimizer_domains:
            role = self._domain_roles[domain_id]
            self.experiment_tracker.start_experiment(
                domain_id,
                optimization_config=role.optimization_config,
                param_bounds={k: list(v) for k, v in role.param_bounds.items()},
                target_performance=role.target_performance,
                optimizer_plugin=role.optimization_plugin,
            )

        # Register HTTP routes
        self._register_http_routes()

        # Start transport
        await self.transport.start()

        # Port connectivity self-check
        await self._check_port_reachable()

        # Connect bootstrap peers
        for addr in self.config.bootstrap_peers:
            host, port_str = addr.rsplit(":", 1)
            peer = self.add_peer(host, int(port_str))
            if self.gossip:
                self.gossip.add_peer(peer.peer_id)
            if self.discovery:
                self.discovery.add_peer(DiscoveredPeer(
                    peer_id=peer.peer_id,
                    address=host,
                    port=int(port_str),
                    source="config",
                ))
                self.discovery.mark_connected(peer.endpoint)

        # Update sync manager state
        height = self._get_height()
        tip = self._get_tip()
        self.sync_manager.update_our_state(
            height,
            tip.hash if tip else "",
            self.finality.finalized_height,
        )

        # Initial sync with peers
        await self._initial_sync()

        # Start background loops for roles
        if self.optimizer_domains:
            self._background_tasks.append(
                asyncio.create_task(self._optimizer_loop())
            )
        if self.evaluator_domains:
            self._background_tasks.append(
                asyncio.create_task(self._evaluator_loop())
            )

        # Gossip heartbeat loop
        if self.gossip:
            self._background_tasks.append(
                asyncio.create_task(self._gossip_heartbeat_loop())
            )

        # Peer discovery loop
        if self.discovery:
            self._background_tasks.append(
                asyncio.create_task(self._discovery_loop())
            )

        # Periodic cleanup
        self._background_tasks.append(
            asyncio.create_task(self._maintenance_loop())
        )

        protocol = "gossipsub" if self.gossip else "flooding"
        storage = "sqlite" if self.chaindb else "json"
        logger.info(
            "Unified node started: peer=%s port=%d optimize=%s evaluate=%s "
            "protocol=%s storage=%s fee_market=%s discovery=%s",
            self.peer_id[:12],
            self.config.port,
            self.optimizer_domains,
            self.evaluator_domains,
            protocol,
            storage,
            self.fee_market is not None,
            self.discovery is not None,
        )

    async def stop(self) -> None:
        """Stop the node."""
        self._running = False
        for t in self._background_tasks:
            t.cancel()
        await self.transport.stop()
        if self.chaindb:
            self.chaindb.close()
        else:
            self.chain.save()
        self.experiment_tracker.finalize()
        logger.info("Unified node stopped")

    # ================================================================
    # Storage abstraction (works with both SQLite and JSON backends)
    # ================================================================

    def _get_height(self) -> int:
        if self.chaindb:
            return self.chaindb.height
        return self.chain.height

    def _get_tip(self) -> Block | None:
        if self.chaindb:
            return self.chaindb.get_tip()
        return self.chain.tip

    def _get_block(self, index: int) -> Block | None:
        if self.chaindb:
            return self.chaindb.get_block(index)
        return self.chain.get_block(index)

    def _get_blocks_range(self, from_idx: int, to_idx: int) -> list[Block]:
        if self.chaindb:
            return self.chaindb.get_blocks_range(from_idx, to_idx)
        return self.chain.get_blocks_range(from_idx, to_idx)

    def _append_block(self, block: Block) -> None:
        if self.chaindb:
            self.chaindb.append_block(block)
        else:
            self.chain.append_block(block)

    def _validate_and_append_blocks(self, blocks: list[Block]) -> int:
        if self.chaindb:
            return self.chaindb.append_blocks(blocks)
        return self.chain.validate_and_append_blocks(blocks)

    def _save_chain(self) -> None:
        if not self.chaindb:
            self.chain.save()

    def _update_domain_best_from_chain(self) -> None:
        """Scan chain for OPTIMAE_ACCEPTED transactions and update domain best.

        This is critical for the island model: when syncing blocks from peers,
        we pick up champion solutions from other nodes and use them as the
        starting point for our next optimization round.
        Also restores champion metrics (MAE breakdowns) from the chain.
        """
        height = self._get_height()
        for i in range(height):
            block = self._get_block(i)
            if block is None:
                continue
            for tx in block.transactions:
                if tx.tx_type == TransactionType.OPTIMAE_ACCEPTED:
                    domain_id = tx.domain_id
                    verified = tx.payload.get("verified_performance")
                    parameters = tx.payload.get("parameters")
                    if verified is not None and parameters is not None:
                        current = self._domain_best.get(domain_id, (None, None))
                        if self._is_better(domain_id, verified, current[1]):
                            self._domain_best[domain_id] = (parameters, verified)
                            # Restore champion metrics from chain transaction
                            cm = {k: tx.payload.get(k) for k in [
                                "val_mae", "train_mae", "val_naive_mae",
                                "train_naive_mae", "test_mae", "test_naive_mae",
                            ] if tx.payload.get(k) is not None}
                            if cm:
                                self._domain_champion_metrics[domain_id] = {
                                    "performance": verified,
                                    **cm,
                                }
                            logger.info(
                                "⚡ Synced champion for %s: perf=%.6f (from block #%d)",
                                domain_id, verified, block.header.index,
                            )

    # ================================================================
    # Network abstraction (gossipsub or flooding)
    # ================================================================

    async def _broadcast(self, message: Message) -> None:
        """Broadcast a message using the configured protocol."""
        # Include sender port so receivers can connect back on the right port
        if message.payload and isinstance(message.payload, dict):
            message.payload["_sender_port"] = self.config.port
            message.payload["_component_versions"] = self._component_versions
            message.payload["_plugin_configs"] = self._plugin_configs
        elif message.payload is None:
            message.payload = {"_sender_port": self.config.port, "_component_versions": self._component_versions, "_plugin_configs": self._plugin_configs}
        if self.gossip:
            sent = await self.gossip.publish(message)
            logger.info("📤 Broadcast %s → %d peers (gossip mesh)", message.msg_type.value, sent)
        else:
            await self.transport.broadcast(list(self._peers.keys()), message)
            logger.info("📤 Broadcast %s → %d peers (flooding)", message.msg_type.value, len(self._peers))

    async def _gossip_send(self, peer_id: str, payload: dict) -> bool:
        """Send a message to a specific peer (callback for GossipSub)."""
        for ep, peer in self._peers.items():
            if peer.peer_id == peer_id:
                url = f"http://{ep}/message"
                session = self.transport._session
                if not session:
                    logger.warning("GossipSend: no transport session")
                    return False
                try:
                    async with session.post(url, json=payload) as resp:
                        return resp.status == 200
                except Exception as e:
                    logger.warning("GossipSend to %s (%s) error: %s: %s", peer_id[:12], ep, type(e).__name__, e)
                    return False
        logger.debug("GossipSend: peer %s not found in _peers (have: %s)", peer_id[:12],
                      [f"{p.peer_id[:12]}@{e}" for e, p in self._peers.items()])
        return False

    # ================================================================
    # Message handling (relay — always on)
    # ================================================================

    async def _on_transport_message(self, message: Message, sender: str) -> None:
        # Auto-discover peers from incoming connections.
        # Extract the sender's advertised port from the payload.
        sender_port = (message.payload or {}).get("_sender_port", self.config.port)
        sender_endpoint = f"{sender}:{sender_port}"

        # Version handshake: reject messages from peers with mismatched component versions.
        peer_versions = (message.payload or {}).get("_component_versions")
        if peer_versions and self._component_versions:
            mismatches = {
                k: (v, peer_versions.get(k, "?"))
                for k, v in self._component_versions.items()
                if peer_versions.get(k) != v
            }
            if mismatches:
                logger.warning(
                    "🚫 Dropping message %s from %s — version mismatch: %s",
                    message.msg_type.value, sender_endpoint,
                    ", ".join(f"{k}: ours={ov} theirs={tv}" for k, (ov, tv) in mismatches.items()),
                )
                return

        # Plugin config handshake: reject messages from peers with different predictor/optimizer plugins.
        peer_plugins = (message.payload or {}).get("_plugin_configs")
        if peer_plugins and self._plugin_configs:
            plugin_mismatches = {
                k: (v, peer_plugins.get(k, "?"))
                for k, v in self._plugin_configs.items()
                if peer_plugins.get(k) != v
            }
            if plugin_mismatches:
                logger.warning(
                    "🚫 Dropping message %s from %s — plugin mismatch: %s",
                    message.msg_type.value, sender_endpoint,
                    ", ".join(f"{k}: ours={ov} theirs={tv}" for k, (ov, tv) in plugin_mismatches.items()),
                )
                return

        # _forwarder_id is stamped by the node that physically relayed this message.
        # When present, the HTTP connection came from the forwarder, not the original author.
        # Rule:
        #   - Direct message  (no _forwarder_id): sender IP → message.sender_id
        #   - Forwarded message (_forwarder_id present): sender IP → _forwarder_id
        #     The original message.sender_id is preserved for protocol logic but does NOT
        #     change the routing table (it may be a peer on a different subnet).
        forwarder_id = (message.payload or {}).get("_forwarder_id")
        is_forwarded = bool(forwarder_id and forwarder_id != self.peer_id)
        routing_peer_id = forwarder_id if is_forwarded else message.sender_id

        # Never register our own IPs as peers (happens when messages loop back
        # via a different interface on dual-homed machines).
        if sender not in self._own_addresses and routing_peer_id != self.peer_id:
            existing = self._peers.get(sender_endpoint)
            if existing:
                # Update placeholder ID (e.g. "IP:port" string) with the real peer_id.
                if existing.peer_id != routing_peer_id and existing.peer_id == sender_endpoint:
                    old_id = existing.peer_id
                    existing.peer_id = routing_peer_id
                    logger.info("🔍 Resolved peer %s → %s", old_id, routing_peer_id[:12])
                    if self.gossip:
                        self.gossip.rename_peer(old_id, routing_peer_id)
            else:
                self.add_peer(sender, sender_port, peer_id=routing_peer_id)
                logger.info("🔍 Auto-discovered peer %s:%s (id=%s)",
                             sender, sender_port, routing_peer_id[:12])

        if self.gossip:
            # GossipSub handles dedup, dispatch, and mesh forwarding
            await self.gossip.handle_incoming(message, sender)
        else:
            # Legacy flooding
            should_forward = await self.flooding.handle_incoming(message, sender)
            if should_forward:
                forward_msg = self.flooding.prepare_forward(message)
                # Copy payload dict to avoid mutating the original (prepare_forward
                # shares the dict by reference, causing forwarder_id cross-contamination).
                forward_msg = forward_msg.model_copy(
                    update={"payload": {**(forward_msg.payload or {}),
                                        "_forwarder_id": self.peer_id,
                                        "_sender_port": self.config.port,
                                        "_component_versions": self._component_versions,
                                        "_plugin_configs": self._plugin_configs}}
                )
                # Exclude the actual sender endpoint (compare by IP prefix, since
                # _peers keys are "IP:port" but `sender` is a bare IP string).
                endpoints = [ep for ep in self._peers if not ep.startswith(sender + ":")]
                await self.transport.broadcast(endpoints, forward_msg)

    # ── Commit-reveal flow ───────────────────────────────────────

    async def _handle_optimae_commit(self, message: Message, from_peer: str) -> None:
        """Handle Phase 1: commitment hash received."""
        data = OptimaeCommit.model_validate(message.payload)
        commitment = Commitment(
            commitment_hash=data.commitment_hash,
            domain_id=data.domain_id,
            optimizer_id=message.sender_id,
        )
        added = self.commit_reveal.add_commitment(commitment)
        if added:
            is_remote = message.sender_id != self.peer_id
            if is_remote:
                logger.info(
                    "📥 Received commitment from peer %s for %s",
                    message.sender_id[:12], data.domain_id,
                )
            else:
                logger.debug(
                    "Commitment %s from self for %s",
                    data.commitment_hash[:12], data.domain_id,
                )

    async def _handle_optimae_reveal(self, message: Message, from_peer: str) -> None:
        """Handle Phase 2: reveal — validate hash, validate seed, start quorum."""
        data = OptimaeReveal.model_validate(message.payload)

        reveal = Reveal(
            commitment_hash=data.commitment_hash,
            domain_id=data.domain_id,
            optimizer_id=message.sender_id,
            parameters=data.parameters,
            nonce=data.nonce,
            reported_performance=data.reported_performance,
        )

        # Verify commitment matches
        if not self.commit_reveal.process_reveal(reveal):
            logger.warning("Invalid reveal from %s (hash mismatch or expired)", message.sender_id[:12])
            return

        # Validate parameter bounds (warn only — don't reject peer optimae)
        validator = self._bounds_validators.get(data.domain_id)
        if validator:
            ok, reason = validator.validate(data.parameters)
            if not ok:
                logger.info("Bounds note (not rejecting): %s", reason)

        # Check minimum reputation
        if not self.reputation.meets_threshold(message.sender_id):
            logger.warning(
                "Optimizer %s below reputation threshold (%.2f < %.2f)",
                message.sender_id[:12],
                self.reputation.get_score(message.sender_id),
                2.0,
            )
            # Still allow — but their optimae won't count toward consensus
            # (effective increment will be near-zero due to low reputation)

        # ── Synthetic data validation bypass ──────────────────────
        role = self._domain_roles.get(data.domain_id)
        skip_validation = role and not role.synthetic_data_validation

        # ── Optimistic adoption (island model) ──────────────────────
        # Only when using full validation (consensus runs in parallel).
        # When skipping validation, auto-accept handles domain_best update.
        if not skip_validation and message.sender_id != self.peer_id:
            current_best = self._domain_best.get(data.domain_id, (None, None))
            if self._is_better(data.domain_id, data.reported_performance, current_best[1]):
                self._domain_best[data.domain_id] = (data.parameters, data.reported_performance)
                # Update champion metrics from reveal
                cm = getattr(data, 'champion_metrics', None) or {}
                if cm:
                    self._domain_champion_metrics[data.domain_id] = {
                        "performance": data.reported_performance,
                        **{k: cm[k] for k in [
                            "val_mae", "train_mae", "val_naive_mae", "train_naive_mae",
                            "test_mae", "test_naive_mae",
                        ] if k in cm and cm[k] is not None},
                    }
                logger.info(
                    "🏝️  MIGRATION: optimistic adopt from peer %s for %s: %.6f → %.6f",
                    message.sender_id[:12], data.domain_id,
                    current_best[1] if current_best[1] is not None else float('-inf'),
                    data.reported_performance,
                )
                # Inject into optimizer plugin's population (migration IN)
                plugin = self._optimizer_plugins.get(data.domain_id)
                if plugin and hasattr(plugin, "set_network_champion"):
                    plugin.set_network_champion(data.parameters)

        if skip_validation:
            # Auto-accept/reject based on reported performance vs current best
            current_best = self._domain_best.get(data.domain_id, (None, None))
            is_better = self._is_better(data.domain_id, data.reported_performance, current_best[1])

            if is_better:
                # Auto-accept — treat reported_performance as verified
                await self._auto_accept_optimae(data, message.sender_id)
                # Extract champion metrics from reveal (if originator included them)
                cm = getattr(data, 'champion_metrics', None) or {}
                self._log_event("auto_accept",
                    domain_id=data.domain_id, optimae_id=data.optimae_id[:12],
                    peer=message.sender_id[:12], is_self=message.sender_id == self.peer_id,
                    performance=data.reported_performance,
                    previous_best=current_best[1] if current_best[1] is not None else None,
                    val_mae=cm.get("val_mae"), train_mae=cm.get("train_mae"),
                    val_naive_mae=cm.get("val_naive_mae"), train_naive_mae=cm.get("train_naive_mae"),
                    test_mae=cm.get("test_mae"), test_naive_mae=cm.get("test_naive_mae"))
                logger.info(
                    "✅ AUTO-ACCEPT optimae %s (no synthetic validation): perf=%.6f beats best=%.6f (%s)",
                    data.optimae_id[:12], data.reported_performance,
                    current_best[1] if current_best[1] is not None else float('inf'),
                    "lower=better" if role and not role.higher_is_better else "higher=better",
                )
            else:
                # Auto-reject — log on blockchain
                self.consensus.record_transaction(Transaction(
                    tx_type=TransactionType.OPTIMAE_REJECTED,
                    domain_id=data.domain_id,
                    peer_id=message.sender_id,
                    payload={
                        "optimae_id": data.optimae_id,
                        "reason": f"reported perf {data.reported_performance:.6f} <= current best {current_best[1]:.6f}",
                    },
                ))
                self._log_event("auto_reject",
                    domain_id=data.domain_id, optimae_id=data.optimae_id[:12],
                    peer=message.sender_id[:12], is_self=message.sender_id == self.peer_id,
                    performance=data.reported_performance,
                    current_best=current_best[1])
                logger.info(
                    "❌ AUTO-REJECT optimae %s: perf=%.6f does not beat best=%.6f (%s)",
                    data.optimae_id[:12], data.reported_performance,
                    current_best[1] if current_best[1] is not None else float('inf'),
                    "lower=better" if role and not role.higher_is_better else "higher=better",
                )

                # ── Send current champion back to the rejected sender ──
                # This accelerates convergence: the sender learns the latest
                # network optimum immediately instead of waiting for the next
                # generation-start champion request.
                if message.sender_id != self.peer_id and current_best[0] is not None:
                    champion_msg = Message(
                        msg_type=MessageType.CHAMPION_RESPONSE,
                        sender_id=self.peer_id,
                        payload={
                            "domain_id": data.domain_id,
                            "request_id": f"reject-push-{data.optimae_id[:12]}",
                            "parameters": current_best[0],
                            "performance": current_best[1],
                            "has_champion": True,
                        },
                    )
                    endpoint = self._find_peer_endpoint(from_peer, message.sender_id)
                    if endpoint:
                        try:
                            await self.transport.send(endpoint, champion_msg)
                            logger.info(
                                "📡 Pushed champion to rejected sender %s for %s (best=%.6f)",
                                message.sender_id[:12], data.domain_id, current_best[1],
                            )
                        except Exception:
                            logger.debug("Failed to push champion to %s", message.sender_id[:12])
            return

        # ── Full synthetic data verification path ──────────────────
        if role and not role.has_synthetic_data:
            logger.warning(
                "Domain %s has no synthetic data — optimae will have zero consensus weight",
                data.domain_id,
            )

        # Select quorum evaluators
        eligible = self._get_eligible_evaluators(data.domain_id)
        chain_tip = self.chain.tip
        chain_tip_hash = chain_tip.hash if chain_tip else "genesis"

        selected = self.quorum.select_evaluators(
            optimae_id=data.optimae_id,
            domain_id=data.domain_id,
            optimizer_id=message.sender_id,
            reported_performance=data.reported_performance,
            eligible_evaluators=eligible,
            chain_tip_hash=chain_tip_hash,
        )

        if not selected:
            logger.warning("No eligible evaluators for domain %s", data.domain_id)
            return

        # Create verification task
        task = Task(
            task_type=TaskType.OPTIMAE_VERIFICATION,
            domain_id=data.domain_id,
            requester_id=message.sender_id,
            parameters=data.parameters,
            optimae_id=data.optimae_id,
            reported_performance=data.reported_performance,
            priority=0,
        )
        self.task_queue.add(task)
        await self._flood_task_created(task)

        logger.info(
            "Reveal accepted: optimae=%s, quorum=%s, task=%s",
            data.optimae_id[:12],
            [s[:8] for s in selected],
            task.id[:12],
        )

    async def _auto_accept_optimae(self, data: Any, sender_id: str) -> None:
        """Auto-accept an optimae without synthetic data verification.

        Used when synthetic_data_validation=False. Records the acceptance on-chain,
        updates domain best, distributes rewards, and triggers block generation —
        same as the full quorum path but skipping evaluator verification.
        """
        import math

        reported = data.reported_performance
        verified = reported  # Trust reported performance

        # Reputation
        self.reputation.record_optimae_accepted(sender_id)
        rep_score = self.reputation.get_score(sender_id)
        rep_factor = min(1.0, math.log1p(rep_score) / math.log1p(10.0)) if rep_score > 0 else 0.0

        # Incentive
        role = self._domain_roles.get(data.domain_id)
        incentive_cfg = role.incentive_config if role else IncentiveConfig()
        raw_increment = abs(reported)
        weights = self.vuw.compute_weights()
        domain_weight = weights.get(data.domain_id, 0.0)

        incentive_result = evaluate_verification_incentive(
            reported_performance=reported,
            verified_performance=verified,
            raw_increment=raw_increment,
            domain_weight=domain_weight,
            reputation_factor=rep_factor,
            config=incentive_cfg,
        )
        effective_increment = incentive_result.effective_increment

        logger.info(
            "🧮 INCENTIVE: raw=%.6f, dw=%.4f, rep=%.4f, reward=%.2f → eff=%.8f | threshold=%.2e | pending=%s",
            raw_increment, domain_weight, rep_factor, incentive_result.reward_fraction,
            effective_increment, self.consensus.state.threshold,
            {k: f"{v:.8f}" for k, v in self.consensus.state.pending_increments.items()},
        )

        # Record optimae
        optimae = Optimae(
            id=data.optimae_id,
            domain_id=data.domain_id,
            optimizer_id=sender_id,
            parameters=data.parameters,
            reported_performance=reported,
            verified_performance=verified,
            performance_increment=effective_increment,
        )
        self.consensus.record_optimae(optimae)

        # On-chain experiment metrics
        from doin_node.stats.chain_metrics import build_onchain_metrics
        onchain_metrics: dict[str, Any] = {}
        exp_state = self.experiment_tracker.get_experiment_state(data.domain_id)
        if exp_state is not None:
            onchain_metrics = build_onchain_metrics(
                experiment_id=exp_state["experiment_id"],
                round_number=exp_state["round_count"],
                time_to_this_result_seconds=time.monotonic() - exp_state["start_mono"],
                optimization_config=role.optimization_config if role else {},
                data_hash=None,
                previous_best_performance=exp_state["best_performance"],
                reported_performance=reported,
            )

        # Record OPTIMAE_ACCEPTED transaction on chain (include MAE breakdowns for chain history)
        cm = getattr(data, 'champion_metrics', None) or {}
        self.consensus.record_transaction(Transaction(
            tx_type=TransactionType.OPTIMAE_ACCEPTED,
            domain_id=data.domain_id,
            peer_id=sender_id,
            payload={
                "optimae_id": data.optimae_id,
                "parameters": data.parameters,
                "verified_performance": verified,
                "effective_increment": effective_increment,
                "reward_fraction": incentive_result.reward_fraction,
                "quorum_agree_fraction": 1.0,  # Auto-accepted
                "incentive_reason": "auto_accept_no_synthetic_validation",
                "val_mae": cm.get("val_mae"),
                "train_mae": cm.get("train_mae"),
                "val_naive_mae": cm.get("val_naive_mae"),
                "train_naive_mae": cm.get("train_naive_mae"),
                "test_mae": cm.get("test_mae"),
                "test_naive_mae": cm.get("test_naive_mae"),
                **onchain_metrics,
            },
        ))
        self.vuw.update_from_block([{
            "tx_type": "optimae_accepted",
            "domain_id": data.domain_id,
            "payload": {"increment": effective_increment},
        }])

        # Track contributor for coin distribution
        self._block_contributors.append(ContributorWork(
            peer_id=sender_id,
            role="optimizer",
            domain_id=data.domain_id,
            effective_increment=effective_increment,
            reward_fraction=incentive_result.reward_fraction,
        ))

        # Resolve optimae stake
        if self.fee_market:
            self.fee_market.resolve_optimae(data.optimae_id, accepted=True)

        # Update domain best
        current_best = self._domain_best.get(data.domain_id, (None, None))
        if self._is_better(data.domain_id, verified, current_best[1]):
            prev_best = current_best[1]
            self._domain_best[data.domain_id] = (data.parameters, verified)

            # Update champion metrics from originator (if attached to reveal)
            cm = getattr(data, 'champion_metrics', None) or {}
            if cm:
                self._domain_champion_metrics[data.domain_id] = {
                    "round": -1,
                    "performance": verified,
                    "parameters": data.parameters,
                    **{k: cm[k] for k in [
                        "val_mae", "train_mae", "val_naive_mae", "train_naive_mae",
                        "test_mae", "test_naive_mae",
                    ] if k in cm and cm[k] is not None},
                }

            is_remote = sender_id != self.peer_id
            if is_remote:
                logger.info(
                    "🏝️  MIGRATION: auto-accepted from peer %s for %s: %.6f → %.6f",
                    sender_id[:12], data.domain_id,
                    prev_best if prev_best is not None else float('-inf'),
                    verified,
                )
                # Inject into optimizer plugin's population
                plugin = self._optimizer_plugins.get(data.domain_id)
                if plugin and hasattr(plugin, "set_network_champion"):
                    plugin.set_network_champion(data.parameters)

        # Try to generate a block
        await self.try_generate_block()

    async def _handle_optimae_announcement(self, message: Message, from_peer: str) -> None:
        """Handle legacy direct announcement (no commit-reveal).

        Still supported for testing and non-adversarial networks.
        """
        from doin_core.protocol.messages import OptimaeAnnouncement
        data = OptimaeAnnouncement.model_validate(message.payload)

        task = Task(
            task_type=TaskType.OPTIMAE_VERIFICATION,
            domain_id=data.domain_id,
            requester_id=message.sender_id,
            parameters=data.parameters,
            optimae_id=data.optimae_id,
            reported_performance=data.reported_performance,
            priority=0,
        )
        self.task_queue.add(task)
        await self._flood_task_created(task)

    # ── Block handling ───────────────────────────────────────────

    async def _handle_block_announcement(self, message: Message, from_peer: str) -> None:
        ann = BlockAnnouncement.model_validate(message.payload)

        # Finality check — don't accept blocks that revert finalized state
        if ann.block_index <= self.finality.finalized_height:
            logger.warning("Ignoring block #%d — below finality", ann.block_index)
            return

        logger.info("Block #%d announced by %s", ann.block_index, from_peer[:12] if from_peer else "?")

        # If the announced block is ahead of us, trigger sync
        if ann.block_index >= self._get_height():
            # Find the peer endpoint that sent this
            peer_endpoint = self._find_peer_endpoint(from_peer, message.sender_id)
            if peer_endpoint:
                # Update peer status
                self.sync_manager.update_peer_status(peer_endpoint, ChainStatus(
                    chain_height=ann.block_index + 1,
                    tip_hash=ann.block_hash,
                    tip_index=ann.block_index,
                ))
                # Trigger async sync
                asyncio.create_task(self._sync_with_peer(peer_endpoint))

    # ── Candidate evaluation handling (research mode) ────────────

    async def _handle_candidate_evaluation(self, message: Message, from_peer: str) -> None:
        """Handle incoming candidate evaluation from a peer.

        Research mode: accept without verification, dedup by tx_id.
        """
        payload = message.payload or {}
        tx_id = payload.get("tx_id", "")
        if not tx_id:
            return

        # Dedup — skip if we've already seen this candidate evaluation
        if tx_id in self._seen_candidate_tx_ids:
            return
        self._seen_candidate_tx_ids.add(tx_id)

        # Skip our own broadcasts that come back via gossip
        if message.sender_id == self.peer_id:
            return

        # Reconstruct the transaction and record it as pending
        from doin_core.models.transaction import Transaction, TransactionType
        from datetime import datetime, timezone
        ts_str = payload.get("timestamp")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        tx = Transaction(
            id=tx_id,
            tx_type=TransactionType.CANDIDATE_EVALUATED,
            domain_id=payload.get("domain_id", ""),
            peer_id=payload.get("peer_id", message.sender_id),
            payload=payload.get("candidate_data", {}),
            timestamp=ts,
        )
        self.consensus.record_transaction(tx)
        logger.info(
            "📊 Accepted candidate eval from peer %s (gen=%s stage=%s fitness=%.6f)",
            message.sender_id[:12],
            payload.get("candidate_data", {}).get("generation", "?"),
            payload.get("candidate_data", {}).get("stage_name", "?"),
            payload.get("candidate_data", {}).get("fitness", 0.0),
        )

    async def _broadcast_candidate_evaluation(
        self, domain_id: str, tx_id: str, peer_id: str,
        candidate_data: dict, timestamp_iso: str,
    ) -> None:
        """Broadcast a candidate evaluation result to the network."""
        msg = Message(
            msg_type=MessageType.CANDIDATE_EVALUATION,
            sender_id=self.peer_id,
            payload={
                "tx_id": tx_id,
                "domain_id": domain_id,
                "peer_id": peer_id,
                "candidate_data": candidate_data,
                "timestamp": timestamp_iso,
            },
        )
        await self._broadcast(msg)

    # ── Task lifecycle (flooding) ────────────────────────────────

    async def _handle_task_created(self, message: Message, from_peer: str) -> None:
        tc = TaskCreated.model_validate(message.payload)
        if tc.task_id in self.task_queue.tasks:
            return
        task = Task(
            id=tc.task_id,
            task_type=TaskType(tc.task_type),
            domain_id=tc.domain_id,
            requester_id=tc.requester_id,
            parameters=tc.parameters,
            optimae_id=tc.optimae_id,
            reported_performance=tc.reported_performance,
            priority=tc.priority,
        )
        self.task_queue.add(task)

    async def _handle_task_claimed(self, message: Message, from_peer: str) -> None:
        tc = TaskClaimed.model_validate(message.payload)
        task = self.task_queue.tasks.get(tc.task_id)
        if task and task.status == TaskStatus.PENDING:
            task.claim(tc.evaluator_id)

    async def _handle_task_completed(self, message: Message, from_peer: str) -> None:
        tc = TaskCompleted.model_validate(message.payload)
        task = self.task_queue.tasks.get(tc.task_id)
        if not task or task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
            return

        task.complete(verified_performance=tc.verified_performance, result=tc.result)
        await self._process_task_completion(task, tc.evaluator_id, tc.verified_performance)

    # ================================================================
    # Task completion processing (with quorum + reputation)
    # ================================================================

    async def _process_task_completion(
        self,
        task: Task,
        evaluator_id: str,
        verified_performance: float | None,
        synthetic_data_hash: str = "",
    ) -> None:
        """Process a completed verification task through quorum consensus."""
        if task.task_type != TaskType.OPTIMAE_VERIFICATION:
            # Inference task — just record completion
            self.consensus.record_transaction(Transaction(
                tx_type=TransactionType.TASK_COMPLETED,
                domain_id=task.domain_id,
                peer_id=evaluator_id,
                payload={"task_id": task.id, "task_type": task.task_type.value},
            ))
            return

        if verified_performance is None or not task.optimae_id:
            return

        # Add vote to quorum (includes synthetic data hash for consensus)
        quorum_state = self.quorum.add_vote(
            task.optimae_id, evaluator_id, verified_performance,
            used_synthetic=True,
            synthetic_data_hash=synthetic_data_hash,
        )

        if quorum_state is None:
            return  # Not quorum yet, or not a selected evaluator

        # Quorum reached — evaluate
        result = self.quorum.evaluate_quorum(task.optimae_id)

        # Update reputation for all evaluators who voted
        for eval_id, agreed in result.agreements.items():
            self.reputation.record_evaluation_completed(eval_id, agreed)
            self._block_contributors.append(ContributorWork(
                peer_id=eval_id,
                role="evaluator",
                domain_id=task.domain_id,
                evaluations_completed=1,
                agreed_with_quorum=agreed,
            ))

        if result.accepted:
            # Optimae accepted by quorum — compute incentive-adjusted reward
            self.reputation.record_optimae_accepted(task.requester_id)

            # Get domain's incentive config
            role = self._domain_roles.get(task.domain_id)
            incentive_cfg = role.incentive_config if role else IncentiveConfig()

            # Use the incentive model to compute reward fraction
            # Each evaluator tested on DIFFERENT synthetic data, so
            # the median verified performance may be slightly different
            # from reported — the incentive model handles this gracefully
            reported = task.reported_performance or 0
            verified = result.median_performance or 0
            raw_increment = abs(reported)  # Base increment from the optimization

            rep_score = self.reputation.get_score(task.requester_id)
            import math
            rep_factor = min(1.0, math.log1p(rep_score) / math.log1p(10.0)) if rep_score > 0 else 0.0

            weights = self.vuw.compute_weights()
            domain_weight = weights.get(task.domain_id, 0.0)

            incentive_result = evaluate_verification_incentive(
                reported_performance=reported,
                verified_performance=verified,
                raw_increment=raw_increment,
                domain_weight=domain_weight,
                reputation_factor=rep_factor,
                config=incentive_cfg,
            )

            effective_increment = incentive_result.effective_increment

            optimae = Optimae(
                id=task.optimae_id,
                domain_id=task.domain_id,
                optimizer_id=task.requester_id,
                parameters=task.parameters,
                reported_performance=reported,
                verified_performance=verified,
                performance_increment=effective_increment,
            )
            self.consensus.record_optimae(optimae)

            # Build experiment context for on-chain storage
            from doin_node.stats.chain_metrics import build_onchain_metrics
            onchain_metrics: dict[str, Any] = {}
            exp_state = self.experiment_tracker.get_experiment_state(task.domain_id)
            if exp_state is not None:
                role = self._domain_roles.get(task.domain_id)
                onchain_metrics = build_onchain_metrics(
                    experiment_id=exp_state["experiment_id"],
                    round_number=exp_state["round_count"],
                    time_to_this_result_seconds=time.monotonic() - exp_state["start_mono"],
                    optimization_config=role.optimization_config if role else {},
                    data_hash=None,
                    previous_best_performance=exp_state["best_performance"],
                    reported_performance=reported,
                )

            self.consensus.record_transaction(Transaction(
                tx_type=TransactionType.OPTIMAE_ACCEPTED,
                domain_id=task.domain_id,
                peer_id=task.requester_id,
                payload={
                    "optimae_id": task.optimae_id,
                    "parameters": task.parameters,  # On-chain for island model sync
                    "verified_performance": verified,
                    "effective_increment": effective_increment,
                    "reward_fraction": incentive_result.reward_fraction,
                    "quorum_agree_fraction": result.agree_fraction,
                    "incentive_reason": incentive_result.reason,
                    **onchain_metrics,
                },
            ))
            self.vuw.update_from_block([{
                "tx_type": "optimae_accepted",
                "domain_id": task.domain_id,
                "payload": {"increment": effective_increment},
            }])

            # Track contributor for coin distribution
            self._block_contributors.append(ContributorWork(
                peer_id=task.requester_id,
                role="optimizer",
                domain_id=task.domain_id,
                effective_increment=effective_increment,
                reward_fraction=incentive_result.reward_fraction,
            ))

            # Resolve optimae stake (full refund on accept)
            if self.fee_market:
                self.fee_market.resolve_optimae(task.optimae_id, accepted=True)

            # Update domain best from accepted optimae (critical for island model!)
            # This ensures nodes pick up champions from OTHER nodes, not just their own.
            current_best = self._domain_best.get(task.domain_id, (None, None))
            if self._is_better(task.domain_id, verified, current_best[1]):
                prev_best = current_best[1]
                self._domain_best[task.domain_id] = (task.parameters, verified)
                is_remote = task.requester_id != self.peer_id
                if is_remote:
                    logger.info(
                        "🏝️  MIGRATION: adopted champion from peer %s for %s: %.6f → %.6f (Δ%.6f)",
                        task.requester_id[:12], task.domain_id,
                        prev_best if prev_best is not None else float('-inf'),
                        verified,
                        verified - (prev_best or 0),
                    )
                else:
                    logger.info(
                        "⚡ Domain %s new local best: %.6f (from our own optimae)",
                        task.domain_id, verified,
                    )

            logger.info(
                "Optimae %s ACCEPTED (median=%.4f, reward=%.2f, eff=%.4f, rep=%.2f) — %s",
                task.optimae_id[:12],
                verified,
                incentive_result.reward_fraction,
                effective_increment,
                rep_score,
                incentive_result.reason,
            )
        else:
            # Rejected — partial stake burn
            if self.fee_market:
                self.fee_market.resolve_optimae(task.optimae_id, accepted=False)
            self.reputation.record_optimae_rejected(task.requester_id)
            self.consensus.record_transaction(Transaction(
                tx_type=TransactionType.OPTIMAE_REJECTED,
                domain_id=task.domain_id,
                peer_id=task.requester_id,
                payload={
                    "optimae_id": task.optimae_id,
                    "reason": result.reason,
                },
            ))
            logger.info(
                "Optimae %s REJECTED: %s",
                task.optimae_id[:12], result.reason,
            )

            # ── Push current champion to the rejected optimizer ──
            # Quorum rejected their result — send the best known solution
            # so they can immediately update their population baseline.
            if task.requester_id != self.peer_id:
                current_best = self._domain_best.get(task.domain_id, (None, None))
                if current_best[0] is not None:
                    champion_msg = Message(
                        msg_type=MessageType.CHAMPION_RESPONSE,
                        sender_id=self.peer_id,
                        payload={
                            "domain_id": task.domain_id,
                            "request_id": f"quorum-reject-push-{task.optimae_id[:12]}",
                            "parameters": current_best[0],
                            "performance": current_best[1],
                            "has_champion": True,
                        },
                    )
                    # Find the requester's endpoint — they may be a known peer
                    endpoint = self._find_peer_endpoint_by_id(task.requester_id)
                    if endpoint:
                        try:
                            await self.transport.send(endpoint, champion_msg)
                            logger.info(
                                "📡 Pushed champion to quorum-rejected sender %s for %s (best=%.6f)",
                                task.requester_id[:12], task.domain_id, current_best[1],
                            )
                        except Exception:
                            logger.debug("Failed to push champion to %s", task.requester_id[:12])

        # Record task completion
        self.consensus.record_transaction(Transaction(
            tx_type=TransactionType.TASK_COMPLETED,
            domain_id=task.domain_id,
            peer_id=evaluator_id,
            payload={"task_id": task.id, "task_type": task.task_type.value},
        ))

    # ================================================================
    # Champion sync (island model — fetch best on startup)
    # ================================================================

    async def _handle_champion_request(self, message: Message, from_peer: str) -> None:
        """Respond to a champion request with our current domain best."""
        domain_id = message.payload.get("domain_id", "")
        request_id = message.payload.get("request_id", "")
        current_best = self._domain_best.get(domain_id, (None, None))
        params, perf = current_best
        resp = Message(
            msg_type=MessageType.CHAMPION_RESPONSE,
            sender_id=self.peer_id,
            payload={
                "domain_id": domain_id,
                "request_id": request_id,
                "parameters": params,
                "performance": perf,
                "has_champion": params is not None and perf is not None,
                "champion_metrics": self._domain_champion_metrics.get(domain_id),
            },
        )
        # Send directly to requester (not broadcast)
        endpoint = self._find_peer_endpoint(from_peer, message.sender_id)
        if endpoint:
            try:
                await self.transport.send(endpoint, resp)
            except Exception:
                logger.debug("Failed to send champion response to %s", endpoint)

    async def _handle_champion_response(self, message: Message, from_peer: str) -> None:
        """Handle a champion response from a peer — update domain best if better."""
        payload = message.payload
        domain_id = payload.get("domain_id", "")
        if not payload.get("has_champion"):
            return
        perf = payload.get("performance")
        params = payload.get("parameters")
        if perf is None or params is None:
            return

        current_best = self._domain_best.get(domain_id, (None, None))
        if self._is_better(domain_id, perf, current_best[1]):
            self._domain_best[domain_id] = (params, perf)
            logger.info(
                "🏝️  Champion synced from peer %s for %s: perf=%.6f",
                message.sender_id[:12], domain_id, perf,
            )
            # Update champion metrics from peer (authoritative from originator)
            cm = payload.get("champion_metrics")
            if cm and isinstance(cm, dict):
                self._domain_champion_metrics[domain_id] = {
                    "performance": perf,
                    **{k: cm[k] for k in [
                        "round", "val_mae", "train_mae", "val_naive_mae",
                        "train_naive_mae", "test_mae", "test_naive_mae",
                    ] if k in cm and cm[k] is not None},
                }
            # Also inject into optimizer plugin if running
            plugin = self._optimizer_plugins.get(domain_id)
            if plugin and hasattr(plugin, "set_network_champion"):
                plugin.set_network_champion(params)

    async def _request_champions_from_peers(self) -> None:
        """Request current best champion for each domain from all connected peers."""
        import uuid
        for domain_id in self.optimizer_domains:
            request_id = str(uuid.uuid4())[:8]
            msg = Message(
                msg_type=MessageType.CHAMPION_REQUEST,
                sender_id=self.peer_id,
                payload={
                    "domain_id": domain_id,
                    "request_id": request_id,
                },
            )
            await self._broadcast(msg)
            logger.info("📡 Requested champion for %s from peers", domain_id)

        # Wait a bit for responses to arrive
        await asyncio.sleep(3)

    # ================================================================
    # Block generation (with finality + anchoring)
    # ================================================================

    async def try_generate_block(self) -> Block | None:
        can = self.consensus.can_generate_block()
        ws = sum(self.consensus.state.pending_increments.values())
        logger.info(
            "🔨 try_generate_block: can=%s weighted_sum=%.8f threshold=%.2e pending=%s",
            can, ws, self.consensus.state.threshold,
            {k: f"{v:.8f}" for k, v in self.consensus.state.pending_increments.items()},
        )
        if not can:
            return None

        tip = self._get_tip()
        if tip is None:
            return None

        block = self.consensus.generate_block(tip, self.peer_id)
        if block is None:
            return None

        # Compute and apply coinbase (block reward distribution)
        coinbase = distribute_block_reward(
            block_index=block.header.index,
            generator_id=self.peer_id,
            contributors=list(self._block_contributors),
        )
        self.balance_tracker.apply_coinbase(coinbase)
        self._block_contributors.clear()

        # Adjust fee market base fee based on block fullness
        if self.fee_market:
            self.fee_market.adjust_base_fee(len(block.transactions))

        # Update difficulty controller
        self.difficulty.on_new_block(
            block.header.index,
            block.header.timestamp.timestamp(),
        )
        # Apply new threshold to consensus
        self.consensus.state.threshold = self.difficulty.threshold

        self._append_block(block)
        self._save_chain()

        # Update finality
        depth_block_hash = None
        target_height = block.header.index - self.finality.confirmation_depth
        if target_height >= 0:
            b = self._get_block(target_height)
            if b:
                depth_block_hash = b.hash
        self.finality.on_new_block(block.header.index, depth_block_hash)

        # State snapshot at intervals (SQLite only)
        if self.chaindb and block.header.index % self.config.snapshot_interval == 0:
            self.chaindb.save_snapshot(
                block_index=block.header.index,
                block_hash=block.hash,
                balances=self.balance_tracker.all_balances,
                reputation=self.reputation.all_scores,
                domain_stats={d: self.vuw.compute_weights().get(d, 0) for d in self._domains},
            )
            logger.info("State snapshot saved at block #%d", block.header.index)

            # Prune old transaction bodies
            if self.config.prune_keep_blocks > 0:
                prune_before = block.header.index - self.config.prune_keep_blocks
                if prune_before > 0:
                    self.chaindb.prune_transactions_before(prune_before)

        # External anchoring
        if self.anchor_manager.should_anchor(block.header.index):
            # For SQLite backend, get block hashes efficiently
            if self.chaindb:
                blocks_range = self.chaindb.get_blocks_range(0, block.header.index)
                block_hashes = [b.hash for b in blocks_range]
            else:
                block_hashes = [b.hash for b in self.chain.blocks]
            state_hash = self.anchor_manager.compute_chain_state_hash(block_hashes)
            self.anchor_manager.create_anchor(
                block.header.index, block.hash, state_hash,
            )
            logger.info("External anchor created at block #%d", block.header.index)

        # Announce via configured protocol
        announcement = BlockAnnouncement(
            block_index=block.header.index,
            block_hash=block.hash,
            previous_hash=block.header.previous_hash,
            generator_id=self.peer_id,
            transaction_count=len(block.transactions),
            weighted_performance_sum=block.header.weighted_performance_sum,
            threshold=block.header.threshold,
        )
        msg = Message(
            msg_type=MessageType.BLOCK_ANNOUNCEMENT,
            sender_id=self.peer_id,
            payload=json.loads(announcement.model_dump_json()),
        )
        await self._broadcast(msg)

        # Update sync manager state
        height = self._get_height()
        self.sync_manager.update_our_state(
            height, block.hash, self.finality.finalized_height,
        )

        logger.info("Block #%d generated (hash=%s)", block.header.index, block.hash[:12])
        return block

    # ================================================================
    # Optimizer loop (when this node optimizes)
    # ================================================================

    async def _optimizer_loop(self) -> None:
        """Background loop: run FULL DEAP GA optimization for each domain.

        Each domain runs the complete predictor optimizer (DEAP GA with
        incremental stages, populations, generations). DOIN callbacks
        handle champion broadcasting and evaluation service during the run.
        """
        # Wait for LAN discovery + peer connections before starting optimization
        logger.info("Optimizer waiting for peer discovery...")
        try:
            for i in range(30):  # Up to 30s
                await asyncio.sleep(1)
                if self._peers:
                    logger.info("Optimizer: peers found (%d) after %ds, waiting for mesh...", len(self._peers), i + 1)
                    # Give a moment for gossip mesh to form
                    await asyncio.sleep(3)
                    break

            # Request current best champion from peers (island model sync)
            if self._peers:
                logger.info("Optimizer: requesting champions from %d peers...", len(self._peers))
                await self._request_champions_from_peers()
            else:
                logger.info("No peers found — starting optimization from scratch")
        except Exception:
            logger.exception("Optimizer loop error during peer discovery/champion request")

        logger.info(
            "Optimizer loop: domains=%s registered_plugins=%s converged=%s",
            self.optimizer_domains,
            list(self._optimizer_plugins.keys()),
            list(self._domain_converged),
        )
        for domain_id in self.optimizer_domains:
            if domain_id in self._domain_converged:
                logger.warning("Skipping %s — already converged", domain_id)
                continue
            plugin = self._optimizer_plugins.get(domain_id)
            if plugin is None:
                logger.error("No optimizer plugin registered for %s — skipping!", domain_id)
                continue

            # Wire up DOIN callbacks on the plugin
            self._setup_optimizer_callbacks(domain_id, plugin)

            try:
                logger.info("🧬 Starting optimization for %s", domain_id)
                await self._run_full_optimization(domain_id, plugin)
                logger.info("✅ Optimization complete for %s", domain_id)
                self._log_event("optimization_complete", domain_id=domain_id)
            except Exception:
                logger.exception("Optimization error for %s", domain_id)

    def _setup_optimizer_callbacks(self, domain_id: str, plugin: Any) -> None:
        """Wire DOIN callbacks onto the optimizer plugin."""
        if not hasattr(plugin, "set_local_champion_callback"):
            return  # Not a DOIN-aware plugin

        loop = asyncio.get_event_loop()
        node = self  # Capture for closures

        def on_local_champion(params, fitness, metrics, gen, stage_info):
            """Called from optimizer thread when new local champion found → broadcast if better than network.
            
            Also triggers network-wide stage advance if the cumulative improvement
            within the current stage exceeds the configured threshold.
            """
            import asyncio as _aio

            # Guard: only broadcast if this local champion actually beats the current
            # network best.  Without this check every node floods the network with its
            # local improvements even when peers already hold a superior solution,
            # causing a cascade of auto-rejections and wasted bandwidth.
            current_net_best = node._domain_best.get(domain_id, (None, None))
            if not node._is_better(domain_id, fitness, current_net_best[1]):
                net_params, net_perf = current_net_best
                logger.info(
                    "[%s] Local champion (%.6f) does not beat network best (%.6f)"
                    " — skipping broadcast, injecting network champion",
                    domain_id, fitness, net_perf,
                )
                # Inject the superior network champion so the optimizer benefits immediately
                if net_params is not None:
                    p = node._optimizer_plugins.get(domain_id)
                    if p and hasattr(p, "set_network_champion"):
                        p.set_network_champion(net_params)
                return

            # Local champion is genuinely better than the known network best → broadcast
            fut = _aio.run_coroutine_threadsafe(
                node._broadcast_champion(domain_id, params, fitness, metrics, gen, stage_info),
                loop,
            )
            try:
                fut.result(timeout=30)  # Block optimizer thread until broadcast completes
            except Exception as e:
                logger.warning("Champion broadcast failed: %s", e)

            # ── Stage-sync: trigger network-wide stage advance on significant improvement ──
            # If the fitness improvement since stage start exceeds the threshold,
            # broadcast STAGE_COMPLETE so ALL nodes advance to the next stage.
            stage_advance_threshold = node.config.get("stage_advance_threshold", 1e-5)
            stage_start_fit = node._domain_stage_start_fitness.get(domain_id)
            current_stage = stage_info.get("stage", 1)
            total_stages = stage_info.get("total_stages", 1)

            # Initialize stage-start fitness if not set yet
            if stage_start_fit is None:
                node._domain_stage_start_fitness[domain_id] = fitness
                logger.info(
                    "[%s] Stage %d: initialized stage-start fitness to %.6f",
                    domain_id, current_stage, fitness,
                )
            else:
                # Lower is better for predictor NDA
                improvement = stage_start_fit - fitness
                if improvement > stage_advance_threshold:
                    logger.info(
                        "🚀 [%s] Stage %d: improvement %.6f > threshold %.6f"
                        " — triggering network-wide stage advance!",
                        domain_id, current_stage, improvement, stage_advance_threshold,
                    )
                    # Broadcast stage complete to all peers
                    stage_metrics = {k: v for k, v in metrics.items() if k != "_model_b64" and k != "fitness"}
                    fut2 = _aio.run_coroutine_threadsafe(
                        node._broadcast_stage_complete(
                            domain_id, current_stage, total_stages,
                            params, fitness, stage_metrics,
                        ),
                        loop,
                    )
                    try:
                        fut2.result(timeout=30)
                    except Exception as e:
                        logger.warning("Stage complete broadcast failed: %s", e)

                    # NOTE: Do NOT call force_stage_advance() on our own optimizer.
                    # The local optimizer must complete its stage naturally (via
                    # patience / n_generations).  Only *remote* STAGE_COMPLETE
                    # messages should trigger an early stage advance so that each
                    # node explores its own stage fully before being forced ahead.

        def on_eval_service(gen, candidate_num, stage_info):
            """Called from optimizer thread between candidates → process 1 pending eval + log event + request champion."""
            import asyncio as _aio
            fut = _aio.run_coroutine_threadsafe(
                node._process_one_pending_eval(domain_id),
                loop,
            )
            try:
                fut.result(timeout=120)  # Eval can take a while
            except Exception as e:
                logger.debug("Eval service between candidates: %s", e)

            # Log per-candidate evaluation event (for dashboard)
            role = node._domain_roles.get(domain_id)
            opt_cfg = role.optimization_config if role else {}
            pat = node._domain_patience.get(domain_id, (0, 0))
            # Use patience from stage_info (optimizer sends it per-candidate)
            si_patience = stage_info.get("patience", 0)
            si_no_improve = stage_info.get("no_improve_counter", 0)
            eff_no_improve = si_no_improve if si_no_improve else pat[0]
            eff_patience = si_patience if si_patience else (pat[1] or opt_cfg.get("optimization_patience", 0))

            # Store current candidate state for /api/candidate (local, per-machine)
            _cand_params = stage_info.get("candidate_params")
            _model_summary = stage_info.get("model_summary")
            node._current_candidate = {
                "domain_id": domain_id,
                "gen": gen,
                "candidate_num": stage_info.get("candidate_num"),
                "total_candidates": stage_info.get("total_candidates"),
                "stage": stage_info.get("stage"),
                "total_stages": stage_info.get("total_stages"),
                "stage_name": stage_info.get("stage_name"),
                "total_evals": stage_info.get("total_candidates_evaluated"),
                "fitness": stage_info.get("fitness"),
                "val_mae": stage_info.get("val_mae"),
                "train_mae": stage_info.get("train_mae"),
                "val_naive_mae": stage_info.get("val_naive_mae"),
                "train_naive_mae": stage_info.get("train_naive_mae"),
                "champion_fitness": stage_info.get("champion_fitness"),
                "candidate_params": _cand_params,
                "model_summary": _model_summary,
                "n_generations": stage_info.get("n_generations_total") or opt_cfg.get("n_generations", 15),
                "n_generations_stage": stage_info.get("n_generations_stage"),
                "gen_in_stage": stage_info.get("gen_in_stage"),
                "no_improve_counter": eff_no_improve,
                "optimization_patience": eff_patience,
                "neat_species_count": stage_info.get("neat_species_count"),
                "neat_complexity": stage_info.get("neat_complexity"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            node._log_event("candidate_eval",
                domain_id=domain_id, gen=gen,
                candidate_num=stage_info.get("candidate_num"),
                total_candidates=stage_info.get("total_candidates"),
                stage=stage_info.get("stage"), total_stages=stage_info.get("total_stages"),
                stage_name=stage_info.get("stage_name"),
                total_evals=stage_info.get("total_candidates_evaluated"),
                fitness=stage_info.get("fitness"),
                val_mae=stage_info.get("val_mae"), train_mae=stage_info.get("train_mae"),
                val_naive_mae=stage_info.get("val_naive_mae"),
                train_naive_mae=stage_info.get("train_naive_mae"),
                champion_fitness=stage_info.get("champion_fitness"),
                n_generations=stage_info.get("n_generations_total") or opt_cfg.get("n_generations", 15),
                n_generations_stage=stage_info.get("n_generations_stage"),
                gen_in_stage=stage_info.get("gen_in_stage"),
                no_improve_counter=eff_no_improve, optimization_patience=eff_patience,
                neat_species_count=stage_info.get("neat_species_count"),
                neat_complexity=stage_info.get("neat_complexity"))

            # Fire-and-forget champion request: keeps network champion fresh
            # so by the next generation-start injection the node has the latest optimum.
            _aio.run_coroutine_threadsafe(
                node._request_champions_from_peers(),
                loop,
            )  # result NOT awaited — non-blocking

        def on_generation_end(population, hof, hyper_keys, gen, stage_info, stats):
            """Called from optimizer thread at end of each generation."""
            # Update tracking
            node._domain_round_count[domain_id] = gen + 1
            # Log generation summary
            champ_fit = stage_info.get("champion_fitness")
            champ_val = stage_info.get("champion_val_mae")
            champ_train = stage_info.get("champion_train_mae")
            champ_val_naive = stage_info.get("champion_naive_mae")
            champ_train_naive = stage_info.get("champion_train_naive_mae")
            avg_fit = stage_info.get("avg_fitness")
            stage = stage_info.get("stage", 1)
            total_stages = stage_info.get("total_stages", 1)
            n_evals = stage_info.get("total_candidates_evaluated", 0)
            no_improve = stage_info.get('no_improve_counter', 0)
            pat_max = stage_info.get('patience', 0)
            node._domain_patience[domain_id] = (no_improve, pat_max)
            patience_str = f"{no_improve}/{pat_max}"
            role = node._domain_roles.get(domain_id)
            opt_cfg = role.optimization_config if role else {}
            n_gens = stage_info.get("n_generations_total") or opt_cfg.get("n_generations", 15)
            stage_name = stage_info.get("stage_name")

            node._log_event("generation_end",
                domain_id=domain_id, gen=gen, stage=stage, total_stages=total_stages,
                stage_name=stage_name,
                total_evals=n_evals,
                champion_fitness=champ_fit, champion_val_mae=champ_val,
                champion_train_mae=champ_train,
                champion_val_naive_mae=champ_val_naive,
                champion_train_naive_mae=champ_train_naive,
                avg_fitness=avg_fit, patience=patience_str,
                n_generations=n_gens,
                n_generations_stage=stage_info.get("n_generations_stage"),
                gen_in_stage=stage_info.get("gen_in_stage"),
                no_improve_counter=no_improve, optimization_patience=pat_max,
                neat_species_count=stage_info.get("neat_species_count"),
                neat_avg_complexity=stage_info.get("neat_avg_complexity"),
                neat_species_details=stage_info.get("neat_species_details"))

            logger.info(
                "[%s] gen=%d stage=%d/%d evals=%d  champ_fitness=%.6f  champ_val_mae=%s  avg_fitness=%.6f  patience=%s",
                domain_id, gen, stage, total_stages, n_evals,
                champ_fit if champ_fit is not None else float("nan"),
                f"{champ_val:.6f}" if champ_val is not None else "N/A",
                avg_fit if avg_fit is not None else float("nan"),
                patience_str,
            )

            # Record to experiment tracker
            champ_test = stage_info.get("champion_test_mae")
            champ_test_naive = stage_info.get("champion_test_naive_mae")
            champ_params = stage_info.get("champion_parameters", {})
            detail_metrics = {
                "generation": gen,
                "stage": stage,
                "total_stages": total_stages,
                "stage_name": stage_name or "",
                "gen_in_stage": stage_info.get("gen_in_stage", ""),
                "n_generations_stage": stage_info.get("n_generations_stage", ""),
                "n_generations_total": stage_info.get("n_generations_total", ""),
                "total_candidates_evaluated": n_evals,
                "population_size": stage_info.get("population_size", ""),
                "no_improve_counter": no_improve,
                "optimization_patience": pat_max,
                "avg_fitness": avg_fit,
                "best_fitness_gen": stage_info.get("best_fitness_gen", ""),
                "neat_species_count": stage_info.get("neat_species_count", ""),
                "neat_avg_complexity": stage_info.get("neat_avg_complexity", ""),
                "train_mae": champ_train,
                "train_naive_mae": champ_train_naive,
                "val_mae": champ_val,
                "val_naive_mae": champ_val_naive,
                "test_mae": champ_test,
                "test_naive_mae": champ_test_naive,
            }
            try:
                node.experiment_tracker.record_round(
                    domain_id,
                    performance=champ_fit if champ_fit is not None else 0.0,
                    parameters=champ_params,
                    wall_clock_seconds=0,
                    chain_height=node._get_height(),
                    peers_count=len(node._peers),
                    detail_metrics=detail_metrics,
                )
            except Exception:
                pass

        def on_stage_start(stage, total_stages):
            """Called from optimizer thread when a new stage begins → request champion from peers
            and initialize stage-start fitness tracking for the stage-advance threshold."""
            import asyncio as _aio
            logger.info(
                "🔄 Stage %d/%d starting for %s — requesting champion from peers",
                stage, total_stages, domain_id,
            )
            # Reset stage-start fitness for the new stage
            current_best = node._domain_best.get(domain_id, (None, None))
            if current_best[1] is not None:
                node._domain_stage_start_fitness[domain_id] = current_best[1]
                logger.info(
                    "[%s] Stage %d: tracking stage-start fitness = %.6f",
                    domain_id, stage, current_best[1],
                )
            else:
                # First stage, no best yet — will be set when first champion is found
                node._domain_stage_start_fitness.pop(domain_id, None)

            fut = _aio.run_coroutine_threadsafe(
                node._request_champions_from_peers(),
                loop,
            )
            try:
                fut.result(timeout=10)  # Wait for responses
            except Exception as e:
                logger.debug("Stage champion request failed: %s", e)

        plugin.set_local_champion_callback(on_local_champion)
        plugin.set_eval_service_callback(on_eval_service)
        plugin.set_generation_end_callback(on_generation_end)
        if hasattr(plugin, "set_stage_start_callback"):
            plugin.set_stage_start_callback(on_stage_start)

        # Stage-end broadcast: called when a stage completes so all nodes advance together
        def on_stage_end_broadcast(stage, total_stages, champion_params, champion_fitness, metrics):
            """Called from optimizer thread when a stage completes → broadcast to peers."""
            # Update plugin's _current_stage so stale-guard rejects late arrivals
            # for this stage (the optimizer has now moved past it).
            if hasattr(plugin, "_current_stage"):
                plugin._current_stage = stage
            # Reset dashboard patience for the new stage
            node._domain_patience[domain_id] = (0, 0)
            import asyncio as _aio
            fut = _aio.run_coroutine_threadsafe(
                node._broadcast_stage_complete(domain_id, stage, total_stages,
                                               champion_params, champion_fitness, metrics),
                loop,
            )
            try:
                fut.result(timeout=30)
            except Exception as e:
                logger.warning("Stage complete broadcast failed: %s", e)

        if hasattr(plugin, "set_stage_end_callback"):
            plugin.set_stage_end_callback(on_stage_end_broadcast)

        # Per-candidate evaluation tracking: record every candidate to experiment tracker + blockchain
        def on_candidate_evaluated(candidate_info):
            """Called from optimizer thread after each candidate finishes training.

            1. Records to local experiment_tracker (OLAP)
            2. Creates a CANDIDATE_EVALUATED transaction for the blockchain
            3. Broadcasts to peers so all nodes accumulate all results
            """
            try:
                detail_metrics = {
                    "generation": candidate_info.get("generation", ""),
                    "stage": candidate_info.get("stage", ""),
                    "total_stages": candidate_info.get("total_stages", ""),
                    "stage_name": candidate_info.get("stage_name", ""),
                    "gen_in_stage": candidate_info.get("gen_in_stage", ""),
                    "n_generations_stage": candidate_info.get("n_generations_stage", ""),
                    "n_generations_total": candidate_info.get("n_generations_total", ""),
                    "total_candidates_evaluated": candidate_info.get("total_eval", ""),
                    "population_size": candidate_info.get("population_size", ""),
                    "no_improve_counter": candidate_info.get("no_improve_counter", ""),
                    "optimization_patience": candidate_info.get("optimization_patience", ""),
                    "avg_fitness": "",
                    "best_fitness_gen": candidate_info.get("champion_fitness", ""),
                    "neat_species_count": candidate_info.get("neat_species_count", ""),
                    "neat_avg_complexity": candidate_info.get("neat_avg_complexity", ""),
                    "train_mae": candidate_info.get("train_mae"),
                    "train_naive_mae": candidate_info.get("train_naive_mae"),
                    "val_mae": candidate_info.get("val_mae"),
                    "val_naive_mae": candidate_info.get("val_naive_mae"),
                    "test_mae": candidate_info.get("test_mae"),
                    "test_naive_mae": candidate_info.get("test_naive_mae"),
                }
                node.experiment_tracker.record_round(
                    domain_id,
                    performance=candidate_info.get("fitness", 0.0),
                    parameters=candidate_info.get("parameters", {}),
                    wall_clock_seconds=0,
                    chain_height=node._get_height(),
                    peers_count=len(node._peers),
                    detail_metrics=detail_metrics,
                )
            except Exception as e:
                logger.debug("Candidate tracking error: %s", e)

            # --- Blockchain: create transaction + broadcast to peers ---
            try:
                from doin_core.models.transaction import Transaction, TransactionType
                import asyncio as _aio

                candidate_data = {
                    "generation": candidate_info.get("generation"),
                    "candidate_in_gen": candidate_info.get("candidate_in_gen"),
                    "total_eval": candidate_info.get("total_eval"),
                    "stage": candidate_info.get("stage"),
                    "total_stages": candidate_info.get("total_stages"),
                    "stage_name": candidate_info.get("stage_name"),
                    "gen_in_stage": candidate_info.get("gen_in_stage"),
                    "n_generations_stage": candidate_info.get("n_generations_stage"),
                    "n_generations_total": candidate_info.get("n_generations_total"),
                    "population_size": candidate_info.get("population_size"),
                    "species_id": candidate_info.get("species_id"),
                    "complexity": candidate_info.get("complexity"),
                    "is_champion": candidate_info.get("is_champion"),
                    "fitness": candidate_info.get("fitness"),
                    "champion_fitness": candidate_info.get("champion_fitness"),
                    "parameters": candidate_info.get("parameters", {}),
                    "champion_parameters": candidate_info.get("champion_parameters", {}),
                    "train_mae": candidate_info.get("train_mae"),
                    "train_naive_mae": candidate_info.get("train_naive_mae"),
                    "val_mae": candidate_info.get("val_mae"),
                    "val_naive_mae": candidate_info.get("val_naive_mae"),
                    "test_mae": candidate_info.get("test_mae"),
                    "test_naive_mae": candidate_info.get("test_naive_mae"),
                    "no_improve_counter": candidate_info.get("no_improve_counter"),
                    "optimization_patience": candidate_info.get("optimization_patience"),
                    "neat_species_count": candidate_info.get("neat_species_count"),
                    "neat_avg_complexity": candidate_info.get("neat_avg_complexity"),
                }

                tx = Transaction(
                    tx_type=TransactionType.CANDIDATE_EVALUATED,
                    domain_id=domain_id,
                    peer_id=node.peer_id,
                    payload=candidate_data,
                )
                # Dedup locally
                if tx.id not in node._seen_candidate_tx_ids:
                    node._seen_candidate_tx_ids.add(tx.id)
                    node.consensus.record_transaction(tx)
                    # Broadcast to peers (async from sync thread)
                    fut = _aio.run_coroutine_threadsafe(
                        node._broadcast_candidate_evaluation(
                            domain_id, tx.id, node.peer_id,
                            candidate_data, tx.timestamp.isoformat(),
                        ),
                        loop,
                    )
                    try:
                        fut.result(timeout=10)
                    except Exception as e:
                        logger.debug("Candidate eval broadcast error: %s", e)
            except Exception as e:
                logger.debug("Candidate blockchain error: %s", e)

        if hasattr(plugin, "set_candidate_evaluated_callback"):
            plugin.set_candidate_evaluated_callback(on_candidate_evaluated)

    async def _run_full_optimization(self, domain_id: str, plugin: Any) -> None:
        """Run the full DEAP GA optimization in an executor thread."""
        current_best = self._domain_best.get(domain_id, (None, None))

        raw = await asyncio.get_event_loop().run_in_executor(
            None, lambda: plugin.optimize(current_best[0], current_best[1]),
        )

        if raw is None:
            return

        # Final result
        if isinstance(raw, tuple):
            parameters, performance = raw
        elif isinstance(raw, dict):
            parameters = raw.get("parameters", {})
            performance = raw.get("performance", 0.0)
        else:
            return

        # Update domain best
        current_best = self._domain_best.get(domain_id, (None, None))
        if self._is_better(domain_id, performance, current_best[1]):
            self._domain_best[domain_id] = (parameters, performance)

        logger.info(
            "🏁 Full optimization finished for %s: final_perf=%.6f",
            domain_id, performance,
        )

    async def _broadcast_champion(
        self, domain_id: str, params: dict, fitness: float,
        metrics: dict, gen: int, stage_info: dict,
    ) -> None:
        """Broadcast a new local champion to the DOIN network (commit→reveal)."""
        performance = fitness  # Use raw fitness; comparison respects higher_is_better per domain

        # Include trained model in broadcast parameters for evaluator verification (inference-only)
        model_b64 = metrics.pop("_model_b64", None)
        if model_b64:
            params = {**params, "_model_b64": model_b64}

        # Check if this is an improvement (for logging); do NOT update
        # _domain_best here — the reveal handler (_auto_accept_optimae) will
        # update it after proper validation. Updating prematurely causes the
        # reveal handler's _is_better() check to compare perf against itself
        # and always return False, preventing block creation.
        current_best = self._domain_best.get(domain_id, (None, None))
        is_improvement = self._is_better(domain_id, performance, current_best[1])

        # Store champion metrics
        self._domain_champion_metrics[domain_id] = {
            "round": gen,
            "performance": performance,
            "parameters": params,
            **{k: v for k, v in metrics.items() if k != "fitness"},
        }

        self._log_event("champion",
            domain_id=domain_id, gen=gen, stage=stage_info.get("stage", 1),
            performance=performance,
            val_mae=metrics.get("val_mae"), train_mae=metrics.get("train_mae"),
            test_mae=metrics.get("test_mae"),
            val_naive_mae=metrics.get("val_naive_mae"), train_naive_mae=metrics.get("train_naive_mae"),
            test_naive_mae=metrics.get("test_naive_mae"),
            is_improvement=is_improvement)

        logger.info(
            "🏆 New champion [%s] gen=%d stage=%d  perf=%.6f  val_mae=%.6f  train_mae=%.6f  test_mae=%s",
            domain_id, gen, stage_info.get("stage", 1), performance,
            metrics.get("val_mae", float("nan")),
            metrics.get("train_mae", float("nan")),
            f"{metrics['test_mae']:.6f}" if metrics.get("test_mae") is not None else "N/A",
        )

        # Commit→Reveal flow
        nonce = secrets.token_hex(16)
        commitment_hash = compute_commitment(params, nonce)

        # Phase 1: Commit
        commit_msg = Message(
            msg_type=MessageType.OPTIMAE_COMMIT,
            sender_id=self.peer_id,
            payload=json.loads(OptimaeCommit(
                commitment_hash=commitment_hash,
                domain_id=domain_id,
            ).model_dump_json()),
        )
        # Register the commit locally FIRST so the self-reveal can be validated.
        # Broadcast AFTER to prevent a peer-relay race from consuming the commitment.
        await self._handle_optimae_commit(commit_msg, "self")
        await self._broadcast(commit_msg)
        await asyncio.sleep(2.0)

        # Phase 2: Reveal
        optimae_id = f"opt-{self.peer_id[:8]}-{int(time.time())}"

        if self.fee_market:
            stake = self.fee_market.get_suggested_fee()["optimae_stake"]
            self.fee_market.stake_for_optimae(optimae_id, stake)

        reveal_msg = Message(
            msg_type=MessageType.OPTIMAE_REVEAL,
            sender_id=self.peer_id,
            payload=json.loads(OptimaeReveal(
                commitment_hash=commitment_hash,
                domain_id=domain_id,
                optimae_id=optimae_id,
                parameters=params,
                reported_performance=performance,
                nonce=nonce,
                champion_metrics={
                    "val_mae": metrics.get("val_mae"),
                    "train_mae": metrics.get("train_mae"),
                    "val_naive_mae": metrics.get("val_naive_mae"),
                    "train_naive_mae": metrics.get("train_naive_mae"),
                    "test_mae": metrics.get("test_mae"),
                    "test_naive_mae": metrics.get("test_naive_mae"),
                },
            ).model_dump_json()),
        )

        # Process the reveal locally FIRST — critical so the auto-accept path
        # (block creation, domain_best update) runs before any peer relay can
        # consume the commitment.  The commit-reveal manager marks the commitment
        # as revealed, so any peer relay of the same reveal is safely ignored.
        await self._handle_optimae_reveal(reveal_msg, "self")
        await self._broadcast(reveal_msg)

        self._log_event("broadcast",
            domain_id=domain_id, gen=gen, performance=performance,
            optimae_id=optimae_id[:16])

        logger.info(
            "📡 Champion broadcast: domain=%s gen=%d perf=%.4f optimae=%s",
            domain_id, gen, performance, optimae_id[:16],
        )

    async def _broadcast_stage_complete(
        self, domain_id: str, stage: int, total_stages: int,
        champion_params: dict, champion_fitness: float, metrics: dict,
    ) -> None:
        """Broadcast stage completion so all nodes advance stages together."""
        msg = Message(
            msg_type=MessageType.STAGE_COMPLETE,
            sender_id=self.peer_id,
            payload={
                "domain_id": domain_id,
                "stage": stage,
                "total_stages": total_stages,
                "champion_params": champion_params,
                "champion_fitness": champion_fitness,
                "champion_metrics": metrics or {},
            },
        )
        await self._broadcast(msg)

        # Reset stage-start fitness for the next stage (local tracking)
        if champion_fitness is not None:
            self._domain_stage_start_fitness[domain_id] = champion_fitness

        self._log_event("stage_end",
            domain_id=domain_id, stage=stage, total_stages=total_stages,
            performance=champion_fitness)

        logger.info(
            "📢 Stage %d/%d complete broadcast for %s (champion fitness=%.6f)",
            stage, total_stages, domain_id,
            champion_fitness if champion_fitness is not None else float("nan"),
        )

    async def _handle_stage_complete(self, message: Message, from_peer: str) -> None:
        """Handle STAGE_COMPLETE from a peer — inject champion and signal stage advance."""
        payload = message.payload
        domain_id = payload.get("domain_id")
        stage = payload.get("stage")
        total_stages = payload.get("total_stages")
        champion_params = payload.get("champion_params")
        champion_fitness = payload.get("champion_fitness")

        if domain_id is None or stage is None:
            return

        logger.info(
            "📢 Received stage_complete for %s: stage=%d/%d from %s (champion=%.6f)",
            domain_id, stage, total_stages or 0, message.sender_id[:12],
            champion_fitness if champion_fitness is not None else float("nan"),
        )

        # Inject the stage champion into our optimizer's next generation
        plugin = self._optimizer_plugins.get(domain_id)
        if champion_params:
            if plugin and hasattr(plugin, "set_network_champion"):
                plugin.set_network_champion(champion_params)

        # Update domain best if this champion is better
        if champion_params and champion_fitness is not None:
            current_best = self._domain_best.get(domain_id, (None, None))
            if self._is_better(domain_id, champion_fitness, current_best[1]):
                self._domain_best[domain_id] = (champion_params, champion_fitness)

        # Reset stage-start fitness for the new stage (the champion's fitness
        # becomes the baseline for the next stage's improvement threshold)
        if champion_fitness is not None:
            self._domain_stage_start_fitness[domain_id] = champion_fitness
            logger.info(
                "[%s] Stage-start fitness reset to %.6f for next stage",
                domain_id, champion_fitness,
            )

        # Signal the optimizer to finish its current stage at the next candidate boundary.
        # Guard against stale messages: only advance if the incoming stage matches or
        # exceeds the optimizer's current stage, preventing a late-arriving
        # STAGE_COMPLETE for an already-passed stage from triggering an extra advance.
        if plugin and hasattr(plugin, "force_stage_advance"):
            optimizer_stage = getattr(plugin, "_current_stage", None)
            if optimizer_stage is not None and stage <= optimizer_stage:
                logger.info(
                    "⏭️  Ignoring stale STAGE_COMPLETE (peer stage=%d <= our stage=%d) for %s",
                    stage, optimizer_stage, domain_id,
                )
            else:
                plugin.force_stage_advance()
                logger.info("↗️  Signalled optimizer to advance stage for %s", domain_id)

        self._log_event("stage_sync",
            domain_id=domain_id, stage=stage, total_stages=total_stages,
            peer=message.sender_id[:12], performance=champion_fitness)

    async def _process_one_pending_eval(self, domain_id: str) -> None:
        """Process one pending evaluation task (called between candidate evaluations)."""
        tasks = self.task_queue.get_pending_for_domains(
            [domain_id], limit=1,
        )
        if not tasks:
            return

        task = tasks[0]
        if task.task_type != TaskType.OPTIMAE_VERIFICATION:
            return

        # Check if we're in the quorum
        if task.optimae_id:
            state = self.quorum.get_state(task.optimae_id)
            if state and self.peer_id not in state.required_evaluators:
                return

        logger.info("🔧 Processing eval task %s between candidates", task.task_id[:12])
        try:
            await self._evaluate_task(task)
        except Exception:
            logger.exception("Eval task %s failed", task.task_id[:12])

    # ================================================================
    # Evaluator loop (when this node evaluates)
    # ================================================================

    async def _evaluator_loop(self) -> None:
        """Background loop: poll for verification tasks and evaluate."""
        while self._running:
            tasks = self.task_queue.get_pending_for_domains(
                self.evaluator_domains,
                limit=self.config.eval_max_concurrent,
            )

            for task in tasks:
                if task.task_type != TaskType.OPTIMAE_VERIFICATION:
                    continue

                # Check if we're in the quorum for this optimae
                if task.optimae_id:
                    state = self.quorum.get_state(task.optimae_id)
                    if state and self.peer_id not in state.required_evaluators:
                        continue  # Not selected for this quorum

                # Claim and evaluate
                claimed = self.task_queue.claim(task.id, self.peer_id)
                if claimed is None:
                    continue

                await self._flood_task_claimed(claimed)

                try:
                    perf, synth_hash = await self._evaluate_task(claimed)
                    self.task_queue.complete(task.id, verified_performance=perf)
                    await self._flood_task_completed_with(task.id, perf)
                    await self._process_task_completion(
                        claimed, self.peer_id, perf, synth_hash,
                    )
                except Exception:
                    logger.exception("Evaluation failed for task %s", task.id[:12])
                    task.fail("evaluation error")

            await asyncio.sleep(self.config.eval_poll_interval)

    async def _evaluate_task(self, task: Task) -> tuple[float, str]:
        """Evaluate a verification task using the domain's plugin.

        Each evaluator generates DIFFERENT synthetic data using a seed
        derived from: commitment_hash + domain + evaluator_id + chain_tip.
        The optimizer CANNOT predict this seed because they don't know:
          - Which evaluators will be selected (random quorum)
          - The chain tip hash at quorum selection time

        A genuinely good model generalizes across different synthetic
        datasets (within the incentive tolerance margin). An overfitted
        model fails badly.

        Returns:
            (verified_performance, synthetic_data_hash)
        """
        domain_id = task.domain_id
        plugin = self._evaluator_plugins.get(domain_id)
        if plugin is None:
            raise ValueError(f"No evaluator plugin for domain {domain_id}")

        # Derive per-evaluator seed for synthetic data generation
        # This seed is UNIQUE to this evaluator and UNPREDICTABLE to the optimizer
        synth_seed = None
        commitment_hash = ""
        if task.optimae_id:
            commitment_hash = task.parameters.get("_don_commitment_hash", "")
            if not commitment_hash:
                commitment_hash = task.optimae_id
            chain_tip = self._get_tip()
            chain_tip_hash = chain_tip.hash if chain_tip else "genesis"

            synth_seed = self.seed_policy.get_seed_for_synthetic_data(
                commitment_hash, domain_id, self.peer_id, chain_tip_hash,
            )

        # Generate synthetic data with hash (each evaluator gets different data)
        synthetic_plugin = self._synthetic_plugins.get(domain_id)
        synthetic_data = None
        synthetic_data_hash = ""
        if synthetic_plugin is not None:
            synthetic_data, synthetic_data_hash = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: synthetic_plugin.generate_with_hash(synth_seed),
            )

        # Run evaluation with synthetic data
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: plugin.evaluate(
                parameters=task.parameters,
                data=synthetic_data,
            ),
        )

        perf = result.get("performance", 0.0) if isinstance(result, dict) else float(result)
        return perf, synthetic_data_hash

    # ================================================================
    # Flooding helpers
    # ================================================================

    async def _flood_task_created(self, task: Task) -> None:
        tc = TaskCreated(
            task_id=task.id,
            task_type=task.task_type.value,
            domain_id=task.domain_id,
            requester_id=task.requester_id,
            parameters=task.parameters,
            optimae_id=task.optimae_id,
            reported_performance=task.reported_performance,
            priority=task.priority,
        )
        msg = Message(
            msg_type=MessageType.TASK_CREATED,
            sender_id=self.peer_id,
            payload=json.loads(tc.model_dump_json()),
        )
        await self._broadcast(msg)

    async def _flood_task_claimed(self, task: Task) -> None:
        tc = TaskClaimed(
            task_id=task.id,
            evaluator_id=task.evaluator_id or "",
            domain_id=task.domain_id,
        )
        msg = Message(
            msg_type=MessageType.TASK_CLAIMED,
            sender_id=self.peer_id,
            payload=json.loads(tc.model_dump_json()),
        )
        await self._broadcast(msg)

    async def _flood_task_completed_with(
        self, task_id: str, verified_performance: float,
    ) -> None:
        task = self.task_queue.tasks.get(task_id)
        if task is None:
            return
        tc = TaskCompleted(
            task_id=task.id,
            evaluator_id=task.evaluator_id or "",
            domain_id=task.domain_id,
            verified_performance=verified_performance,
            result=task.result,
            optimae_id=task.optimae_id,
        )
        msg = Message(
            msg_type=MessageType.TASK_COMPLETED,
            sender_id=self.peer_id,
            payload=json.loads(tc.model_dump_json()),
        )
        await self._broadcast(msg)

    # ================================================================
    # Helpers
    # ================================================================

    def _get_eligible_evaluators(self, domain_id: str) -> list[str]:
        """Get peer IDs of nodes that can evaluate a domain.

        In a real network this would come from peer capability announcements.
        For now, includes this node + all peers (simplified).
        Deduplicates by peer_id (machines with multiple NICs may appear twice).
        """
        seen: set[str] = set()
        evaluators: list[str] = []
        if domain_id in self.evaluator_domains:
            seen.add(self.peer_id)
            evaluators.append(self.peer_id)
        # In production, peers would advertise their evaluator capabilities
        for peer in self._peers.values():
            if peer.peer_id not in seen:
                seen.add(peer.peer_id)
                evaluators.append(peer.peer_id)
        return evaluators

    # ================================================================
    # Block sync
    # ================================================================

    async def _find_common_ancestor(
        self, session, endpoint: str, our_height: int
    ) -> int:
        """Binary-search for the highest block index we share with *endpoint*.

        Returns the common ancestor index (always >= 0 because genesis
        blocks are deterministic and identical across all nodes).
        """
        if our_height <= 1:
            return 0

        lo, hi = 0, our_height - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            our_block = self._get_block(mid)
            if our_block is None:
                hi = mid - 1
                continue
            peer_blocks = await fetch_blocks(session, endpoint, mid, mid)
            if not peer_blocks:
                hi = mid - 1
                continue
            if peer_blocks[0].hash == our_block.hash:
                lo = mid          # common ancestor is at least *mid*
            else:
                hi = mid - 1      # divergence; common ancestor is earlier
        return lo

    async def _attempt_chain_reorg(
        self, session, endpoint: str
    ) -> bool:
        """Detect a chain fork with *endpoint* and reorganise if the peer
        has a strictly longer chain.

        Returns True if a rollback was performed and the caller should
        retry the sync loop, False otherwise.
        """
        if not self.chaindb:
            logger.warning("Chain reorg requires chaindb — skipping")
            return False

        our_height = self._get_height()
        state = self.sync_manager.peers.get(endpoint)
        peer_height = state.their_height if state else 0

        # Only reorg when the peer is strictly ahead
        if peer_height <= our_height:
            logger.debug(
                "Reorg skipped: peer %s not ahead (%d vs %d)",
                endpoint, peer_height, our_height,
            )
            return False

        common = await self._find_common_ancestor(session, endpoint, our_height)

        # Safety: never roll back past finalised blocks
        if common < self.finality.finalized_height:
            logger.warning(
                "Cannot reorg past finalized height %d (common ancestor %d)",
                self.finality.finalized_height, common,
            )
            return False

        if common >= our_height - 1:
            # No actual fork — the chains are compatible, failure must be
            # something else (e.g. hash / merkle mismatch).
            return False

        logger.info(
            "Chain fork detected at index %d — rolling back from height %d "
            "(peer %s has longer chain at height %d)",
            common + 1, our_height, endpoint, peer_height,
        )

        blocks_removed = self.chaindb.rollback_to(common)
        logger.info("Rolled back %d block(s) to index %d", blocks_removed, common)

        # Refresh cached sync-manager state after the rollback
        new_tip = self._get_tip()
        self.sync_manager.update_our_state(
            self._get_height(),
            new_tip.hash if new_tip else "",
            self.finality.finalized_height,
        )
        return True

    async def _sync_with_peer(self, endpoint: str) -> None:
        """Sync our chain with a peer that's ahead of us.

        Fetches blocks in batches and validates them before appending.
        """
        if not self.sync_manager.needs_sync(endpoint):
            return

        self.sync_manager.mark_syncing(endpoint)
        session = self.transport._session
        if session is None:
            self.sync_manager.record_sync_failure(endpoint)
            return

        try:
            reorg_attempts = 0
            while True:
                needed = self.sync_manager.compute_blocks_needed(endpoint)
                if needed is None:
                    break

                from_idx, to_idx = needed
                logger.info("Syncing blocks %d..%d from %s", from_idx, to_idx, endpoint)

                blocks = await fetch_blocks(session, endpoint, from_idx, to_idx)
                if not blocks:
                    logger.warning("Got no blocks from %s for range %d..%d", endpoint, from_idx, to_idx)
                    self.sync_manager.record_sync_failure(endpoint)
                    return

                appended = self._validate_and_append_blocks(blocks)
                if appended == 0:
                    # ── Fork detection & chain reorganisation ───────
                    if reorg_attempts < 1:
                        reorged = await self._attempt_chain_reorg(session, endpoint)
                        if reorged:
                            reorg_attempts += 1
                            # Chain was rolled back — restart the sync loop
                            # so compute_blocks_needed picks up the new height.
                            continue
                    logger.warning("Failed to append any blocks from %s", endpoint)
                    self.sync_manager.record_sync_failure(endpoint)
                    return

                # Update our state
                tip = self._get_tip()
                height = self._get_height()
                self.sync_manager.update_our_state(
                    height,
                    tip.hash if tip else "",
                    self.finality.finalized_height,
                )

                # Update finality for each new block
                for block in blocks[:appended]:
                    depth_block_hash = None
                    target_height = block.header.index - self.finality.confirmation_depth
                    if target_height >= 0:
                        b = self._get_block(target_height)
                        if b:
                            depth_block_hash = b.hash
                    self.finality.on_new_block(block.header.index, depth_block_hash)

                if appended < len(blocks):
                    break  # Some blocks failed validation

                # Check if we still need more
                state = self.sync_manager.peers.get(endpoint)
                if state and height >= state.their_height:
                    break  # Caught up

            # Extract accepted optimae from synced blocks → update domain best
            # This is the island model: pick up champions from other nodes
            self._update_domain_best_from_chain()

            self._save_chain()
            self.sync_manager.record_sync_success(endpoint, self._get_height())
            logger.info("Sync complete with %s (height now %d)", endpoint, self._get_height())

        except Exception:
            logger.exception("Sync error with %s", endpoint)
            self.sync_manager.record_sync_failure(endpoint)

    async def _initial_sync(self) -> None:
        """Sync with all known peers on startup."""
        session = self.transport._session
        if session is None:
            return

        for endpoint in list(self._peers.keys()):
            status = await fetch_chain_status(session, endpoint)
            if status is None:
                continue

            self.sync_manager.update_peer_status(endpoint, status)
            if status.chain_height > self._get_height():
                await self._sync_with_peer(endpoint)

    def _find_peer_endpoint(self, from_addr: str, sender_id: str) -> str | None:
        """Find the endpoint for a peer based on address or sender ID."""
        # Try matching by sender_id (peer_id)
        for ep, peer in self._peers.items():
            if peer.peer_id == sender_id:
                return ep
        # Try matching by address
        for ep in self._peers:
            if from_addr in ep:
                return ep
        return None

    def _find_peer_endpoint_by_id(self, peer_id: str) -> str | None:
        """Find the endpoint for a peer by peer ID only."""
        for ep, peer in self._peers.items():
            if peer.peer_id == peer_id:
                return ep
        return None

    async def _gossip_heartbeat_loop(self) -> None:
        """Periodic GossipSub mesh maintenance."""
        while self._running:
            try:
                if self.gossip:
                    await self.gossip.heartbeat()
            except Exception:
                logger.exception("Gossip heartbeat error")
            await asyncio.sleep(self.config.gossip_heartbeat_interval)

    async def _lan_scan_task(self) -> None:
        """Run LAN scan in background — finds peers without blocking main loop."""
        try:
            if self.discovery:
                found = await self.discovery.lan_scan(None)
                if found:
                    logger.info("LAN scan: %d new peers", found)
                    # Connect to discovered peers
                    for peer in self.discovery.get_connectable_peers(10):
                        if peer.endpoint not in self._peers:
                            p = self.add_peer(peer.address, peer.port, peer.peer_id)
                            if self.gossip:
                                self.gossip.add_peer(p.peer_id)
                            self.discovery.mark_connected(peer.endpoint)
                            self._log_event("peer_connected", endpoint=peer.endpoint, peer_id=p.peer_id[:12])
                            logger.info("Connected to discovered peer: %s", peer.endpoint)
        except Exception:
            logger.exception("LAN scan error")

    async def _discovery_loop(self) -> None:
        """Periodic peer discovery: LAN TCP scan + PEX + random walks."""
        import aiohttp

        # Start LAN discovery and run initial scan as background task
        if self.discovery:
            domain_ids = [d.domain_id for d in self._domain_roles.values()]
            self.discovery.start_lan_discovery(domains=domain_ids)
            asyncio.create_task(self._lan_scan_task())

        _lan_rescan_tick = time.time()
        _LAN_RESCAN_INTERVAL = 300.0  # Re-scan LAN every 5 minutes

        while self._running:
            try:
                if self.discovery:
                    async with aiohttp.ClientSession() as session:
                        # Periodic LAN re-scan (background, non-blocking)
                        now = time.time()
                        if now - _lan_rescan_tick >= _LAN_RESCAN_INTERVAL:
                            asyncio.create_task(self._lan_scan_task())
                            _lan_rescan_tick = now

                        # Bootstrap if we need more peers
                        if self.discovery.needs_more_peers():
                            found = await self.discovery.discover_from_bootstrap(session)
                            if found:
                                logger.info("Bootstrap discovery: %d new peers", found)

                        # Peer exchange with connected peers
                        found = await self.discovery.peer_exchange(session)
                        if found:
                            logger.debug("PEX: %d new peers", found)

                        # Random walk
                        found = await self.discovery.random_walk(session)
                        if found:
                            logger.debug("Random walk: %d new peers", found)

                        # Connect to discovered peers
                        for peer in self.discovery.get_connectable_peers(3):
                            if peer.endpoint not in self._peers:
                                p = self.add_peer(peer.address, peer.port, peer.peer_id)
                                if self.gossip:
                                    self.gossip.add_peer(p.peer_id)
                                self.discovery.mark_connected(peer.endpoint)
                                logger.info("Connected to discovered peer: %s", peer.endpoint)

                    # Cleanup stale peers
                    removed = self.discovery.cleanup()
                    if removed:
                        logger.debug("Cleaned up %d stale peers", removed)

                    # Save peers to DB
                    if self.chaindb:
                        for peer in self.discovery.get_random_peers(50):
                            self.chaindb.save_peer(
                                peer.peer_id, peer.address, peer.port,
                                peer.last_seen, domains=peer.domains, roles=peer.roles,
                            )
            except Exception:
                logger.exception("Discovery loop error")

            await asyncio.sleep(self.config.discovery_interval)

    async def _maintenance_loop(self) -> None:
        """Periodic cleanup: expired commitments, decided quorums, fee market stats."""
        while self._running:
            self.commit_reveal.cleanup_expired()
            self.quorum.cleanup_decided()

            # Log fee market stats periodically
            if self.fee_market:
                stats = self.fee_market.get_stats()
                logger.debug(
                    "Fee market: base=%.6f mempool=%d burned=%.4f",
                    stats["base_fee"], stats["mempool_size"], stats["total_burned"],
                )

            await asyncio.sleep(60.0)

    # ================================================================
    # HTTP endpoints (same as old node for backward compat)
    # ================================================================

    def _register_http_routes(self) -> None:
        app = self.transport._app
        app.router.add_get("/tasks/pending", self._http_tasks_pending)
        app.router.add_post("/tasks/claim", self._http_task_claim)
        app.router.add_post("/tasks/complete", self._http_task_complete)
        app.router.add_post("/inference", self._http_inference_request)
        app.router.add_get("/status", self._http_status)
        app.router.add_get("/chain/status", self._http_chain_status)
        app.router.add_get("/chain/blocks", self._http_chain_blocks)
        app.router.add_get("/chain/block/{index}", self._http_chain_block)
        app.router.add_get("/peers", self._http_peers)
        app.router.add_get("/fees", self._http_fees)
        app.router.add_get("/stats", self._http_stats)
        app.router.add_get("/stats/experiments", self._http_stats_experiments)
        app.router.add_get("/stats/rounds", self._http_stats_rounds)
        app.router.add_get("/stats/export", self._http_stats_export)
        app.router.add_get("/stats/chain-metrics", self._http_stats_chain_metrics)

        # Web dashboard
        if self.config.dashboard_enabled:
            try:
                from doin_node.dashboard.routes import setup_dashboard
                setup_dashboard(app, self)
            except Exception as e:
                logger.warning("Dashboard setup failed: %s", e)

    async def _http_tasks_pending(self, request) -> Any:
        from aiohttp import web
        domains_param = request.query.get("domains", "")
        limit = int(request.query.get("limit", "10"))
        if domains_param:
            domain_ids = [d.strip() for d in domains_param.split(",")]
            tasks = self.task_queue.get_pending_for_domains(domain_ids, limit=limit)
        else:
            tasks = self.task_queue.get_pending(limit=limit)
        return web.json_response({
            "tasks": [json.loads(t.model_dump_json()) for t in tasks],
            "total_pending": self.task_queue.pending_count,
        })

    async def _http_task_claim(self, request) -> Any:
        from aiohttp import web
        try:
            data = await request.json()
            task_id = data["task_id"]
            evaluator_id = data["evaluator_id"]
        except (KeyError, json.JSONDecodeError):
            return web.json_response({"status": "error", "detail": "task_id and evaluator_id required"}, status=400)
        task = self.task_queue.claim(task_id, evaluator_id)
        if task is None:
            return web.json_response({"status": "error", "detail": "task not found or already claimed"}, status=409)
        await self._flood_task_claimed(task)
        return web.json_response({"status": "claimed", "task": json.loads(task.model_dump_json())})

    async def _http_task_complete(self, request) -> Any:
        from aiohttp import web
        try:
            data = await request.json()
            task_id = data["task_id"]
        except (KeyError, json.JSONDecodeError):
            return web.json_response({"status": "error", "detail": "task_id required"}, status=400)

        verified_perf = data.get("verified_performance")
        result = data.get("result")
        evaluator_id = data.get("evaluator_id", "")

        task = self.task_queue.complete(task_id, verified_performance=verified_perf, result=result)
        if task is None:
            return web.json_response({"status": "error", "detail": "task not found or not claimed"}, status=409)

        await self._process_task_completion(task, evaluator_id, verified_perf)
        await self._flood_task_completed_with(task_id, verified_perf or 0)

        block = await self.try_generate_block()
        return web.json_response({"status": "completed", "task_id": task_id, "block_generated": block is not None})

    async def _http_inference_request(self, request) -> Any:
        from aiohttp import web
        try:
            data = await request.json()
            domain_id = data["domain_id"]
        except (KeyError, json.JSONDecodeError):
            return web.json_response({"status": "error", "detail": "domain_id required"}, status=400)
        if domain_id not in self._domains:
            return web.json_response({"status": "error", "detail": f"unknown domain: {domain_id}"}, status=404)

        input_data = data.get("input_data", {})
        client_id = data.get("client_id", "anonymous")
        task = Task(
            task_type=TaskType.INFERENCE_REQUEST,
            domain_id=domain_id,
            requester_id=client_id,
            parameters=input_data,
            priority=10,
        )
        self.task_queue.add(task)
        await self._flood_task_created(task)
        return web.json_response({"status": "queued", "task_id": task.id})

    async def _http_chain_status(self, request) -> Any:
        """Serve our chain status for sync."""
        from aiohttp import web
        tip = self._get_tip()
        height = self._get_height()
        return web.json_response({
            "chain_height": height,
            "tip_hash": tip.hash if tip else "",
            "tip_index": tip.header.index if tip else -1,
            "finalized_height": self.finality.finalized_height,
            "component_versions": self._component_versions,
        })

    async def _http_chain_blocks(self, request) -> Any:
        """Serve blocks by index range for sync."""
        from aiohttp import web
        height = self._get_height()
        from_idx = int(request.query.get("from", "0"))
        to_idx = int(request.query.get("to", str(height - 1)))

        # Cap at 50 blocks per request
        to_idx = min(to_idx, from_idx + 49)

        blocks = self._get_blocks_range(from_idx, to_idx)
        has_more = to_idx < height - 1

        return web.json_response({
            "request_id": request.query.get("request_id", ""),
            "blocks": [json.loads(b.model_dump_json()) for b in blocks],
            "has_more": has_more,
        })

    async def _http_chain_block(self, request) -> Any:
        """Serve a single block by index."""
        from aiohttp import web
        index = int(request.match_info["index"])
        block = self._get_block(index)
        if block is None:
            return web.json_response(
                {"status": "error", "detail": f"block {index} not found"},
                status=404,
            )
        return web.json_response(json.loads(block.model_dump_json()))

    async def _http_peers(self, request) -> Any:
        """Serve known peers for peer discovery."""
        from aiohttp import web
        peers = []
        if self.discovery:
            for p in self.discovery.get_random_peers(20):
                peers.append({
                    "peer_id": p.peer_id,
                    "address": p.address,
                    "port": p.port,
                    "domains": p.domains,
                    "roles": p.roles,
                    "chain_height": p.chain_height,
                })
        else:
            for ep, peer in self._peers.items():
                peers.append({
                    "peer_id": peer.peer_id,
                    "address": peer.address,
                    "port": peer.port,
                })
        return web.json_response({
            "self": {"peer_id": self.peer_id, "component_versions": self._component_versions},
            "peers": peers,
        })

    async def _http_fees(self, request) -> Any:
        """Serve current fee market information."""
        from aiohttp import web
        if not self.fee_market:
            return web.json_response({"enabled": False})
        stats = self.fee_market.get_stats()
        stats["enabled"] = True
        return web.json_response(stats)

    async def _http_stats(self, request) -> Any:
        """Serve current experiment statistics."""
        from aiohttp import web
        domain_id = request.query.get("domain")
        summary = self.experiment_tracker.get_summary(domain_id)
        # Enrich with OLAP data when available
        olap_summaries = self.experiment_tracker.get_olap_summary()
        if olap_summaries is not None:
            summary["olap_summaries"] = olap_summaries
        return web.json_response(summary)

    async def _http_stats_experiments(self, request) -> Any:
        """List all experiments with summaries from OLAP."""
        from aiohttp import web
        if self.experiment_tracker._olap is None:
            return web.json_response({"error": "OLAP not enabled"}, status=404)
        experiments = self.experiment_tracker._olap.get_all_experiments()
        summaries = self.experiment_tracker._olap.get_all_summaries()
        return web.json_response({"experiments": experiments, "summaries": summaries})

    async def _http_stats_rounds(self, request) -> Any:
        """Return round history for an experiment."""
        from aiohttp import web
        if self.experiment_tracker._olap is None:
            return web.json_response({"error": "OLAP not enabled"}, status=404)
        experiment_id = request.query.get("experiment_id", "")
        limit = int(request.query.get("limit", "1000"))
        if not experiment_id:
            return web.json_response({"error": "experiment_id required"}, status=400)
        rounds = self.experiment_tracker._olap.get_rounds(experiment_id, limit)
        return web.json_response({"rounds": rounds})

    async def _http_stats_export(self, request) -> Any:
        """Download the OLAP SQLite database file."""
        from aiohttp import web
        if self.experiment_tracker._olap is None:
            return web.json_response({"error": "OLAP not enabled"}, status=404)
        db_path = self.experiment_tracker._olap.db_path
        return web.FileResponse(db_path, headers={
            "Content-Disposition": "attachment; filename=olap.db",
        })

    async def _http_stats_chain_metrics(self, request) -> Any:
        """Return on-chain experiment metrics from OPTIMAE_ACCEPTED transactions."""
        from aiohttp import web
        from doin_node.stats.chain_metrics import collect_chain_metrics
        domain_id = request.query.get("domain_id")
        blocks = self.consensus.chain.blocks
        metrics = collect_chain_metrics(blocks, domain_id=domain_id)
        return web.json_response({"metrics": metrics, "count": len(metrics)})

    async def _http_status(self, request) -> Any:
        from aiohttp import web
        status: dict[str, Any] = {
            "status": "healthy",
            "peer_id": self.peer_id[:12],
            "port": self.config.port,
            "chain_height": self._get_height(),
            "storage_backend": "sqlite" if self.chaindb else "json",
            "network_protocol": "gossipsub" if self.gossip else "flooding",
            "domains": {
                d: {
                    "optimize": r.optimize, "evaluate": r.evaluate, "synthetic": r.has_synthetic_data,
                    "target_performance": r.target_performance,
                    "best_performance": self._domain_best.get(d, (None, None))[1],
                    "converged": d in self._domain_converged,
                }
                for d, r in self._domain_roles.items()
            },
            "peers": len(self._peers),
            "task_queue": {
                "pending": self.task_queue.pending_count,
                "claimed": self.task_queue.claimed_count,
                "completed": self.task_queue.completed_count,
            },
            "security": {
                "finalized_height": self.finality.finalized_height,
                "reputation_tracked_peers": len(self.reputation.all_scores),
                "pending_commitments": self.commit_reveal.pending_count,
                "pending_quorums": self.quorum.pending_count,
            },
            "optimizer_domains": self.optimizer_domains,
            "evaluator_domains": self.evaluator_domains,
            "coin": {
                "total_supply": self.balance_tracker.total_supply,
                "my_balance": self.balance_tracker.get_balance(self.peer_id),
                "top_holders": self.balance_tracker.top_holders(5),
            },
            "difficulty": self.difficulty.get_stats(),
        }

        # Add gossip mesh stats
        if self.gossip:
            status["gossip"] = self.gossip.get_mesh_stats()

        # Add discovery stats
        if self.discovery:
            status["discovery"] = self.discovery.get_stats()

        # Add fee market stats
        if self.fee_market:
            status["fee_market"] = self.fee_market.get_stats()

        # Add storage stats (SQLite)
        if self.chaindb:
            status["storage"] = self.chaindb.get_stats()

        return web.json_response(status)
