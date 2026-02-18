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

        # Peer discovery
        self.discovery: PeerDiscovery | None = None
        if config.discovery_enabled:
            self.discovery = PeerDiscovery(
                our_peer_id=self.identity.peer_id,
                our_port=config.port,
                bootstrap_nodes=config.bootstrap_peers,
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
        self._start_time: float = time.time()

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
        ]
        for msg_type, handler in all_handlers:
            self.flooding.on_message(msg_type, handler)
            if self.gossip:
                self.gossip.on_message(msg_type, handler)

        self.transport.on_message(self._on_transport_message)

        # ── Register domains ──
        for domain_role in config.domains:
            self._register_domain(domain_role)

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

    def _peer_id_exists(self, peer_id: str) -> bool:
        """Check if a peer with this ID already exists (on any endpoint)."""
        return any(p.peer_id == peer_id for p in self._peers.values())

    def add_peer(self, address: str, port: int, peer_id: str = "") -> Peer:
        pid = peer_id or f"{address}:{port}"
        # Dedup by peer_id: if same identity already connected on another IP, skip
        if peer_id and self._peer_id_exists(peer_id):
            # Return existing peer
            for p in self._peers.values():
                if p.peer_id == peer_id:
                    return p
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

        # Initialize storage
        if self.chaindb:
            self.chaindb.open()
            if self.chaindb.height == 0:
                self.chaindb.initialize(self.peer_id)
            logger.info("SQLite storage: height=%d", self.chaindb.height)
        else:
            chain_path = Path(self.config.data_dir) / "chain.json"
            if chain_path.exists():
                self.chain.load()
            else:
                self.chain.initialize(self.peer_id)
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
                        if current[1] is None or verified > current[1]:
                            self._domain_best[domain_id] = (parameters, verified)
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
        elif message.payload is None:
            message.payload = {"_sender_port": self.config.port}
        if self.gossip:
            await self.gossip.publish(message)
        else:
            await self.transport.broadcast(list(self._peers.keys()), message)

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
        # Auto-discover peers from incoming connections (skip localhost)
        # Extract sender's actual port from message payload or probe
        sender_port = (message.payload or {}).get("_sender_port", self.config.port)
        sender_endpoint = f"{sender}:{sender_port}"
        _local = {"unknown", "127.0.0.1", "::1", "localhost"}
        if sender not in _local:
            # Check if we already have this peer by peer_id (on any endpoint)
            existing = self._peers.get(sender_endpoint)
            if not existing:
                # Check if peer_id exists on a different endpoint
                for ep, p in list(self._peers.items()):
                    if p.peer_id == message.sender_id:
                        existing = p
                        break
            if existing:
                # Update peer_id if we had a placeholder (e.g. from bootstrap)
                if existing.peer_id != message.sender_id:
                    old_id = existing.peer_id
                    existing.peer_id = message.sender_id
                    # Update GossipSub mesh with real peer_id
                    if self.gossip:
                        self.gossip.remove_peer(old_id)
                        self.gossip.add_peer(message.sender_id)
                        for topic_state in self.gossip._topics.values():
                            topic_state.mesh.discard(old_id)
                            topic_state.mesh.add(message.sender_id)
                    logger.info("🔍 Updated peer %s → %s", old_id[:12], message.sender_id[:12])
            else:
                self.add_peer(sender, sender_port, peer_id=message.sender_id)
                logger.info("🔍 Auto-discovered peer %s:%s from incoming message", sender, sender_port)
                # Register in GossipSub mesh so messages flow bidirectionally
                if self.gossip:
                    self.gossip.add_peer(message.sender_id)
                    for topic_state in self.gossip._topics.values():
                        topic_state.mesh.add(message.sender_id)

        if self.gossip:
            # GossipSub handles dedup, dispatch, and mesh forwarding
            await self.gossip.handle_incoming(message, sender)
        else:
            # Legacy flooding
            should_forward = await self.flooding.handle_incoming(message, sender)
            if should_forward:
                forward_msg = self.flooding.prepare_forward(message)
                endpoints = [ep for ep in self._peers if ep != sender]
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

        # Validate parameter bounds
        validator = self._bounds_validators.get(data.domain_id)
        if validator:
            ok, reason = validator.validate(data.parameters)
            if not ok:
                logger.warning("Bounds validation failed: %s", reason)
                return

            role = self._domain_roles.get(data.domain_id)
            if role:
                ok, reason = validator.validate_resource_limits(data.parameters, role.resource_limits)
                if not ok:
                    logger.warning("Resource limits exceeded: %s", reason)
                    return

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

        # ── Optimistic adoption (island model) ──────────────────────
        # Immediately adopt better parameters from peers WITHOUT waiting for
        # consensus. The optimizer is greedy — if someone claims better perf,
        # use it now and keep optimizing from there. Consensus still runs in
        # parallel for rewards, payment, and permanent on-chain record.
        # If the optimae later gets rejected, we've lost a few rounds at worst.
        if message.sender_id != self.peer_id:
            current_best = self._domain_best.get(data.domain_id, (None, None))
            if current_best[1] is None or data.reported_performance > current_best[1]:
                self._domain_best[data.domain_id] = (data.parameters, data.reported_performance)
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

        # Domain must have synthetic data for verification trust
        role = self._domain_roles.get(data.domain_id)
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
        if ann.block_index >= self.chain.height:
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
            if current_best[1] is None or verified > current_best[1]:
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

        # Record task completion
        self.consensus.record_transaction(Transaction(
            tx_type=TransactionType.TASK_COMPLETED,
            domain_id=task.domain_id,
            peer_id=evaluator_id,
            payload={"task_id": task.id, "task_type": task.task_type.value},
        ))

    # ================================================================
    # Block generation (with finality + anchoring)
    # ================================================================

    async def try_generate_block(self) -> Block | None:
        if not self.consensus.can_generate_block():
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
        # Wait for LAN discovery to complete before starting optimization
        logger.info("Optimizer waiting 10s for peer discovery...")
        await asyncio.sleep(10)

        for domain_id in self.optimizer_domains:
            if domain_id in self._domain_converged:
                continue
            plugin = self._optimizer_plugins.get(domain_id)
            if plugin is None:
                continue

            # Wire up DOIN callbacks on the plugin
            self._setup_optimizer_callbacks(domain_id, plugin)

            try:
                logger.info("🧬 Starting DEAP GA optimization for %s", domain_id)
                await self._run_full_optimization(domain_id, plugin)
                logger.info("✅ Optimization complete for %s", domain_id)
            except Exception:
                logger.exception("Optimization error for %s", domain_id)

    def _setup_optimizer_callbacks(self, domain_id: str, plugin: Any) -> None:
        """Wire DOIN callbacks onto the optimizer plugin."""
        if not hasattr(plugin, "set_local_champion_callback"):
            return  # Not a DOIN-aware plugin

        loop = asyncio.get_event_loop()
        node = self  # Capture for closures

        def on_local_champion(params, fitness, metrics, gen, stage_info):
            """Called from optimizer thread when new local champion found → broadcast."""
            # Schedule async broadcast on the event loop
            import asyncio as _aio
            fut = _aio.run_coroutine_threadsafe(
                node._broadcast_champion(domain_id, params, fitness, metrics, gen, stage_info),
                loop,
            )
            try:
                fut.result(timeout=30)  # Block optimizer thread until broadcast completes
            except Exception as e:
                logger.warning("Champion broadcast failed: %s", e)

        def on_eval_service(gen, candidate_num, stage_info):
            """Called from optimizer thread between candidates → process 1 pending eval."""
            import asyncio as _aio
            fut = _aio.run_coroutine_threadsafe(
                node._process_one_pending_eval(domain_id),
                loop,
            )
            try:
                fut.result(timeout=120)  # Eval can take a while
            except Exception as e:
                logger.debug("Eval service between candidates: %s", e)

        def on_generation_end(population, hof, hyper_keys, gen, stage_info, stats):
            """Called from optimizer thread at end of each generation."""
            # Update tracking
            node._domain_round_count[domain_id] = gen + 1
            # Log generation summary
            champ_fit = stage_info.get("champion_fitness")
            champ_val = stage_info.get("champion_val_mae")
            avg_fit = stage_info.get("avg_fitness")
            stage = stage_info.get("stage", 1)
            total_stages = stage_info.get("total_stages", 1)
            n_evals = stage_info.get("total_candidates_evaluated", 0)
            patience_str = f"{stage_info.get('no_improve_counter', 0)}/{stage_info.get('patience', '?')}"

            logger.info(
                "[%s] gen=%d stage=%d/%d evals=%d  champ_fitness=%.6f  champ_val_mae=%s  avg_fitness=%.6f  patience=%s",
                domain_id, gen, stage, total_stages, n_evals,
                champ_fit if champ_fit is not None else float("nan"),
                f"{champ_val:.6f}" if champ_val is not None else "N/A",
                avg_fit if avg_fit is not None else float("nan"),
                patience_str,
            )

            # Record to experiment tracker
            detail_metrics = {
                "generation": gen,
                "stage": stage,
                "total_stages": total_stages,
                "total_candidates_evaluated": n_evals,
                "train_mae": stage_info.get("champion_train_mae"),
                "val_mae": champ_val,
                "val_naive_mae": stage_info.get("champion_naive_mae"),
            }
            try:
                node.experiment_tracker.record_round(
                    domain_id,
                    performance=-champ_fit if champ_fit is not None else 0.0,
                    parameters={},
                    wall_clock_seconds=0,
                    chain_height=node._get_height(),
                    peers_count=len(node._peers),
                    detail_metrics=detail_metrics,
                )
            except Exception:
                pass

        plugin.set_local_champion_callback(on_local_champion)
        plugin.set_eval_service_callback(on_eval_service)
        plugin.set_generation_end_callback(on_generation_end)

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
        if current_best[1] is None or performance > current_best[1]:
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
        performance = -fitness  # DOIN convention: higher = better

        # Update domain best
        current_best = self._domain_best.get(domain_id, (None, None))
        is_improvement = current_best[1] is None or performance > current_best[1]
        if is_improvement:
            self._domain_best[domain_id] = (params, performance)

        # Store champion metrics
        self._domain_champion_metrics[domain_id] = {
            "round": gen,
            "performance": performance,
            "parameters": params,
            **{k: v for k, v in metrics.items() if k != "fitness"},
        }

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
            ).model_dump_json()),
        )
        await self._broadcast(reveal_msg)

        logger.info(
            "📡 Champion broadcast: domain=%s gen=%d perf=%.4f optimae=%s",
            domain_id, gen, performance, optimae_id[:16],
        )

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
            logger.info("Sync complete with %s (height now %d)", endpoint, self.chain.height)

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
            "self": {"peer_id": self.peer_id},
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
