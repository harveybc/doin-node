"""DON Node — the main node that orchestrates networking, validation, and consensus.

A node:
1. Listens for messages from peers via the transport layer
2. Propagates messages via controlled flooding
3. Validates optimae by coordinating with evaluators
4. Tracks performance increments via the consensus engine
5. Generates new blocks when the threshold is met
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from aiohttp import web

from doin_core.consensus import ProofOfOptimization
from doin_core.crypto.identity import PeerIdentity
from doin_core.models.block import Block
from doin_core.models.domain import Domain
from doin_core.models.optimae import Optimae
from doin_core.models.task import Task, TaskQueue, TaskStatus, TaskType
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.protocol.messages import (
    BlockAnnouncement,
    Message,
    MessageType,
    OptimaeAnnouncement,
    TaskClaimed,
    TaskCompleted,
    TaskCreated,
)

from doin_node.blockchain.chain import Chain
from doin_node.network.flooding import FloodingConfig, FloodingProtocol
from doin_node.network.peer import Peer, PeerState
from doin_node.network.transport import Transport
from doin_node.validation.validator import OptimaeValidator

logger = logging.getLogger(__name__)


class NodeConfig:
    """Configuration for a DON node."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8470,
        data_dir: str = "./don-data",
        target_block_time: float = 600.0,
        initial_threshold: float = 1.0,
        validation_tolerance: float = 0.05,
        bootstrap_peers: list[str] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.data_dir = Path(data_dir)
        self.target_block_time = target_block_time
        self.initial_threshold = initial_threshold
        self.validation_tolerance = validation_tolerance
        self.bootstrap_peers = bootstrap_peers or []


class Node:
    """The main DON node.

    Ties together transport, flooding, chain, consensus, and validation
    into a single coherent node that participates in the network.
    """

    def __init__(self, config: NodeConfig, identity: PeerIdentity | None = None) -> None:
        self.config = config
        self.identity = identity or PeerIdentity.generate()

        # Components
        self.transport = Transport(host=config.host, port=config.port)
        self.flooding = FloodingProtocol(FloodingConfig())
        self.chain = Chain(data_dir=config.data_dir)
        self.consensus = ProofOfOptimization(
            target_block_time=config.target_block_time,
            initial_threshold=config.initial_threshold,
        )
        self.validator = OptimaeValidator(tolerance=config.validation_tolerance)
        self.task_queue = TaskQueue()

        # State
        self._peers: dict[str, Peer] = {}
        self._domains: dict[str, Domain] = {}
        self._running = False

        # Wire up message handlers
        self.flooding.on_message(
            MessageType.OPTIMAE_ANNOUNCEMENT, self._handle_optimae_announcement
        )
        self.flooding.on_message(
            MessageType.BLOCK_ANNOUNCEMENT, self._handle_block_announcement
        )
        self.flooding.on_message(
            MessageType.TASK_CREATED, self._handle_task_created
        )
        self.flooding.on_message(
            MessageType.TASK_CLAIMED, self._handle_task_claimed
        )
        self.flooding.on_message(
            MessageType.TASK_COMPLETED, self._handle_task_completed
        )
        self.transport.on_message(self._on_transport_message)

        # Register task queue HTTP endpoints on the transport
        self._register_task_routes()

    @property
    def peer_id(self) -> str:
        """This node's peer ID."""
        return self.identity.peer_id

    def register_domain(self, domain: Domain) -> None:
        """Register a domain for optimization tracking."""
        self._domains[domain.id] = domain
        self.consensus.register_domain(domain)
        self.validator.register_domain(domain)
        logger.info("Domain registered: %s (%s)", domain.name, domain.id)

    def add_peer(self, address: str, port: int, peer_id: str = "") -> Peer:
        """Add a known peer."""
        peer = Peer(
            peer_id=peer_id or f"{address}:{port}",
            address=address,
            port=port,
            state=PeerState.DISCOVERED,
        )
        self._peers[peer.endpoint] = peer
        logger.info("Peer added: %s", peer.endpoint)
        return peer

    async def start(self) -> None:
        """Start the node."""
        self._running = True

        # Initialize or load chain
        chain_path = self.config.data_dir / "chain.json"
        if chain_path.exists():
            self.chain.load()
            logger.info("Chain loaded: %d blocks", self.chain.height)
        else:
            self.chain.initialize(self.peer_id)
            self.chain.save()
            logger.info("New chain initialized")

        # Start transport
        await self.transport.start()

        # Connect to bootstrap peers
        for peer_addr in self.config.bootstrap_peers:
            host, port_str = peer_addr.rsplit(":", 1)
            self.add_peer(host, int(port_str))

        logger.info(
            "Node started: peer_id=%s, port=%d, peers=%d",
            self.peer_id[:12],
            self.config.port,
            len(self._peers),
        )

    async def stop(self) -> None:
        """Stop the node gracefully."""
        self._running = False
        await self.transport.stop()
        self.chain.save()
        logger.info("Node stopped")

    async def _on_transport_message(self, message: Message, sender: str) -> None:
        """Handle a message from the transport layer."""
        should_forward = await self.flooding.handle_incoming(message, sender)

        if should_forward:
            forward_msg = self.flooding.prepare_forward(message)
            endpoints = [ep for ep in self._peers if ep != sender]
            await self.transport.broadcast(endpoints, forward_msg)

    async def _handle_optimae_announcement(
        self, message: Message, from_peer: str
    ) -> None:
        """Handle an incoming optimae announcement.

        Creates a verification task in the work queue and floods TASK_CREATED.
        """
        announcement = OptimaeAnnouncement.model_validate(message.payload)

        optimae = Optimae(
            id=announcement.optimae_id,
            domain_id=announcement.domain_id,
            optimizer_id=message.sender_id,
            parameters=announcement.parameters,
            reported_performance=announcement.reported_performance,
        )

        try:
            self.validator.submit_for_validation(optimae)
        except ValueError as e:
            logger.warning("Rejected optimae: %s", e)
            return

        # Log optimae announced transaction
        self.consensus.record_transaction(Transaction(
            tx_type=TransactionType.OPTIMAE_ANNOUNCED,
            domain_id=optimae.domain_id,
            peer_id=message.sender_id,
            payload={"optimae_id": optimae.id, "reported_performance": optimae.reported_performance},
        ))

        # Create verification task in work queue
        task = Task(
            task_type=TaskType.OPTIMAE_VERIFICATION,
            domain_id=announcement.domain_id,
            requester_id=message.sender_id,
            parameters=announcement.parameters,
            optimae_id=announcement.optimae_id,
            reported_performance=announcement.reported_performance,
            priority=0,  # Verification is highest priority
        )
        self.task_queue.add(task)

        # Flood task created event
        await self._flood_task_created(task)

        logger.info(
            "Optimae %s → verification task %s created",
            optimae.id[:12],
            task.id[:12],
        )

    async def _handle_block_announcement(
        self, message: Message, from_peer: str
    ) -> None:
        """Handle an incoming block announcement."""
        announcement = BlockAnnouncement.model_validate(message.payload)
        logger.info(
            "Block announcement #%d from %s (hash=%s)",
            announcement.block_index,
            from_peer[:12] if from_peer else "unknown",
            announcement.block_hash[:12],
        )
        # TODO: Request full block data and validate/append

    # ----------------------------------------------------------------
    # Task lifecycle handlers (received via flooding from other nodes)
    # ----------------------------------------------------------------

    async def _handle_task_created(self, message: Message, from_peer: str) -> None:
        """Replicate a task created on another node into our local queue."""
        tc = TaskCreated.model_validate(message.payload)
        if tc.task_id in self.task_queue.tasks:
            return  # Already have it

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
        logger.debug("Replicated task %s from peer %s", tc.task_id[:12], from_peer[:12] if from_peer else "?")

    async def _handle_task_claimed(self, message: Message, from_peer: str) -> None:
        """Replicate a task claim from another node/evaluator."""
        tc = TaskClaimed.model_validate(message.payload)
        task = self.task_queue.tasks.get(tc.task_id)
        if task and task.status == TaskStatus.PENDING:
            task.claim(tc.evaluator_id)
            logger.debug("Task %s claimed by %s (via flood)", tc.task_id[:12], tc.evaluator_id[:12])

    async def _handle_task_completed(self, message: Message, from_peer: str) -> None:
        """Replicate a task completion from another node/evaluator."""
        tc = TaskCompleted.model_validate(message.payload)
        task = self.task_queue.tasks.get(tc.task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
            task.complete(verified_performance=tc.verified_performance, result=tc.result)

            # If this was a verification task, feed result to validator
            if task.task_type == TaskType.OPTIMAE_VERIFICATION and task.optimae_id:
                if tc.verified_performance is not None:
                    result = self.validator.record_evaluation(task.optimae_id, tc.verified_performance)
                    if result.is_valid:
                        # Accepted! Record in consensus
                        optimae = Optimae(
                            id=task.optimae_id,
                            domain_id=task.domain_id,
                            optimizer_id=task.requester_id,
                            parameters=task.parameters,
                            reported_performance=task.reported_performance or 0,
                            verified_performance=tc.verified_performance,
                            performance_increment=abs(
                                (task.reported_performance or 0) - tc.verified_performance
                            ) if task.reported_performance else 0,
                        )
                        self.consensus.record_optimae(optimae)
                        self.consensus.record_transaction(Transaction(
                            tx_type=TransactionType.OPTIMAE_ACCEPTED,
                            domain_id=task.domain_id,
                            peer_id=tc.evaluator_id,
                            payload={"optimae_id": task.optimae_id, "verified_performance": tc.verified_performance},
                        ))

            # Log task completion transaction
            self.consensus.record_transaction(Transaction(
                tx_type=TransactionType.TASK_COMPLETED,
                domain_id=task.domain_id,
                peer_id=tc.evaluator_id,
                payload={"task_id": tc.task_id, "task_type": task.task_type.value},
            ))

            logger.debug("Task %s completed by %s (via flood)", tc.task_id[:12], tc.evaluator_id[:12])

    # ----------------------------------------------------------------
    # Task queue flooding helpers
    # ----------------------------------------------------------------

    async def _flood_task_created(self, task: Task) -> None:
        """Flood TASK_CREATED event for queue sync across nodes.

        Flooded but NOT recorded on-chain — task creation is an intention,
        not an outcome. Only completed tasks earn chain space. This also
        prevents DoS via spamming inference requests to bloat the chain.
        """
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
        await self.transport.broadcast(list(self._peers.keys()), msg)

    async def _flood_task_claimed(self, task: Task) -> None:
        """Flood TASK_CLAIMED for queue sync (prevent duplicate work).

        Flooded but NOT recorded on-chain — claims are ephemeral
        operational state, only the completion matters permanently.
        """
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
        await self.transport.broadcast(list(self._peers.keys()), msg)

    async def _flood_task_completed(self, task: Task) -> None:
        """Flood TASK_COMPLETED event to the network."""
        tc = TaskCompleted(
            task_id=task.id,
            evaluator_id=task.evaluator_id or "",
            domain_id=task.domain_id,
            verified_performance=task.verified_performance,
            result=task.result,
            optimae_id=task.optimae_id,
        )
        msg = Message(
            msg_type=MessageType.TASK_COMPLETED,
            sender_id=self.peer_id,
            payload=json.loads(tc.model_dump_json()),
        )
        await self.transport.broadcast(list(self._peers.keys()), msg)

    # ----------------------------------------------------------------
    # Client inference request (creates task in queue)
    # ----------------------------------------------------------------

    async def submit_inference_request(
        self, domain_id: str, input_data: dict[str, Any], client_id: str
    ) -> Task:
        """Submit an inference request from a client → creates a task."""
        task = Task(
            task_type=TaskType.INFERENCE_REQUEST,
            domain_id=domain_id,
            requester_id=client_id,
            parameters=input_data,
            priority=10,  # Lower priority than verification
        )
        self.task_queue.add(task)
        await self._flood_task_created(task)
        logger.info("Inference request → task %s for domain %s", task.id[:12], domain_id)
        return task

    # ----------------------------------------------------------------
    # HTTP endpoints for task queue (evaluators pull from here)
    # ----------------------------------------------------------------

    def _register_task_routes(self) -> None:
        """Register task queue HTTP routes on the transport's aiohttp app."""
        app = self.transport._app
        app.router.add_get("/tasks/pending", self._http_tasks_pending)
        app.router.add_post("/tasks/claim", self._http_task_claim)
        app.router.add_post("/tasks/complete", self._http_task_complete)
        app.router.add_post("/inference", self._http_inference_request)

        # Override health to include task queue stats
        # (Can't remove existing route, so we'll add a /status endpoint)
        app.router.add_get("/status", self._http_status)

    async def _http_tasks_pending(self, request: web.Request) -> web.Response:
        """GET /tasks/pending?domains=domain1,domain2&limit=10

        Evaluators call this to get available work.
        """
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

    async def _http_task_claim(self, request: web.Request) -> web.Response:
        """POST /tasks/claim {task_id, evaluator_id}

        Evaluator claims a pending task.
        """
        try:
            data = await request.json()
            task_id = data["task_id"]
            evaluator_id = data["evaluator_id"]
        except (KeyError, json.JSONDecodeError):
            return web.json_response(
                {"status": "error", "detail": "task_id and evaluator_id required"},
                status=400,
            )

        task = self.task_queue.claim(task_id, evaluator_id)
        if task is None:
            return web.json_response(
                {"status": "error", "detail": "task not found or already claimed"},
                status=409,
            )

        await self._flood_task_claimed(task)
        return web.json_response({
            "status": "claimed",
            "task": json.loads(task.model_dump_json()),
        })

    async def _http_task_complete(self, request: web.Request) -> web.Response:
        """POST /tasks/complete {task_id, verified_performance, result}

        Evaluator reports task completion.
        """
        try:
            data = await request.json()
            task_id = data["task_id"]
        except (KeyError, json.JSONDecodeError):
            return web.json_response(
                {"status": "error", "detail": "task_id required"},
                status=400,
            )

        verified_perf = data.get("verified_performance")
        result = data.get("result")

        task = self.task_queue.complete(task_id, verified_performance=verified_perf, result=result)
        if task is None:
            return web.json_response(
                {"status": "error", "detail": "task not found or not claimed"},
                status=409,
            )

        # If verification task, feed result to validator and consensus
        if task.task_type == TaskType.OPTIMAE_VERIFICATION and task.optimae_id:
            if verified_perf is not None:
                val_result = self.validator.record_evaluation(task.optimae_id, verified_perf)
                if val_result.is_valid:
                    optimae = Optimae(
                        id=task.optimae_id,
                        domain_id=task.domain_id,
                        optimizer_id=task.requester_id,
                        parameters=task.parameters,
                        reported_performance=task.reported_performance or 0,
                        verified_performance=verified_perf,
                        performance_increment=abs(verified_perf - (task.reported_performance or 0)),
                    )
                    self.consensus.record_optimae(optimae)
                    self.consensus.record_transaction(Transaction(
                        tx_type=TransactionType.OPTIMAE_ACCEPTED,
                        domain_id=task.domain_id,
                        peer_id=task.evaluator_id or "",
                        payload={"optimae_id": task.optimae_id, "verified_performance": verified_perf},
                    ))

        # Log completion and flood
        self.consensus.record_transaction(Transaction(
            tx_type=TransactionType.TASK_COMPLETED,
            domain_id=task.domain_id,
            peer_id=task.evaluator_id or "",
            payload={"task_id": task_id, "task_type": task.task_type.value},
        ))
        await self._flood_task_completed(task)

        # Try to generate a block after each completion
        block = await self.try_generate_block()

        return web.json_response({
            "status": "completed",
            "task_id": task_id,
            "block_generated": block is not None,
        })

    async def _http_inference_request(self, request: web.Request) -> web.Response:
        """POST /inference {domain_id, input_data, client_id}

        Client submits an inference request → creates a task in the queue.
        """
        try:
            data = await request.json()
            domain_id = data["domain_id"]
            input_data = data.get("input_data", {})
            client_id = data.get("client_id", "anonymous")
        except (KeyError, json.JSONDecodeError):
            return web.json_response(
                {"status": "error", "detail": "domain_id required"},
                status=400,
            )

        if domain_id not in self._domains:
            return web.json_response(
                {"status": "error", "detail": f"unknown domain: {domain_id}"},
                status=404,
            )

        task = await self.submit_inference_request(domain_id, input_data, client_id)
        return web.json_response({
            "status": "queued",
            "task_id": task.id,
        })

    async def _http_status(self, request: web.Request) -> web.Response:
        """GET /status — node status with task queue stats."""
        return web.json_response({
            "status": "healthy",
            "peer_id": self.peer_id[:12],
            "port": self.config.port,
            "chain_height": self.chain.height,
            "domains": list(self._domains.keys()),
            "peers": len(self._peers),
            "task_queue": {
                "pending": self.task_queue.pending_count,
                "claimed": self.task_queue.claimed_count,
                "completed": self.task_queue.completed_count,
            },
        })

    async def try_generate_block(self) -> Block | None:
        """Attempt to generate a new block if the consensus threshold is met."""
        if not self.consensus.can_generate_block():
            return None

        tip = self.chain.tip
        if tip is None:
            logger.error("Cannot generate block: chain not initialized")
            return None

        block = self.consensus.generate_block(tip, self.peer_id)
        if block is None:
            return None

        self.chain.append_block(block)
        self.chain.save()

        # Announce to network
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

        endpoints = list(self._peers.keys())
        await self.transport.broadcast(endpoints, msg)

        logger.info(
            "Generated and announced block #%d (hash=%s)",
            block.header.index,
            block.hash[:12],
        )
        return block
