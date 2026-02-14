"""GPU Scheduler — resource marketplace for optimization and inference.

Nodes advertise their available compute resources (GPU type, memory, etc.).
The scheduler matches work requests to the best available resources based on:
  - Resource requirements (GPU memory, compute units)
  - Geographic proximity (latency-sensitive inference)
  - Price (bid/ask market for compute time)
  - Reputation (prefer reliable nodes)
  - Domain specialization (nodes with cached models)

This enables a decentralized compute marketplace where:
  - Optimizers can rent GPU time for training
  - Evaluators get assigned to nodes with matching hardware
  - Inference requests route to the lowest-latency capable node
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


PRIORITY_VALUES = {
    JobPriority.LOW: 0,
    JobPriority.NORMAL: 1,
    JobPriority.HIGH: 2,
    JobPriority.URGENT: 3,
}


@dataclass
class GPUInfo:
    """Description of a GPU resource."""

    name: str = "unknown"
    memory_gb: float = 0.0
    compute_units: int = 0
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class NodeResources:
    """Resources advertised by a node."""

    peer_id: str
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_cores: int = 0
    ram_gb: float = 0.0
    disk_gb: float = 0.0
    domains: list[str] = field(default_factory=list)  # Domains with cached models
    region: str = ""  # Geographic region
    price_per_hour: float = 0.0  # DOIN per hour of compute
    reputation: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    available: bool = True
    current_jobs: int = 0
    max_concurrent_jobs: int = 1

    @property
    def total_gpu_memory(self) -> float:
        return sum(g.memory_gb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def has_capacity(self) -> bool:
        return self.available and self.current_jobs < self.max_concurrent_jobs

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_heartbeat) > 300  # 5 min


@dataclass
class ResourceRequirement:
    """Resources required for a job."""

    min_gpu_memory_gb: float = 0.0
    min_gpu_count: int = 0
    min_cpu_cores: int = 1
    min_ram_gb: float = 1.0
    min_disk_gb: float = 0.0
    preferred_region: str = ""
    preferred_domain: str = ""  # Prefer nodes with this domain cached
    max_price_per_hour: float = float("inf")


@dataclass
class ComputeJob:
    """A job requesting compute resources."""

    job_id: str
    requester_id: str
    domain_id: str
    job_type: str  # "optimization", "evaluation", "inference"
    requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    priority: JobPriority = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    assigned_to: str = ""
    started_at: float = 0.0
    estimated_duration_hours: float = 1.0
    max_bid: float = 0.0  # Max DOIN willing to pay

    @property
    def is_assigned(self) -> bool:
        return bool(self.assigned_to)

    @property
    def wait_time(self) -> float:
        return time.time() - self.created_at


@dataclass
class SchedulerConfig:
    """Configuration for the GPU scheduler."""

    max_queue_size: int = 1000
    stale_job_timeout: float = 3600.0  # 1 hour
    prefer_cached_domain_bonus: float = 0.3  # 30% bonus for domain match
    prefer_region_bonus: float = 0.2  # 20% bonus for region match
    reputation_weight: float = 0.2
    price_weight: float = 0.3
    capacity_weight: float = 0.3


class GPUScheduler:
    """Decentralized compute resource scheduler.

    Matches jobs to nodes based on requirements, price, reputation,
    and resource availability.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._nodes: dict[str, NodeResources] = {}
        self._queue: list[tuple[int, float, str]] = []  # (-priority, created_at, job_id)
        self._jobs: dict[str, ComputeJob] = {}
        self._completed: int = 0
        self._total_compute_hours: float = 0.0

    # ── Node management ──────────────────────────────────────────

    def register_node(self, resources: NodeResources) -> None:
        """Register or update a node's available resources."""
        self._nodes[resources.peer_id] = resources

    def update_heartbeat(self, peer_id: str) -> None:
        node = self._nodes.get(peer_id)
        if node:
            node.last_heartbeat = time.time()

    def set_node_availability(self, peer_id: str, available: bool) -> None:
        node = self._nodes.get(peer_id)
        if node:
            node.available = available

    def remove_node(self, peer_id: str) -> None:
        self._nodes.pop(peer_id, None)

    # ── Job submission ───────────────────────────────────────────

    def submit_job(self, job: ComputeJob) -> tuple[bool, str]:
        """Submit a compute job to the queue.

        Returns (success, error_message).
        """
        if len(self._jobs) >= self.config.max_queue_size:
            return False, "Job queue full"
        if job.job_id in self._jobs:
            return False, f"Job {job.job_id} already exists"

        self._jobs[job.job_id] = job
        priority_val = PRIORITY_VALUES.get(job.priority, 1)
        heapq.heappush(self._queue, (-priority_val, job.created_at, job.job_id))
        return True, ""

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and not job.is_assigned:
            del self._jobs[job_id]
            return True
        return False

    # ── Scheduling ───────────────────────────────────────────────

    def schedule_next(self) -> tuple[ComputeJob | None, str]:
        """Assign the highest-priority unassigned job to the best node.

        Returns (job, assigned_node_id) or (None, "") if nothing can be scheduled.
        """
        # Find next unassigned job
        while self._queue:
            _, _, job_id = self._queue[0]
            job = self._jobs.get(job_id)
            if job is None or job.is_assigned:
                heapq.heappop(self._queue)
                continue

            # Find best node
            best_node = self._find_best_node(job)
            if best_node is None:
                break  # No available nodes

            heapq.heappop(self._queue)
            job.assigned_to = best_node.peer_id
            job.started_at = time.time()
            best_node.current_jobs += 1
            return job, best_node.peer_id

        return None, ""

    def schedule_all(self) -> list[tuple[str, str]]:
        """Schedule as many jobs as possible. Returns [(job_id, node_id)]."""
        assignments = []
        while True:
            job, node_id = self.schedule_next()
            if job is None:
                break
            assignments.append((job.job_id, node_id))
        return assignments

    def complete_job(self, job_id: str, duration_hours: float = 0.0) -> float:
        """Mark a job as complete. Returns the compute cost in DOIN."""
        job = self._jobs.get(job_id)
        if job is None or not job.is_assigned:
            return 0.0

        node = self._nodes.get(job.assigned_to)
        if node:
            node.current_jobs = max(0, node.current_jobs - 1)

        hours = duration_hours or job.estimated_duration_hours
        cost = (node.price_per_hour if node else 0) * hours

        self._completed += 1
        self._total_compute_hours += hours
        del self._jobs[job_id]

        return cost

    # ── Matching ─────────────────────────────────────────────────

    def _find_best_node(self, job: ComputeJob) -> NodeResources | None:
        """Find the best available node for a job."""
        candidates = []
        for node in self._nodes.values():
            if not node.has_capacity or node.is_stale:
                continue
            if not self._meets_requirements(node, job.requirements):
                continue
            score = self._score_node(node, job)
            candidates.append((score, node))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _meets_requirements(
        self, node: NodeResources, req: ResourceRequirement,
    ) -> bool:
        """Check if a node meets minimum requirements."""
        if req.min_gpu_count > 0 and node.gpu_count < req.min_gpu_count:
            return False
        if req.min_gpu_memory_gb > 0 and node.total_gpu_memory < req.min_gpu_memory_gb:
            return False
        if node.cpu_cores < req.min_cpu_cores:
            return False
        if node.ram_gb < req.min_ram_gb:
            return False
        if node.disk_gb < req.min_disk_gb:
            return False
        if node.price_per_hour > req.max_price_per_hour:
            return False
        return True

    def _score_node(self, node: NodeResources, job: ComputeJob) -> float:
        """Score a node for a job (higher = better match)."""
        score = 0.0
        req = job.requirements

        # Price score (lower price = higher score)
        if node.price_per_hour > 0:
            max_price = req.max_price_per_hour if req.max_price_per_hour < float("inf") else node.price_per_hour * 2
            price_score = 1.0 - (node.price_per_hour / max_price) if max_price > 0 else 0.5
            score += price_score * self.config.price_weight

        # Reputation score
        score += min(node.reputation / 10.0, 1.0) * self.config.reputation_weight

        # Capacity score (fewer current jobs = better)
        if node.max_concurrent_jobs > 0:
            capacity_score = 1.0 - (node.current_jobs / node.max_concurrent_jobs)
            score += capacity_score * self.config.capacity_weight

        # Domain cache bonus
        if req.preferred_domain and req.preferred_domain in node.domains:
            score += self.config.prefer_cached_domain_bonus

        # Region bonus
        if req.preferred_region and req.preferred_region == node.region:
            score += self.config.prefer_region_bonus

        return score

    # ── Queries ──────────────────────────────────────────────────

    def get_available_nodes(
        self, requirements: ResourceRequirement | None = None,
    ) -> list[NodeResources]:
        """Get nodes that have capacity and optionally meet requirements."""
        nodes = [n for n in self._nodes.values() if n.has_capacity and not n.is_stale]
        if requirements:
            nodes = [n for n in nodes if self._meets_requirements(n, requirements)]
        return nodes

    def get_queue_stats(self) -> dict[str, Any]:
        pending = sum(1 for j in self._jobs.values() if not j.is_assigned)
        running = sum(1 for j in self._jobs.values() if j.is_assigned)
        return {
            "pending_jobs": pending,
            "running_jobs": running,
            "registered_nodes": len(self._nodes),
            "available_nodes": sum(1 for n in self._nodes.values() if n.has_capacity),
            "completed_jobs": self._completed,
            "total_compute_hours": self._total_compute_hours,
            "total_gpu_memory_gb": sum(n.total_gpu_memory for n in self._nodes.values()),
        }

    def cleanup_stale(self) -> int:
        """Remove stale nodes and expired jobs."""
        removed = 0
        # Stale nodes
        stale = [pid for pid, n in self._nodes.items() if n.is_stale]
        for pid in stale:
            del self._nodes[pid]
            removed += 1
        # Expired jobs
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if not j.is_assigned and j.wait_time > self.config.stale_job_timeout
        ]
        for jid in expired:
            del self._jobs[jid]
            removed += 1
        return removed
