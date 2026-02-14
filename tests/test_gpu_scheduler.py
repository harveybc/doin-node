"""Tests for the GPU scheduler / resource marketplace."""

import time
import pytest

from doin_node.scheduling.gpu_scheduler import (
    ComputeJob,
    GPUInfo,
    GPUScheduler,
    JobPriority,
    NodeResources,
    ResourceRequirement,
    SchedulerConfig,
)


def make_node(peer_id: str, gpus: int = 1, memory: float = 8.0, price: float = 1.0, **kwargs) -> NodeResources:
    return NodeResources(
        peer_id=peer_id,
        gpus=[GPUInfo(name="RTX 4090", memory_gb=memory)] * gpus,
        cpu_cores=kwargs.get("cpu_cores", 8),
        ram_gb=kwargs.get("ram_gb", 32.0),
        disk_gb=kwargs.get("disk_gb", 100.0),
        price_per_hour=price,
        reputation=kwargs.get("reputation", 5.0),
        domains=kwargs.get("domains", []),
        region=kwargs.get("region", ""),
        max_concurrent_jobs=kwargs.get("max_jobs", 2),
    )


def make_job(job_id: str, **kwargs) -> ComputeJob:
    return ComputeJob(
        job_id=job_id,
        requester_id=kwargs.get("requester", "alice"),
        domain_id=kwargs.get("domain", "test"),
        job_type=kwargs.get("job_type", "optimization"),
        priority=kwargs.get("priority", JobPriority.NORMAL),
        requirements=kwargs.get("requirements", ResourceRequirement()),
    )


class TestNodeManagement:

    def test_register_node(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1"))
        assert len(sched.get_available_nodes()) == 1

    def test_remove_node(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1"))
        sched.remove_node("node-1")
        assert len(sched.get_available_nodes()) == 0

    def test_set_unavailable(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1"))
        sched.set_node_availability("node-1", False)
        assert len(sched.get_available_nodes()) == 0

    def test_heartbeat_updates(self):
        sched = GPUScheduler()
        node = make_node("node-1")
        node.last_heartbeat = time.time() - 1000
        sched.register_node(node)
        assert sched._nodes["node-1"].is_stale
        sched.update_heartbeat("node-1")
        assert not sched._nodes["node-1"].is_stale


class TestJobSubmission:

    def test_submit_job(self):
        sched = GPUScheduler()
        ok, err = sched.submit_job(make_job("job-1"))
        assert ok
        assert err == ""

    def test_duplicate_job(self):
        sched = GPUScheduler()
        sched.submit_job(make_job("job-1"))
        ok, err = sched.submit_job(make_job("job-1"))
        assert not ok
        assert "already exists" in err

    def test_queue_full(self):
        sched = GPUScheduler(SchedulerConfig(max_queue_size=2))
        sched.submit_job(make_job("job-1"))
        sched.submit_job(make_job("job-2"))
        ok, err = sched.submit_job(make_job("job-3"))
        assert not ok
        assert "full" in err

    def test_cancel_job(self):
        sched = GPUScheduler()
        sched.submit_job(make_job("job-1"))
        assert sched.cancel_job("job-1")


class TestScheduling:

    def test_schedule_basic(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1"))
        sched.submit_job(make_job("job-1"))
        job, node_id = sched.schedule_next()
        assert job is not None
        assert node_id == "node-1"
        assert job.is_assigned

    def test_no_available_nodes(self):
        sched = GPUScheduler()
        sched.submit_job(make_job("job-1"))
        job, _ = sched.schedule_next()
        assert job is None

    def test_priority_order(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1", max_jobs=10))
        sched.submit_job(make_job("low", priority=JobPriority.LOW))
        sched.submit_job(make_job("urgent", priority=JobPriority.URGENT))
        sched.submit_job(make_job("normal", priority=JobPriority.NORMAL))

        assignments = sched.schedule_all()
        ids = [a[0] for a in assignments]
        assert ids[0] == "urgent"
        assert ids[-1] == "low"

    def test_gpu_requirement_filter(self):
        sched = GPUScheduler()
        sched.register_node(make_node("small", gpus=1, memory=4.0))
        sched.register_node(make_node("big", gpus=2, memory=24.0))

        req = ResourceRequirement(min_gpu_count=2, min_gpu_memory_gb=40.0)
        job = make_job("job-1", requirements=req)
        sched.submit_job(job)
        result, node_id = sched.schedule_next()
        assert result is not None
        assert node_id == "big"

    def test_price_filter(self):
        sched = GPUScheduler()
        sched.register_node(make_node("cheap", price=1.0))
        sched.register_node(make_node("expensive", price=100.0))

        req = ResourceRequirement(max_price_per_hour=5.0)
        sched.submit_job(make_job("job-1", requirements=req))
        _, node_id = sched.schedule_next()
        assert node_id == "cheap"

    def test_domain_preference(self):
        sched = GPUScheduler()
        sched.register_node(make_node("generic", price=1.0, reputation=5.0))
        sched.register_node(make_node("specialized", price=1.0, reputation=5.0, domains=["ml-vision"]))

        req = ResourceRequirement(preferred_domain="ml-vision")
        sched.submit_job(make_job("job-1", requirements=req))
        _, node_id = sched.schedule_next()
        assert node_id == "specialized"

    def test_region_preference(self):
        sched = GPUScheduler()
        sched.register_node(make_node("us", price=1.0, reputation=5.0, region="us-east"))
        sched.register_node(make_node("eu", price=1.0, reputation=5.0, region="eu-west"))

        req = ResourceRequirement(preferred_region="eu-west")
        sched.submit_job(make_job("job-1", requirements=req))
        _, node_id = sched.schedule_next()
        assert node_id == "eu"

    def test_capacity_respected(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1", max_jobs=1))
        sched.submit_job(make_job("job-1"))
        sched.submit_job(make_job("job-2"))

        sched.schedule_next()  # Fills node-1
        job, _ = sched.schedule_next()
        assert job is None  # No capacity

    def test_complete_job(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1", price=2.0))
        sched.submit_job(make_job("job-1"))
        sched.schedule_next()
        cost = sched.complete_job("job-1", duration_hours=0.5)
        assert cost == 1.0  # 2.0/hr * 0.5hr
        stats = sched.get_queue_stats()
        assert stats["completed_jobs"] == 1

    def test_schedule_all(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1", max_jobs=3))
        for i in range(5):
            sched.submit_job(make_job(f"job-{i}"))
        assignments = sched.schedule_all()
        assert len(assignments) == 3  # Node capacity = 3


class TestStats:

    def test_queue_stats(self):
        sched = GPUScheduler()
        sched.register_node(make_node("node-1"))
        sched.submit_job(make_job("job-1"))
        sched.submit_job(make_job("job-2"))
        sched.schedule_next()

        stats = sched.get_queue_stats()
        assert stats["pending_jobs"] == 1
        assert stats["running_jobs"] == 1
        assert stats["registered_nodes"] == 1

    def test_cleanup_stale(self):
        sched = GPUScheduler(SchedulerConfig(stale_job_timeout=0.01))
        node = make_node("old-node")
        node.last_heartbeat = time.time() - 600
        sched.register_node(node)
        sched.submit_job(make_job("old-job"))
        time.sleep(0.02)

        removed = sched.cleanup_stale()
        assert removed == 2  # 1 node + 1 job
