"""Shared utilities for benchmark suite."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
SINGLE_CONFIG = EXAMPLES_DIR / "quadratic_single_node.json"


@dataclass
class NodeProcess:
    """Manages a doin-node subprocess."""
    port: int
    data_dir: str
    process: subprocess.Popen | None = None
    bootstrap_peers: list[str] = field(default_factory=list)
    config_path: str = ""

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}"

    def generate_config(self, extra: dict[str, Any] | None = None) -> str:
        """Generate a temporary config file for this node."""
        with open(SINGLE_CONFIG) as f:
            config = json.load(f)

        config["port"] = self.port
        config["data_dir"] = self.data_dir
        config["bootstrap_peers"] = self.bootstrap_peers

        if extra:
            config.update(extra)

        fd, path = tempfile.mkstemp(suffix=".json", prefix=f"bench_node_{self.port}_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        self.config_path = path
        return path

    def start(self) -> None:
        """Start the node subprocess."""
        if not self.config_path:
            self.generate_config()

        self.process = subprocess.Popen(
            [sys.executable, "-m", "doin_node.cli", "--config", self.config_path, "--log-level", "WARNING"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self) -> None:
        """Stop the node subprocess."""
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.config_path and os.path.exists(self.config_path):
            os.unlink(self.config_path)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


async def poll_status(endpoint: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """Poll a node's /status endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{endpoint}/status", timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        return None


async def wait_for_node(endpoint: str, max_wait: float = 30.0) -> bool:
    """Wait until a node responds to /status."""
    start = time.time()
    while time.time() - start < max_wait:
        status = await poll_status(endpoint, timeout=2.0)
        if status is not None:
            return True
        await asyncio.sleep(1.0)
    return False


def create_node_cluster(
    num_nodes: int,
    base_port: int = 8470,
    base_dir: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> list[NodeProcess]:
    """Create a cluster of node processes (not yet started)."""
    if base_dir is None:
        base_dir = tempfile.mkdtemp(prefix="bench_cluster_")

    nodes = []
    for i in range(num_nodes):
        port = base_port + i
        data_dir = os.path.join(base_dir, f"node_{i}")
        os.makedirs(data_dir, exist_ok=True)

        peers = []
        if i > 0:
            peers = [f"localhost:{base_port}"]

        node = NodeProcess(port=port, data_dir=data_dir, bootstrap_peers=peers)
        node.generate_config(extra=extra_config)
        nodes.append(node)

    return nodes


async def start_cluster(nodes: list[NodeProcess], stagger: float = 1.0) -> None:
    """Start all nodes in a cluster with staggered launches."""
    for i, node in enumerate(nodes):
        node.start()
        if i < len(nodes) - 1:
            await asyncio.sleep(stagger)

    # Wait for all nodes to be ready
    for node in nodes:
        ready = await wait_for_node(node.endpoint, max_wait=30.0)
        if not ready:
            print(f"  WARNING: Node on port {node.port} did not become ready")


def stop_cluster(nodes: list[NodeProcess]) -> None:
    """Stop all nodes in a cluster."""
    for node in nodes:
        node.stop()
