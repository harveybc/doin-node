"""Transport layer — async TCP connections between peers.

Uses aiohttp for HTTP-based message exchange. Each node runs a small
HTTP server and sends messages to peers via POST requests.
This is simple, debuggable, and sufficient for the initial implementation.
A libp2p-based transport can replace this later.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

from aiohttp import ClientSession, ClientTimeout, web

from doin_core.protocol.messages import Message

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = ClientTimeout(total=10)


class Transport:
    """HTTP-based transport for peer-to-peer communication.

    Runs an aiohttp server for incoming messages and uses aiohttp client
    sessions for outgoing messages. Messages are JSON-serialized.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8470,
    ) -> None:
        self.host = host
        self.port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._session: ClientSession | None = None
        self._message_callback: Callable[..., Coroutine[Any, Any, None]] | None = None

        # Register routes
        self._app.router.add_post("/message", self._handle_message)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/peers", self._handle_peers)

    def on_message(
        self,
        callback: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Set the callback for incoming messages.

        Args:
            callback: Async callable receiving (Message, sender_endpoint).
        """
        self._message_callback = callback

    async def start(self) -> None:
        """Start the HTTP server and client session."""
        self._session = ClientSession(timeout=DEFAULT_TIMEOUT)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("Transport listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down transport."""
        if self._session:
            await self._session.close()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Transport stopped")

    async def send(self, endpoint: str, message: Message) -> bool:
        """Send a message to a peer.

        Args:
            endpoint: Peer endpoint (host:port).
            message: Message to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._session:
            logger.error("Transport not started")
            return False

        url = f"http://{endpoint}/message"
        try:
            async with self._session.post(
                url,
                json=json.loads(message.model_dump_json()),
                headers={"Content-Type": "application/json"},
            ) as resp:
                return resp.status == 200
        except Exception:
            logger.debug("Failed to send to %s", endpoint, exc_info=True)
            return False

    async def broadcast(
        self,
        endpoints: list[str],
        message: Message,
        exclude: str | None = None,
    ) -> dict[str, bool]:
        """Send a message to multiple peers concurrently.

        Args:
            endpoints: List of peer endpoints.
            message: Message to broadcast.
            exclude: Optional endpoint to skip (e.g., the sender).

        Returns:
            Dict mapping endpoint to success/failure.
        """
        targets = [ep for ep in endpoints if ep != exclude]
        tasks = [self.send(ep, message) for ep in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            ep: isinstance(r, bool) and r
            for ep, r in zip(targets, results)
        }

    async def _handle_message(self, request: web.Request) -> web.Response:
        """Handle incoming message POST."""
        try:
            data = await request.json()
            message = Message.model_validate(data)

            if self._message_callback:
                sender = request.remote or "unknown"
                await self._message_callback(message, sender)

            return web.json_response({"status": "ok"})
        except Exception:
            logger.exception("Error handling incoming message")
            return web.json_response(
                {"status": "error", "detail": "invalid message"},
                status=400,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "port": self.port})

    async def _handle_peers(self, request: web.Request) -> web.Response:
        """Placeholder for peer listing endpoint."""
        return web.json_response({"peers": []})
