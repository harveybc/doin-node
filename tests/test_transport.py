"""Focused transport resource-bound tests."""

import pytest

from doin_node.network.transport import (
    OUTBOUND_CONNECTION_LIMIT,
    OUTBOUND_CONNECTIONS_PER_HOST,
    Transport,
)


@pytest.mark.asyncio
async def test_transport_uses_bounded_outbound_connection_pool(unused_tcp_port: int) -> None:
    transport = Transport(host="127.0.0.1", port=unused_tcp_port)
    await transport.start()
    try:
        assert transport._session is not None
        connector = transport._session.connector
        assert connector is not None
        assert connector.limit == OUTBOUND_CONNECTION_LIMIT
        assert connector.limit_per_host == OUTBOUND_CONNECTIONS_PER_HOST
        payload = await transport.get_json(
            f"http://127.0.0.1:{unused_tcp_port}/health"
        )
        assert payload == {"status": "healthy", "port": unused_tcp_port}
    finally:
        await transport.stop()
