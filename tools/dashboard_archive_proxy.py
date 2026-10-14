"""Read-only dashboard proxy with verified hash-only metric resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiohttp import ClientSession, web

from doin_node.cli import load_config
from doin_node.dashboard.routes import _blockchain_metrics_payload, _dumps
from doin_node.unified import UnifiedNode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    return parser


async def _metrics(request: web.Request) -> web.Response:
    node = request.app["node"]
    limit = int(request.query.get("limit", "500"))
    payload = _blockchain_metrics_payload(node, limit)
    return web.json_response(payload, dumps=_dumps)


async def _dashboard(request: web.Request) -> web.Response:
    template = (
        Path(__file__).resolve().parents[1]
        / "src/doin_node/dashboard/templates/dashboard.html"
    )
    return web.Response(text=template.read_text(), content_type="text/html")


async def _proxy(request: web.Request) -> web.Response:
    upstream = request.app["upstream"] + request.rel_url.path_qs
    session = request.app["session"]
    body = await request.read()
    async with session.request(request.method, upstream, data=body) as response:
        content = await response.read()
        headers = {}
        if response.content_type:
            headers["Content-Type"] = response.headers.get("Content-Type", response.content_type)
        return web.Response(status=response.status, body=content, headers=headers)


async def _close(app: web.Application) -> None:
    await app["session"].close()
    chaindb = app["node"].chaindb
    if chaindb is not None:
        chaindb.close()


async def _build_app(args: argparse.Namespace) -> web.Application:
    config = load_config(args.config, {})
    node = UnifiedNode(config)
    if node.chaindb is None:
        raise SystemExit("SQLite chain storage is required")
    node.chaindb.open()

    app = web.Application()
    app["node"] = node
    app["upstream"] = args.upstream.rstrip("/")
    app["session"] = ClientSession()
    app.router.add_get("/dashboard", _dashboard)
    app.router.add_get("/api/metrics", _metrics)
    app.router.add_route("*", "/{path:.*}", _proxy)
    app.on_cleanup.append(_close)
    return app


def main() -> None:
    args = _parser().parse_args()
    web.run_app(_build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
