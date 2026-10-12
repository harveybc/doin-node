"""External-payload publication and resolution on a node.

Order of a publication (``ExternalPayloadArchive.store_body``):

1. PUT the canonical body bytes and the manifest bytes to the configured
   content-addressed store; each PUT must answer the digest of what was sent;
2. GET both objects back and verify them byte for byte against the reference;
3. only then return the reference payload. The caller builds the reference
   transaction (``build_reference_transaction``) and hands it to consensus.

A failed PUT or a readback that does not hash to the digest raises, so no
reference exists to commit. ``full_payload`` mode skips the store and keeps
the body inside the transaction; ``resolve`` returns identical bytes for both.

The chain helpers (``verify_chain``, ``canonical_references``,
``finality_of``) re-check genesis, heights, previous-hash links, merkle roots,
header hashes and every reference transaction before anything downstream may
treat a reference as finalized. A hash proves integrity, not availability and
not truth: a reference whose bytes cannot be fetched stays unresolved.

The byte stores used by the tests are doubles. Nothing here claims a lake
deployment, and ``chain_verified`` claims of the stage-1 shadow are untouched.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from doin_core.archive.body import ArchiveRefusal, digest_bytes
from doin_core.archive.external import (
    REFERENCE_TX_TYPE,
    ExternalBody,
    ResolvedBody,
    check_reference_payload,
    check_reference_transaction,
    is_reference_transaction,
    parse_locator,
    reference_payload,
    resolve_full_payload,
    verify_external_bytes,
)
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models.block import Block
from doin_core.models.transaction import Transaction

FINALIZED = "FINALIZED"
CONFIRMING = "CONFIRMING"
NOT_CANONICAL = "NOT_CANONICAL"


class GrantedByteStore:
    """Adapter for a store whose calls need a caller grant (``GovernedByteStore``)."""

    def __init__(self, store: Any, grant: str) -> None:
        if not isinstance(grant, str) or not grant:
            raise ArchiveRefusal("GRANT_REQUIRED")
        self._store = store
        self._grant = grant

    def describe(self) -> dict[str, Any]:
        return dict(self._store.describe())

    def put_bytes(self, content: bytes) -> str:
        return self._store.put_bytes(content, grant=self._grant)

    def get_bytes(self, digest: str) -> bytes:
        return self._store.get_bytes(digest, grant=self._grant)

    def __repr__(self) -> str:
        return f"GrantedByteStore({type(self._store).__name__})"


Transport = Callable[..., tuple[int, bytes]]


def _urllib_transport(timeout: float) -> Transport:
    def call(method: str, url: str, data: bytes | None = None, headers: Mapping[str, str] | None = None):
        req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured URL
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    return call


class HttpByteStore:
    """Client of the data-lake host's ``/api/v1/objects/<sha256>`` routes.

    The bearer token comes from ``token_provider`` at call time and is never
    stored in ``repr`` or ``describe``.
    """

    LABEL = "HTTP_LAKE_BYTE_STORE"

    def __init__(
        self,
        base_url: str,
        *,
        token_provider: Callable[[], str],
        transport: Transport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ArchiveRefusal("BYTE_STORE_URL")
        self._base = base_url.rstrip("/")
        self._token = token_provider
        self._transport = transport or _urllib_transport(timeout)

    def __repr__(self) -> str:
        return "HttpByteStore(<configured url>)"

    def describe(self) -> dict[str, Any]:
        return {"label": self.LABEL, "client": "http", "deployed_lake": None}

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": "Bearer " + str(self._token())}
        headers.update(extra or {})
        return headers

    def put_bytes(self, content: bytes) -> str:
        if not isinstance(content, (bytes, bytearray)):
            raise ArchiveRefusal("BYTES_REQUIRED")
        payload = bytes(content)
        digest = digest_bytes(payload)
        status, body = self._call("PUT", digest, payload, {"Content-Type": "application/octet-stream"})
        if status not in (200, 201):
            raise ArchiveRefusal(f"STORE_REFUSED: {status} {_error(body)}")
        try:
            answered = json.loads(body).get("digest")
        except (ValueError, AttributeError) as exc:
            raise ArchiveRefusal("STORE_REPLY_UNREADABLE") from exc
        return answered

    def get_bytes(self, digest: str) -> bytes:
        status, body = self._call("GET", digest, None, None)
        if status == 404:
            raise ArchiveRefusal("MISSING")
        if status != 200:
            raise ArchiveRefusal(f"STORE_REFUSED: {status} {_error(body)}")
        return body

    def _call(self, method: str, digest: str, data: bytes | None, extra: Mapping[str, str] | None):
        url = f"{self._base}/api/v1/objects/{digest}"
        try:
            return self._transport(method, url, data=data, headers=self._headers(extra))
        except OSError as exc:
            raise ArchiveRefusal(f"STORE_UNAVAILABLE: {type(exc).__name__}") from exc


def _error(body: bytes) -> str:
    try:
        return str(json.loads(body).get("error"))[:120]
    except (ValueError, AttributeError):
        return ""


class ExternalPayloadArchive:
    """Write through one configured store; resolve through any configured store by locator."""

    def __init__(self, stores: Mapping[str, Any], *, write_store_id: str | None = None) -> None:
        self._stores = dict(stores)
        if write_store_id is not None and write_store_id not in self._stores:
            raise ArchiveRefusal("LOCATOR_STORE_UNKNOWN")
        self._write = write_store_id

    def store_body(self, external: ExternalBody, *, mode: str) -> dict[str, Any]:
        """PUT, GET exact, compare digests; return the reference only after all three."""
        if mode == "full_payload":
            ref = reference_payload(external, mode=mode)
            resolve_full_payload(ref)
            return ref
        if self._write is None:
            raise ArchiveRefusal("LOCATOR_STORE_UNKNOWN")
        ref = reference_payload(external, mode=mode, store_id=self._write)
        store = self._stores[self._write]
        for content, digest in ((external.body, external.body_digest),
                                (external.manifest, external.manifest_digest)):
            if digest_bytes(content) != digest:
                raise ArchiveRefusal("DIGEST_MISMATCH")
            answered = store.put_bytes(content)
            if answered != digest:
                raise ArchiveRefusal("DIGEST_MISMATCH: store answered another digest")
        self._fetch_verified(store, ref)
        return ref

    def resolve(self, reference: Mapping[str, Any]) -> ResolvedBody:
        ref = check_reference_payload(reference)
        if ref["archive_mode"] == "full_payload":
            return resolve_full_payload(ref)
        store_id, _digest = parse_locator(ref["locator"])
        store = self._stores.get(store_id)
        if store is None:
            raise ArchiveRefusal("LOCATOR_STORE_UNKNOWN")
        return self._fetch_verified(store, ref)

    @staticmethod
    def _fetch_verified(store: Any, ref: Mapping[str, Any]) -> ResolvedBody:
        _store_id, body_digest = parse_locator(ref["locator"])
        try:
            body = store.get_bytes(body_digest)
            manifest = store.get_bytes(ref["manifest_digest"])
        except LookupError as exc:
            raise ArchiveRefusal("MISSING") from exc
        if isinstance(body, bytearray):
            body = bytes(body)
        if isinstance(manifest, bytearray):
            manifest = bytes(manifest)
        return verify_external_bytes(ref, body=body, manifest=manifest)


def build_reference_transaction(
    reference: Mapping[str, Any],
    *,
    peer_id: str,
    timestamp: datetime,
) -> Transaction:
    """A candidate transaction that commits to ``reference``.

    ``timestamp`` is required so a resubmission after a crash yields the same id.
    """
    ref = check_reference_payload(reference)
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ArchiveRefusal("TIMESTAMP_REQUIRED")
    tx = Transaction(
        tx_type=REFERENCE_TX_TYPE,
        domain_id=ref["domain_id"],
        peer_id=peer_id,
        payload=dict(ref),
        timestamp=timestamp,
    )
    check_reference_transaction(tx)
    return tx


def spool_submitter(spool: Any) -> Callable[[Transaction], None]:
    """Return an idempotent, fsyncing writer for the node reference spool.

    The filename is the transaction identity. A retry with identical bytes is
    a no-op; the same identity with different bytes is a hard conflict.
    """
    from pathlib import Path

    root = Path(spool)
    root.mkdir(parents=True, exist_ok=True)

    def submit(tx: Transaction) -> None:
        check_reference_transaction(tx)
        data = tx.model_dump_json().encode("utf-8")
        target = root / f"{tx.id}.json"
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if target.read_bytes() != data:
                raise ArchiveRefusal("SPOOL_CONFLICT: same tx id, different bytes")
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    return submit


@dataclass(frozen=True)
class ChainReference:
    block_hash: str
    height: int
    tx_index: int
    tx_id: str
    reference: dict[str, Any]
    finality: str
    depth: int


def verify_chain(blocks: Sequence[Block]) -> None:
    """Refuse a block list that is not one linked chain from a genesis block."""
    if not blocks:
        raise ArchiveRefusal("GENESIS")
    first = blocks[0]
    if (
        first.header.index != 0
        or first.header.previous_hash != "0" * 64
        or first.hash != first.header.compute_hash()
    ):
        raise ArchiveRefusal("GENESIS")
    prev = first
    for block in blocks[1:]:
        if block.header.index != prev.header.index + 1 or block.header.previous_hash != prev.hash:
            raise ArchiveRefusal(f"CHAIN_LINK at height {block.header.index}")
        if block.header.merkle_root != compute_merkle_root([tx.id for tx in block.transactions]):
            raise ArchiveRefusal(f"MERKLE_ROOT at height {block.header.index}")
        if block.hash != block.header.compute_hash():
            raise ArchiveRefusal(f"HEADER_HASH at height {block.header.index}")
        for tx in block.transactions:
            if is_reference_transaction(tx):
                check_reference_transaction(tx)
        prev = block


def finality_of(
    blocks: Sequence[Block],
    block_hash: str,
    *,
    confirmation_depth: int,
) -> tuple[str, int | None]:
    """Finality of one block on the verified chain ``blocks``."""
    verify_chain(blocks)
    tip = blocks[-1].header.index
    for block in blocks:
        if block.hash == block_hash:
            depth = tip - block.header.index
            return (FINALIZED if depth >= confirmation_depth else CONFIRMING), depth
    return NOT_CANONICAL, None


def canonical_references(
    blocks: Sequence[Block],
    *,
    confirmation_depth: int,
    finalized_only: bool = False,
) -> list[ChainReference]:
    """Every reference transaction on the verified chain, in chain order."""
    if isinstance(confirmation_depth, bool) or not isinstance(confirmation_depth, int) or confirmation_depth < 1:
        raise ArchiveRefusal("CONFIRMATION_DEPTH")
    verify_chain(blocks)
    tip = blocks[-1].header.index
    found: list[ChainReference] = []
    for block in blocks[1:]:
        depth = tip - block.header.index
        finality = FINALIZED if depth >= confirmation_depth else CONFIRMING
        if finalized_only and finality != FINALIZED:
            continue
        for index, tx in enumerate(block.transactions):
            if is_reference_transaction(tx):
                found.append(
                    ChainReference(
                        block_hash=block.hash,
                        height=block.header.index,
                        tx_index=index,
                        tx_id=tx.id,
                        reference=dict(check_reference_transaction(tx)),
                        finality=finality,
                        depth=depth,
                    )
                )
    return found


def ingest_reference_spool(
    spool: Any,
    record_transaction: Callable[[Transaction], Any],
    *,
    known_tx_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Hand spooled reference transactions (``<tx_id>.json``) to consensus, once each.

    Each file is parsed, its id re-derived and its reference checked before
    ``record_transaction`` sees it; a refused file is moved to ``refused/`` with
    the reason, an accepted one to ``ingested/``. Nothing is fetched and no
    validation is skipped: blocks carrying these transactions still go through
    ``Chain._validate_block``.
    """
    from pathlib import Path

    root = Path(spool)
    done, refused = root / "ingested", root / "refused"
    done.mkdir(parents=True, exist_ok=True)
    refused.mkdir(parents=True, exist_ok=True)
    seen = known_tx_ids if known_tx_ids is not None else set()
    summary: dict[str, Any] = {"ingested": 0, "already_known": 0, "refused": []}
    for path in sorted(root.glob("*.json")):
        try:
            tx = Transaction.model_validate_json(path.read_bytes())
            check_reference_transaction(tx)
            if path.stem != tx.id:
                raise ArchiveRefusal("SPOOL_NAME_IS_NOT_TX_ID")
        except (ArchiveRefusal, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"[:300]
            (refused / (path.name + ".reason")).write_text(reason)
            path.replace(refused / path.name)
            summary["refused"].append({"file": path.name, "reason": reason})
            continue
        if tx.id in seen:
            summary["already_known"] += 1
        else:
            record_transaction(tx)
            seen.add(tx.id)
            summary["ingested"] += 1
        path.replace(done / path.name)
    return summary
