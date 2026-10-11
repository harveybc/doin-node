"""External-payload publication on a node: PUT -> GET exact -> digest -> reference.

Byte stores here are in-memory test doubles. They do not certify a lake
deployment; the HTTP client is exercised against a fake transport.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from doin_core.archive import ArchiveRefusal
from doin_core.archive.external import build_external_body, parse_locator
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.archive.external import (
    ExternalPayloadArchive,
    HttpByteStore,
    build_reference_transaction,
    canonical_references,
    finality_of,
    verify_chain,
)
from doin_node.blockchain.chain import Chain, ChainError

TS = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


class MemoryStore:
    def __init__(self, *, fail_put=False, tamper_get=False):
        self.objects: dict[str, bytes] = {}
        self.fail_put = fail_put
        self.tamper_get = tamper_get
        self.puts = 0

    def describe(self):
        return {"label": "TEST_MEMORY_STORE", "deployed_lake": False}

    def put_bytes(self, content: bytes) -> str:
        self.puts += 1
        if self.fail_put:
            raise OSError("disk full")
        digest = hashlib.sha256(content).hexdigest()
        self.objects.setdefault(digest, bytes(content))
        return digest

    def get_bytes(self, digest: str) -> bytes:
        blob = self.objects[digest]
        return blob + b" " if self.tamper_get else blob


def _body(n=1):
    return build_external_body(domain_id="dom", producer_peer_id="peer-1",
                               payload_schema="i6n.archive.round.v1",
                               payload={"rows": [{"n": n, "sum_abs_error": 0.5}]})


def _mint(prev: Block, txs, index=None, gen="peer-1", offset=0):
    header = BlockHeader(
        index=prev.header.index + 1 if index is None else index,
        previous_hash=prev.hash,
        timestamp=TS + timedelta(seconds=prev.header.index + 1 + offset),
        merkle_root=compute_merkle_root([tx.id for tx in txs]),
        generator_id=gen,
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    return Block(header=header, transactions=list(txs))


def _chain(txs, extra=6):
    blocks = [Block.genesis()]
    blocks.append(_mint(blocks[-1], txs))
    for _ in range(extra):
        blocks.append(_mint(blocks[-1], []))
    return blocks


def test_store_then_reference_order_and_new_peer_resolves():
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    body = _body()
    ref = archive.store_body(body, mode="external_payload")
    assert set(store.objects) == {body.body_digest, body.manifest_digest}
    assert parse_locator(ref["locator"]) == ("cas", body.body_digest)
    tx = build_reference_transaction(ref, peer_id="peer-1", timestamp=TS)
    assert tx.id == tx.compute_id()
    # Same inputs produce the same transaction id (idempotent resubmission).
    assert build_reference_transaction(ref, peer_id="peer-1", timestamp=TS).id == tx.id
    blocks = _chain([tx])
    verify_chain(blocks)
    refs = canonical_references(blocks, confirmation_depth=6)
    assert [r.tx_id for r in refs] == [tx.id]
    assert refs[0].finality == "FINALIZED"
    # A new peer: fresh archive object, only the chain and read access to the store.
    peer = ExternalPayloadArchive({"cas": store})
    resolved = peer.resolve(refs[0].reference)
    assert resolved.body == body.body
    assert resolved.manifest_digest == body.manifest_digest


def test_failed_put_returns_no_reference():
    store = MemoryStore(fail_put=True)
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    with pytest.raises(OSError):
        archive.store_body(_body(), mode="external_payload")
    assert store.objects == {}


def test_readback_mismatch_returns_no_reference():
    store = MemoryStore(tamper_get=True)
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        archive.store_body(_body(), mode="external_payload")


def test_store_returning_another_digest_is_refused():
    class Liar(MemoryStore):
        def put_bytes(self, content):
            super().put_bytes(content)
            return "0" * 64

    archive = ExternalPayloadArchive({"cas": Liar()}, write_store_id="cas")
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        archive.store_body(_body(), mode="external_payload")


def test_unknown_locator_store_and_wrong_bytes_are_refused():
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    ref = archive.store_body(_body(), mode="external_payload")
    other = ExternalPayloadArchive({"elsewhere": store})
    with pytest.raises(ArchiveRefusal, match="LOCATOR_STORE_UNKNOWN"):
        other.resolve(ref)
    store.tamper_get = True
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        archive.resolve(ref)


def test_full_payload_needs_no_store_and_resolves_identically():
    store = MemoryStore()
    body = _body()
    ext = ExternalPayloadArchive({"cas": store}, write_store_id="cas").store_body(body, mode="external_payload")
    full = ExternalPayloadArchive({}).store_body(body, mode="full_payload")
    a = ExternalPayloadArchive({"cas": store}).resolve(ext)
    b = ExternalPayloadArchive({}).resolve(json.loads(json.dumps(full)))
    assert a.body == b.body and a.payload == b.payload
    assert a.archive_mode != b.archive_mode


def test_chain_refuses_forged_reference_and_broken_links():
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    ref = archive.store_body(_body(), mode="external_payload")
    tx = build_reference_transaction(ref, peer_id="peer-1", timestamp=TS)
    forged = tx.model_copy(update={"payload": dict(ref, manifest_digest="e" * 64)})
    chain = Chain()
    chain.initialize()
    with pytest.raises(ChainError, match="reference"):
        chain.append_block(_mint(chain.tip, [forged]))
    chain.append_block(_mint(chain.tip, [tx]))
    assert chain.height == 2
    blocks = _chain([tx])
    broken = list(blocks)
    broken[3] = _mint(blocks[1], [], index=3)
    with pytest.raises(ArchiveRefusal, match="CHAIN_LINK"):
        verify_chain(broken)
    with pytest.raises(ArchiveRefusal, match="GENESIS"):
        verify_chain(blocks[1:])


def test_plain_candidate_transactions_still_validate():
    tx = Transaction(tx_type=TransactionType.CANDIDATE_EVALUATED, domain_id="dom",
                     peer_id="p", payload={"fitness": 1.0}, timestamp=TS)
    chain = Chain()
    chain.initialize()
    chain.append_block(_mint(chain.tip, [tx]))
    assert chain.height == 2


def test_finality_and_reorg_view():
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    ref = archive.store_body(_body(), mode="external_payload")
    tx = build_reference_transaction(ref, peer_id="peer-1", timestamp=TS)
    shallow = _chain([tx], extra=2)
    refs = canonical_references(shallow, confirmation_depth=6)
    assert refs[0].finality == "CONFIRMING" and refs[0].depth == 2
    assert canonical_references(shallow, confirmation_depth=6, finalized_only=True) == []
    blocks = _chain([tx])
    block_hash = blocks[1].hash
    assert finality_of(blocks, block_hash, confirmation_depth=6) == ("FINALIZED", 6)
    # Competing branch from genesis without the transaction: the old block is not canonical.
    fork = [blocks[0]]
    for i in range(9):
        fork.append(_mint(fork[-1], [], gen="peer-2", offset=100))
    verify_chain(fork)
    assert finality_of(fork, block_hash, confirmation_depth=6) == ("NOT_CANONICAL", None)
    assert canonical_references(fork, confirmation_depth=6) == []


class FakeTransport:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def __call__(self, method, url, data=None, headers=None):
        self.calls.append((method, url, dict(headers or {})))
        digest = url.rsplit("/", 1)[-1]
        if method == "PUT":
            if hashlib.sha256(data).hexdigest() != digest:
                return 400, json.dumps({"error": "HASH_MISMATCH"}).encode()
            created = digest not in self.objects
            self.objects[digest] = data
            return (201 if created else 200), json.dumps({"digest": digest, "size": len(data)}).encode()
        if digest not in self.objects:
            return 404, b'{"error":"MISSING"}'
        return 200, self.objects[digest]


def test_http_byte_store_client_round_trip_without_printing_the_token():
    transport = FakeTransport()
    store = HttpByteStore("http://lake.invalid:1", token_provider=lambda: "s3cret", transport=transport)
    digest = store.put_bytes(b"abc")
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert store.get_bytes(digest) == b"abc"
    assert all(h.get("Authorization") == "Bearer s3cret" for _, _, h in transport.calls)
    assert "s3cret" not in repr(store) and "s3cret" not in json.dumps(store.describe())
    with pytest.raises(ArchiveRefusal, match="MISSING"):
        store.get_bytes("f" * 64)
