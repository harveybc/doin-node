"""WP2 mandatory integration tests: full chain verification + chain identity.

Order reference: MUSASHI_TO_GENERAL_SATOSHI_III_BLOCKCHAIN_AND_FOUR_FRONT
_CORRECTION_ORDER_2026_08_10.md §5 (WP2), findings 202-203. The ten
mandatory cases, in order:

 1. valid restart;
 2. corrupted historical body;
 3. corrupted header;
 4. missing transaction row;
 5. duplicate/gapped tx index;
 6. wrong genesis;
 7. wrong chain-ID peer (rejected before block exchange);
 8. pruned suffix (typed, never fully verified);
 9. reorg plus OLAP reprojection (deterministic);
10. restart refusal before any network/optimizer side effect.

All databases are temporary (pytest tmp_path). No real chain is touched.
No external network is used — peers are localhost aiohttp servers.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiohttp import ClientSession, web

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.models.verification import ChainVerificationOutcome
from doin_core.protocol.messages import PROTOCOL_VERSION
from doin_node.blockchain.verify import (
    ChainStartupRefused,
    main as verify_main,
    verify_chain_db,
)
from doin_node.stats.olap_db import OLAPDatabase, OLAPProvenanceError
from doin_node.storage.chaindb import ChainDB
from doin_node.unified import UnifiedNode, UnifiedNodeConfig

TS = datetime(2026, 8, 10, tzinfo=timezone.utc)


# ── Helpers (temp DBs only) ──────────────────────────────────────────

def _tx(n: int, *, domain: str = "eth-usd-4h", tx_type=TransactionType.OPTIMAE_ANNOUNCED,
        payload: dict | None = None) -> Transaction:
    return Transaction(
        tx_type=tx_type,
        domain_id=domain,
        peer_id=f"peer-{n}",
        payload=payload if payload is not None else {"n": n},
        timestamp=TS,
    )


def _block_on(prev: Block, txs: list[Transaction]) -> Block:
    header = BlockHeader(
        index=prev.header.index + 1,
        previous_hash=prev.hash,
        timestamp=TS,
        merkle_root=compute_merkle_root([tx.id for tx in txs]),
        generator_id="gen",
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    return Block(header=header, transactions=txs)


def _build_chain(db_path: Path, n_blocks: int = 4, txs_per_block: int = 2) -> str:
    """Create a valid chain DB with *n_blocks* appended blocks.

    Returns the genesis hash.
    """
    db = ChainDB(db_path)
    db.open()
    genesis = db.initialize("genesis")
    prev = genesis
    for i in range(1, n_blocks + 1):
        txs = [_tx(i * 100 + j) for j in range(txs_per_block)]
        block = _block_on(prev, txs)
        db.append_block(block)
        prev = block
    db.close()
    return genesis.hash


def _corrupt(db_path: Path, stmt: str, params: tuple = ()) -> None:
    """Mutate the SQLite file directly, bypassing every validator."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(stmt, params)
    conn.commit()
    conn.close()


def _node_config(tmp_path: Path, port: int, **overrides) -> UnifiedNodeConfig:
    fields = dict(
        port=port,
        data_dir=str(tmp_path / "node"),
        db_path=str(tmp_path / "chain.db"),
        storage_backend="sqlite",
        dashboard_enabled=False,
        discovery_enabled=False,
        fee_market_enabled=False,
    )
    fields.update(overrides)
    return UnifiedNodeConfig(**fields)


# ── 1. Valid restart ─────────────────────────────────────────────────

class TestValidRestart:
    def test_verifier_passes_valid_chain(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        genesis_hash = _build_chain(db_path)
        report = verify_chain_db(db_path, expected_genesis_hash=genesis_hash)
        assert report.outcome is ChainVerificationOutcome.FULLY_VERIFIED
        assert report.ok
        assert report.height == 5
        assert [c.status.value for c in report.checks] == ["pass"] * 10

    async def test_node_restarts_on_valid_chain(self, tmp_path) -> None:
        _build_chain(tmp_path / "chain.db")
        for boot in range(2):  # first boot stamps identity, second re-verifies it
            node = UnifiedNode(_node_config(tmp_path, port=18471))
            await node.start()
            try:
                report = node.chain_verification_report
                assert report is not None and report.ok
                assert report.outcome is ChainVerificationOutcome.FULLY_VERIFIED
                assert node.quarantine_report is None
                stored = node.chaindb.get_chain_identity()
                assert stored == (node.chain_id, node.expected_genesis_hash)
            finally:
                await node.stop()


# ── 2. Corrupted historical body ─────────────────────────────────────

class TestCorruptedHistoricalBody:
    def test_body_mutation_detected_with_coordinates(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE transactions SET payload = ? WHERE block_index = 2 AND tx_index = 0",
            (json.dumps({"n": 999999}),),
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure is not None
        assert report.first_failure.check_number == 7
        assert report.first_failure.block_index == 2
        assert report.first_failure.tx_index == 0
        # no payload dump in the typed report
        assert "999999" not in report.model_dump_json()


# ── 3. Corrupted header ──────────────────────────────────────────────

class TestCorruptedHeader:
    def test_header_mutation_detected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE blocks SET generator_id = 'evil' WHERE block_index = 3",
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 4
        assert report.first_failure.block_index == 3

    def test_merkle_root_mutation_detected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        # Forge merkle root AND recompute a consistent header hash so the
        # corruption survives check 4 — check 8 must still catch it.
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM blocks WHERE block_index = 2"
        ).fetchone()
        import hashlib
        forged_merkle = "f" * 64
        forged_hash = hashlib.sha256(json.dumps({
            "index": row["block_index"],
            "previous_hash": row["previous_hash"],
            "timestamp": row["timestamp"],
            "merkle_root": forged_merkle,
            "generator_id": row["generator_id"],
            "weighted_performance_sum": row["weighted_performance_sum"],
            "threshold": row["threshold"],
        }, sort_keys=True).encode()).hexdigest()
        conn.execute(
            "UPDATE blocks SET merkle_root = ?, hash = ? WHERE block_index = 2",
            (forged_merkle, forged_hash),
        )
        conn.commit()
        conn.close()
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        # the forged header hash breaks the previous_hash chain at block 3
        # (check 5) — or, were it the tip, the merkle check (8) refuses.
        assert report.first_failure.check_number in (5, 8, 10)


# ── 4. Missing transaction row ───────────────────────────────────────

class TestMissingTxRow:
    def test_missing_row_detected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "DELETE FROM transactions WHERE block_index = 2 AND tx_index = 1",
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6
        assert report.first_failure.block_index == 2

    def test_all_rows_missing_without_provenance_is_failure(self, tmp_path) -> None:
        # bodies absent entirely + no pruning provenance = corruption
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(db_path, "DELETE FROM transactions WHERE block_index = 1")
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6
        assert report.first_failure.block_index == 1


# ── 5. Duplicate / gapped tx index ───────────────────────────────────

class TestDuplicateGappedTxIndex:
    def test_duplicate_tx_index_detected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE transactions SET tx_index = 0 WHERE block_index = 3 AND tx_index = 1",
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6
        assert report.first_failure.block_index == 3

    def test_gapped_tx_index_detected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE transactions SET tx_index = 7 WHERE block_index = 3 AND tx_index = 1",
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6
        assert report.first_failure.block_index == 3


# ── 6. Wrong genesis ─────────────────────────────────────────────────

class TestWrongGenesis:
    def test_verifier_rejects_wrong_expected_genesis(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        report = verify_chain_db(db_path, expected_genesis_hash="f" * 64)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 3
        assert report.first_failure.block_index == 0

    def test_non_deterministic_genesis_rejected(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE blocks SET previous_hash = ? WHERE block_index = 0",
            ("1" * 64,),
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 2
        assert report.first_failure.block_index == 0

    async def test_node_refuses_wrong_configured_genesis(self, tmp_path) -> None:
        _build_chain(tmp_path / "chain.db")
        node = UnifiedNode(
            _node_config(tmp_path, port=18472, genesis_hash="f" * 64)
        )
        with pytest.raises(ChainStartupRefused) as e:
            await node.start()
        assert e.value.report.first_failure.check_number == 3
        await node.stop()

    def test_cli_exit_codes(self, tmp_path, capsys) -> None:
        db_path = tmp_path / "chain.db"
        genesis_hash = _build_chain(db_path)
        assert verify_main(
            ["--db", str(db_path), "--expect-genesis", genesis_hash]
        ) == 0
        capsys.readouterr()
        assert verify_main(
            ["--db", str(db_path), "--expect-genesis", "f" * 64]
        ) == 2
        capsys.readouterr()
        # expected chain-id but no attestation → refused (exit 3)
        assert verify_main(
            ["--db", str(db_path), "--expect-chain-id", "doin-x"]
        ) == 3
        capsys.readouterr()


# ── 7. Wrong chain-ID peer rejected before block exchange ────────────

class _FakePeer:
    """Minimal localhost peer serving a configurable /chain/status."""

    def __init__(self, status: dict) -> None:
        self.status = status
        self.block_requests: list[str] = []
        self._runner: web.AppRunner | None = None
        self.port: int = 0

    async def start(self, port: int) -> None:
        app = web.Application()
        app.router.add_get("/chain/status", self._status)
        app.router.add_get("/chain/blocks", self._blocks)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", port)
        await site.start()
        self.port = port

    async def _status(self, request) -> web.Response:
        return web.json_response(self.status)

    async def _blocks(self, request) -> web.Response:
        self.block_requests.append(str(request.rel_url))
        return web.json_response({"request_id": "", "blocks": [], "has_more": False})

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()


class TestWrongChainIdPeer:
    async def _run_case(self, tmp_path, port_base: int, peer_status_overrides: dict,
                        expect_rejected: bool) -> tuple[UnifiedNode, _FakePeer]:
        _build_chain(tmp_path / "chain.db")
        node = UnifiedNode(_node_config(tmp_path, port=port_base))
        await node.start()
        status = {
            "chain_height": 99,
            "tip_hash": "b" * 64,
            "tip_index": 98,
            "finalized_height": 0,
            "protocol_version": PROTOCOL_VERSION,
            "chain_id": node.chain_id,
            "genesis_hash": node.expected_genesis_hash,
        }
        status.update(peer_status_overrides)
        peer = _FakePeer(status)
        await peer.start(port_base + 1)
        endpoint = f"127.0.0.1:{peer.port}"
        node.add_peer("127.0.0.1", peer.port)
        await node._initial_sync()
        if expect_rejected:
            assert endpoint in node._identity_rejected_peers
            assert peer.block_requests == []  # NO block exchange happened
            assert any(
                a["category"] == "chain_identity" for a in node._alerts
            )
        return node, peer

    async def test_wrong_chain_id_rejected_before_block_exchange(self, tmp_path) -> None:
        node, peer = await self._run_case(
            tmp_path, 18474, {"chain_id": "doin-other-net"}, expect_rejected=True
        )
        try:
            # the direct sync path refuses too
            await node._sync_with_peer(f"127.0.0.1:{peer.port}")
            assert peer.block_requests == []
        finally:
            await peer.stop()
            await node.stop()

    async def test_wrong_genesis_peer_rejected(self, tmp_path) -> None:
        node, peer = await self._run_case(
            tmp_path, 18476, {"genesis_hash": "c" * 64}, expect_rejected=True
        )
        await peer.stop()
        await node.stop()

    async def test_legacy_unversioned_peer_refused_not_default_accepted(
        self, tmp_path
    ) -> None:
        # A legacy peer that omits identity fields must be refused: the
        # ChainStatus defaults parse, but never count as acceptance.
        _build_chain(tmp_path / "chain.db")
        node = UnifiedNode(_node_config(tmp_path, port=18478))
        await node.start()
        peer = _FakePeer({
            "chain_height": 99,
            "tip_hash": "b" * 64,
            "tip_index": 98,
            "finalized_height": 0,
        })
        await peer.start(18479)
        try:
            node.add_peer("127.0.0.1", peer.port)
            await node._initial_sync()
            assert f"127.0.0.1:{peer.port}" in node._identity_rejected_peers
            assert peer.block_requests == []
        finally:
            await peer.stop()
            await node.stop()

    async def test_matching_peer_is_not_rejected(self, tmp_path) -> None:
        node, peer = await self._run_case(tmp_path, 18480, {}, expect_rejected=False)
        try:
            ok = await node._verify_peer_chain_identity(f"127.0.0.1:{peer.port}")
            assert ok
            assert f"127.0.0.1:{peer.port}" not in node._identity_rejected_peers
        finally:
            await peer.stop()
            await node.stop()

    async def test_our_status_attests_identity(self, tmp_path) -> None:
        _build_chain(tmp_path / "chain.db")
        node = UnifiedNode(_node_config(tmp_path, port=18482))
        await node.start()
        try:
            async with ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{node.config.port}/chain/status"
                ) as resp:
                    data = await resp.json()
            assert data["protocol_version"] == PROTOCOL_VERSION
            assert data["chain_id"] == node.chain_id
            assert data["genesis_hash"] == node.expected_genesis_hash
        finally:
            await node.stop()


# ── 8. Pruned suffix ─────────────────────────────────────────────────

class TestPrunedSuffix:
    def test_pruned_chain_is_suffix_verified_never_full(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path, n_blocks=5)
        db = ChainDB(db_path)
        db.open()
        pruned = db.prune_transactions_before(3)
        assert pruned == 4  # blocks 1 and 2, two txs each
        checkpoint = db.get_pruning_checkpoint()
        db.close()
        assert checkpoint["pruned_before_index"] == 3
        assert checkpoint["checkpoint_block_index"] == 2

        report = verify_chain_db(db_path)
        assert report.outcome is (
            ChainVerificationOutcome.VERIFIED_SUFFIX_FROM_CHECKPOINT
        )
        assert report.outcome is not ChainVerificationOutcome.FULLY_VERIFIED
        assert report.ok
        suffix = report.verified_suffix
        assert suffix is not None
        assert suffix.suffix_start_index == 3
        assert suffix.suffix_end_index == 5
        assert suffix.checkpoint_block_index == 2
        assert suffix.pruned_body_blocks == 2  # blocks 1 and 2

    def test_pruned_without_checkpoint_commitment_is_refused(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path, n_blocks=5)
        db = ChainDB(db_path)
        db.open()
        db.prune_transactions_before(3)
        db.close()
        # strip the commitment: absence of required provenance = refusal
        _corrupt(db_path, "DELETE FROM metadata WHERE key = 'checkpoint_block_hash'")
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.REFUSED
        assert report.first_failure.check_number == 6

    def test_pruned_with_forged_commitment_fails(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path, n_blocks=5)
        db = ChainDB(db_path)
        db.open()
        db.prune_transactions_before(3)
        db.close()
        _corrupt(
            db_path,
            "UPDATE metadata SET value = ? WHERE key = 'checkpoint_block_hash'",
            ("e" * 64,),
        )
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6

    def test_missing_body_inside_suffix_still_fails(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path, n_blocks=5)
        db = ChainDB(db_path)
        db.open()
        db.prune_transactions_before(2)
        db.close()
        # remove a body INSIDE the retained suffix — pruning provenance
        # must not excuse it
        _corrupt(db_path, "DELETE FROM transactions WHERE block_index = 4")
        report = verify_chain_db(db_path)
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 6
        assert report.first_failure.block_index == 4


# ── 9. Reorg plus OLAP reprojection ──────────────────────────────────

def _optimae_tx(n: int, optimae_id: str) -> Transaction:
    return _tx(
        n,
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        payload={
            "optimae_id": optimae_id,
            "verified_performance": 0.5 + n / 1000,
            "experiment_id": "exp-reorg",
            "round_number": n,
            "time_to_this_result_seconds": 1.0,
            "optimization_config_hash": "c" * 64,
            "data_hash": "",
            "reported_performance": 0.5,
            "previous_best_performance": None,
        },
    )


class TestReorgOlapReprojection:
    def _project(self, olap: OLAPDatabase, db: ChainDB, chain_id: str,
                 genesis_hash: str) -> None:
        blocks = db.get_blocks_range(0, db.height - 1)
        olap.ingest_from_chain(
            blocks,
            chain_id=chain_id,
            genesis_hash=genesis_hash,
            source_tip_hash=db.tip_hash,
            source_height=db.height,
        )

    def _rows(self, olap: OLAPDatabase) -> set[tuple[str, int]]:
        return {
            (r["optimae_id"], r["block_height"])
            for r in olap._conn.execute(
                "SELECT optimae_id, block_height FROM fact_chain_optimae"
            ).fetchall()
        }

    def test_reorg_invalidates_and_reprojects_deterministically(self, tmp_path) -> None:
        db = ChainDB(tmp_path / "chain.db")
        db.open()
        genesis = db.initialize("genesis")
        b1 = _block_on(genesis, [_optimae_tx(1, "opt-a1")])
        b2 = _block_on(b1, [_optimae_tx(2, "opt-a2")])
        b3 = _block_on(b2, [_optimae_tx(3, "opt-a3")])
        for b in (b1, b2, b3):
            db.append_block(b)

        olap = OLAPDatabase(tmp_path / "olap.db")
        chain_id, genesis_hash = "doin-reorg-test", genesis.hash
        self._project(olap, db, chain_id, genesis_hash)
        assert self._rows(olap) == {("opt-a1", 1), ("opt-a2", 2), ("opt-a3", 3)}
        prov = olap.get_chain_provenance()
        assert prov["source_tip_hash"] == b3.hash and prov["source_height"] == 4

        # ── reorg: roll back to block 1, adopt a different branch ──
        db.rollback_to(1)
        olap.invalidate_chain_rows_above(
            1,
            chain_id=chain_id,
            genesis_hash=genesis_hash,
            new_tip_hash=db.tip_hash,
            new_height=db.height,
        )
        assert self._rows(olap) == {("opt-a1", 1)}

        c2 = _block_on(b1, [_optimae_tx(20, "opt-b2")])
        c3 = _block_on(c2, [_optimae_tx(30, "opt-b3")])
        c4 = _block_on(c3, [_optimae_tx(40, "opt-b4")])
        for b in (c2, c3, c4):
            db.append_block(b)
        self._project(olap, db, chain_id, genesis_hash)
        expected = {("opt-a1", 1), ("opt-b2", 2), ("opt-b3", 3), ("opt-b4", 4)}
        assert self._rows(olap) == expected

        # determinism: reprojection from the same chain changes nothing
        self._project(olap, db, chain_id, genesis_hash)
        assert self._rows(olap) == expected
        prov = olap.get_chain_provenance()
        assert prov["source_tip_hash"] == c4.hash and prov["source_height"] == 5

        # the reorged chain itself still fully verifies
        db.close()
        report = verify_chain_db(tmp_path / "chain.db")
        assert report.outcome is ChainVerificationOutcome.FULLY_VERIFIED
        olap.close()

    def test_unbound_ingestion_refused(self, tmp_path) -> None:
        olap = OLAPDatabase(tmp_path / "olap.db")
        with pytest.raises(OLAPProvenanceError):
            olap.ingest_from_chain([])
        olap.close()

    def test_cross_chain_ingestion_refused(self, tmp_path) -> None:
        olap = OLAPDatabase(tmp_path / "olap.db")
        olap.ingest_from_chain(
            [], chain_id="doin-a", genesis_hash="a" * 64,
            source_tip_hash="b" * 64, source_height=1,
        )
        with pytest.raises(OLAPProvenanceError):
            olap.ingest_from_chain(
                [], chain_id="doin-b", genesis_hash="d" * 64,
                source_tip_hash="b" * 64, source_height=1,
            )
        with pytest.raises(OLAPProvenanceError):
            olap.invalidate_chain_rows_above(
                0, chain_id="doin-b", genesis_hash="d" * 64,
                new_tip_hash="", new_height=1,
            )
        olap.close()

    def test_node_reorg_helpers_bind_provenance(self, tmp_path) -> None:
        """The runtime helpers project/invalidate through the same binding."""
        node = UnifiedNode(_node_config(tmp_path, port=18484))
        node.chaindb.open()
        genesis = node.chaindb.initialize("genesis")
        b1 = _block_on(genesis, [_optimae_tx(1, "opt-n1")])
        b2 = _block_on(b1, [_optimae_tx(2, "opt-n2")])
        node.chaindb.append_block(b1)
        node.chaindb.append_block(b2)

        node._olap_project_chain()
        olap = node.experiment_tracker._olap
        assert self._rows(olap) == {("opt-n1", 1), ("opt-n2", 2)}
        prov = olap.get_chain_provenance()
        assert prov["chain_id"] == node.chain_id
        assert prov["genesis_hash"] == node.expected_genesis_hash

        node.chaindb.rollback_to(1)
        node._olap_invalidate_reorg(1)
        assert self._rows(olap) == {("opt-n1", 1)}

        c2 = _block_on(b1, [_optimae_tx(20, "opt-n2b")])
        node.chaindb.append_block(c2)
        node._olap_project_chain()
        assert self._rows(olap) == {("opt-n1", 1), ("opt-n2b", 2)}
        node.chaindb.close()
        node.experiment_tracker.finalize()


# ── 10. Restart refusal before any network/optimizer side effect ─────

class TestRestartRefusalBeforeSideEffects:
    async def test_refusal_precedes_all_side_effects(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE transactions SET payload = ? WHERE block_index = 1 AND tx_index = 0",
            (json.dumps({"evil": True}),),
        )
        node = UnifiedNode(_node_config(tmp_path, port=18486))

        transport_started = []
        original_transport_start = node.transport.start

        async def spy_start():
            transport_started.append(True)
            await original_transport_start()

        node.transport.start = spy_start  # type: ignore[method-assign]

        with pytest.raises(ChainStartupRefused) as e:
            await node.start()

        report = e.value.report
        assert report.outcome is ChainVerificationOutcome.FAILED
        assert report.first_failure.check_number == 7
        assert report.first_failure.block_index == 1

        # NO side effects happened before the typed refusal:
        assert transport_started == []            # no network listener
        assert node._background_tasks == []       # no optimizer/evaluator/gossip loops
        assert node.sync_manager.peers == {}      # no sync state
        assert node._peers == {}                  # no bootstrap connections
        assert node.experiment_tracker._experiments == {}  # no experiment started
        # no OLAP experiment projection either
        olap = node.experiment_tracker._olap
        n_exp = olap._conn.execute(
            "SELECT COUNT(*) FROM dim_experiment"
        ).fetchone()[0]
        assert n_exp == 0
        await node.stop()

    async def test_quarantine_mode_is_read_only_diagnostic(self, tmp_path) -> None:
        db_path = tmp_path / "chain.db"
        _build_chain(db_path)
        _corrupt(
            db_path,
            "UPDATE blocks SET generator_id = 'evil' WHERE block_index = 2",
        )
        node = UnifiedNode(
            _node_config(
                tmp_path, port=18488, on_verification_failure="quarantine"
            )
        )
        await node.start()
        try:
            assert node.quarantine_report is not None
            assert node._background_tasks == []   # no loops at all
            assert (await node.try_generate_block()) is None  # append refused

            async with ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{node.config.port}/status"
                ) as resp:
                    status = await resp.json()
                assert status["state"] == "quarantined"
                assert status["outcome"] == "failed"
                async with session.get(
                    f"http://127.0.0.1:{node.config.port}/chain/status"
                ) as resp:
                    assert resp.status == 503  # chain never served for sync
                async with session.get(
                    f"http://127.0.0.1:{node.config.port}/verify/report"
                ) as resp:
                    report = await resp.json()
                assert report["outcome"] == "failed"
                assert report["first_failure"]["block_index"] == 2
        finally:
            await node.stop()
