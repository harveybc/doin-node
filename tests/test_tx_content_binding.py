"""Adversarial tests: both validators bind transaction IDs to content.

Order reference: MUSASHI_TO_GENERAL_SATOSHI_III_BLOCKCHAIN_AND_FOUR_FRONT
_CORRECTION_ORDER_2026_08_10.md §4 (WP1) — node-side mandatory cases:

- supplied arbitrary ID with valid body is rejected (both validators,
  independent of Pydantic construction — forged txs are built with
  ``model_construct`` to bypass model validation entirely);
- body mutation with original ID is rejected on load/verification;
- Merkle root cannot be made valid from forged IDs;
- duplicate IDs with different bodies refuse;
- the auditor's forged-ID append + tamper + next-block sequence refuses.

All databases are temporary (pytest tmp_path). No real chain is touched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import (
    Transaction,
    TransactionIntegrityError,
    TransactionType,
)
from doin_node.blockchain.chain import Chain, ChainError, ChainIntegrityError
from doin_node.storage.chaindb import ChainDB

TS = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _valid_tx(n: int = 0) -> Transaction:
    return Transaction(
        tx_type=TransactionType.OPTIMAE_ANNOUNCED,
        domain_id="eth-usd-4h",
        peer_id=f"peer-{n}",
        payload={"optimae_id": f"opt-{n}", "reported_performance": 0.5 + n / 100},
        timestamp=TS,
    )


def _forged_tx(n: int = 0, forged_id: str | None = None) -> Transaction:
    """Build a transaction whose ID is NOT its content hash.

    Construction-time validation refuses forged IDs (and ``model_construct``
    also runs ``model_post_init``, so it refuses too). The remaining bypass
    is post-construction attribute mutation — exactly what a hostile peer or
    corrupted store amounts to. The validators must catch it independently.
    """
    tx = _valid_tx(n)
    tx.id = forged_id if forged_id is not None else ("f" * 63 + f"{n:x}")
    return tx


def _block_on(prev: Block, txs: list[Transaction], merkle: str | None = None) -> Block:
    header = BlockHeader(
        index=prev.header.index + 1,
        previous_hash=prev.hash,
        timestamp=TS,
        merkle_root=merkle
        if merkle is not None
        else compute_merkle_root([tx.id for tx in txs]),
        generator_id="gen",
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    return Block(header=header, transactions=txs)


@pytest.fixture()
def db(tmp_path):
    chaindb = ChainDB(tmp_path / "chain.sqlite")
    chaindb.open()
    genesis = chaindb.initialize("genesis")
    yield chaindb, genesis
    chaindb.close()


@pytest.fixture()
def mem(tmp_path):
    chain = Chain(data_dir=tmp_path)
    genesis = chain.initialize("genesis")
    return chain, genesis


# ── Forged supplied IDs rejected by both validators ──────────────────


class TestForgedIdRejected:
    def test_chaindb_rejects_forged_id(self, db) -> None:
        chaindb, genesis = db
        forged = _forged_tx()
        block = _block_on(genesis, [forged])  # merkle over the forged ID
        with pytest.raises(TransactionIntegrityError) as exc_info:
            chaindb.append_block(block)
        assert exc_info.value.block_index == 1
        assert exc_info.value.tx_index == 0
        assert chaindb.height == 1  # nothing appended

    def test_chain_rejects_forged_id(self, mem) -> None:
        chain, genesis = mem
        forged = _forged_tx()
        block = _block_on(genesis, [forged])
        with pytest.raises(ChainIntegrityError) as exc_info:
            chain.append_block(block)
        assert exc_info.value.block_index == 1
        assert chain.height == 1

    def test_chain_integrity_error_is_a_chain_error(self, mem) -> None:
        # sync paths catch ChainError; the typed refusal must not escape
        # as an unrelated crash type.
        assert issubclass(ChainIntegrityError, ChainError)
        chain, genesis = mem
        appended = chain.validate_and_append_blocks(
            [_block_on(genesis, [_forged_tx()])]
        )
        assert appended == 0
        assert chain.height == 1

    def test_error_contains_no_payload(self, db) -> None:
        chaindb, genesis = db
        forged = Transaction(
            tx_type=TransactionType.OPTIMAE_ANNOUNCED,
            domain_id="d",
            peer_id="p",
            payload={"secret": "DO-NOT-LEAK"},
            timestamp=TS,
        )
        forged.id = "a" * 64
        with pytest.raises(TransactionIntegrityError) as exc_info:
            chaindb.append_block(_block_on(genesis, [forged]))
        assert "DO-NOT-LEAK" not in str(exc_info.value)

    @pytest.mark.parametrize("bad_id", ["F" * 64, "f" * 10, ""])
    def test_malformed_ids_rejected(self, db, bad_id: str) -> None:
        chaindb, genesis = db
        forged = _forged_tx(forged_id=bad_id)
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(_block_on(genesis, [forged]))


# ── Merkle root cannot be made valid from forged IDs ─────────────────


class TestMerkleFromForgedIds:
    def test_chaindb_merkle_over_forged_ids_refuses(self, db) -> None:
        chaindb, genesis = db
        forged = [_forged_tx(0), _forged_tx(1)]
        # Attacker computes a "consistent" merkle root over the forged IDs.
        merkle = compute_merkle_root([tx.id for tx in forged])
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(_block_on(genesis, forged, merkle=merkle))

    def test_chain_merkle_over_forged_ids_refuses(self, mem) -> None:
        chain, genesis = mem
        forged = [_forged_tx(0), _forged_tx(1)]
        merkle = compute_merkle_root([tx.id for tx in forged])
        with pytest.raises(ChainIntegrityError):
            chain.append_block(_block_on(genesis, forged, merkle=merkle))

    def test_merkle_over_true_hashes_with_forged_tx_refuses(self, db) -> None:
        chaindb, genesis = db
        forged = _forged_tx()
        true_hash = forged.compute_id()
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(
                _block_on(genesis, [forged], merkle=compute_merkle_root([true_hash]))
            )

    def test_valid_block_still_appends(self, db) -> None:
        chaindb, genesis = db
        block = _block_on(genesis, [_valid_tx(0), _valid_tx(1)])
        chaindb.append_block(block)
        assert chaindb.height == 2


# ── Body mutation with original ID rejected on load ──────────────────


class TestBodyMutationRejectedOnLoad:
    def test_chaindb_payload_tamper_refused_on_load(self, db) -> None:
        chaindb, genesis = db
        tx = _valid_tx()
        chaindb.append_block(_block_on(genesis, [tx]))

        chaindb._conn.execute(
            "UPDATE transactions SET payload = ? WHERE tx_id = ?",
            (json.dumps({"optimae_id": "opt-0", "reported_performance": 0.01}), tx.id),
        )
        with pytest.raises(TransactionIntegrityError) as exc_info:
            chaindb.get_block(1)
        assert exc_info.value.block_index == 1
        assert exc_info.value.tx_index == 0
        assert "0.01" not in str(exc_info.value)  # no payload dump

    @pytest.mark.parametrize(
        "column,value",
        [
            ("tx_type", "optimae_accepted"),
            ("domain_id", "tampered-domain"),
            ("peer_id", "tampered-peer"),
            ("timestamp", "2026-08-11T00:00:00+00:00"),
        ],
    )
    def test_chaindb_field_tamper_refused_on_load(self, db, column, value) -> None:
        chaindb, genesis = db
        tx = _valid_tx()
        chaindb.append_block(_block_on(genesis, [tx]))
        chaindb._conn.execute(
            f"UPDATE transactions SET {column} = ? WHERE tx_id = ?", (value, tx.id)
        )
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_block(1)

    def test_chaindb_tamper_refused_via_query_paths(self, db) -> None:
        chaindb, genesis = db
        tx = _valid_tx()
        chaindb.append_block(_block_on(genesis, [tx]))
        chaindb._conn.execute(
            "UPDATE transactions SET payload = '{}' WHERE tx_id = ?", (tx.id,)
        )
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_transactions(1)
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_transactions_by_peer(tx.peer_id)
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_transactions_by_type(tx.tx_type.value)

    def test_chain_json_tamper_refused_on_load(self, mem, tmp_path) -> None:
        chain, genesis = mem
        tx = _valid_tx()
        chain.append_block(_block_on(genesis, [tx]))
        chain.save()

        chain_file = tmp_path / "chain.json"
        data = json.loads(chain_file.read_text())
        data[1]["transactions"][0]["payload"]["reported_performance"] = 0.01
        chain_file.write_text(json.dumps(data))

        fresh = Chain(data_dir=tmp_path)
        with pytest.raises(ChainIntegrityError) as exc_info:
            fresh.load()
        assert exc_info.value.block_index == 1
        assert "0.01" not in str(exc_info.value)


# ── Duplicate IDs refuse ─────────────────────────────────────────────


class TestDuplicateIds:
    def test_duplicate_id_different_bodies_refuses_chaindb(self, db) -> None:
        chaindb, genesis = db
        honest = _valid_tx()
        # Same ID, different body: with content binding the imposter's ID
        # cannot match its own content, so the block refuses.
        imposter = _valid_tx(1)
        imposter.id = honest.id
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(_block_on(genesis, [honest, imposter]))

    def test_duplicate_id_same_body_within_block_refuses(self, db) -> None:
        chaindb, genesis = db
        tx = _valid_tx()
        dup = _valid_tx()  # identical content → identical derived ID
        assert dup.id == tx.id
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(_block_on(genesis, [tx, dup]))

    def test_duplicate_id_across_blocks_refuses_chaindb(self, db) -> None:
        chaindb, genesis = db
        tx = _valid_tx()
        block1 = _block_on(genesis, [tx])
        chaindb.append_block(block1)
        # Same transaction replayed in the next block: refused by the
        # tx_id PRIMARY KEY, atomically (block not appended).
        block2 = _block_on(block1, [tx])
        with pytest.raises(Exception):
            chaindb.append_block(block2)
        assert chaindb.height == 2

    def test_duplicate_id_within_block_refuses_chain(self, mem) -> None:
        chain, genesis = mem
        tx = _valid_tx()
        dup = _valid_tx()  # identical content → identical derived ID
        with pytest.raises(ChainIntegrityError):
            chain.append_block(_block_on(genesis, [tx, dup]))


# ── The auditor's full counterexample now refuses ────────────────────


class TestFinding201Counterexample:
    def test_forged_append_tamper_next_block_refuses(self, db) -> None:
        chaindb, genesis = db

        # Step 1: forged-ID transaction cannot even be appended.
        forged = _forged_tx()
        with pytest.raises(TransactionIntegrityError):
            chaindb.append_block(_block_on(genesis, [forged]))
        assert chaindb.height == 1

        # Step 2: honest block appends; then the persisted body is tampered.
        tx = _valid_tx()
        block1 = _block_on(genesis, [tx])
        chaindb.append_block(block1)
        chaindb._conn.execute(
            "UPDATE transactions SET payload = ? WHERE tx_id = ?",
            (json.dumps({"tampered": True}), tx.id),
        )

        # Step 3: the tamper is detected on load — typed, with coordinates.
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_block(1)

        # Step 4: appending the next block still works structurally (the
        # tip transition uses headers), but the tampered history can no
        # longer be read or re-validated — the forgery is detectable at
        # every load, unlike before the fix.
        tx2 = _valid_tx(2)
        chaindb.append_block(_block_on(block1, [tx2]))
        with pytest.raises(TransactionIntegrityError):
            chaindb.get_block(1)
