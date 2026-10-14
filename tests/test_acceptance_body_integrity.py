"""Accepted bodies must match their core IDs at storage boundaries."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models.block import Block, BlockHeader
from doin_core.models.transaction import Transaction, TransactionType
from doin_node.blockchain.chain import Chain, ChainError
from doin_node.storage.chaindb import ChainDB
from doin_node.storage.migrate import migrate_json_to_sqlite


TS = datetime(2026, 10, 13, tzinfo=timezone.utc)


def accepted_tx():
    return Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="integrity-test",
        peer_id="test-peer",
        payload={"verified_performance": -0.2, "parameters": {"seed": 42}},
        timestamp=TS,
    )


def mint(previous, tx=None):
    tx = accepted_tx() if tx is None else tx
    return Block(
        header=BlockHeader(
            index=previous.header.index + 1,
            previous_hash=previous.hash,
            timestamp=TS + timedelta(seconds=previous.header.index + 1),
            merkle_root=compute_merkle_root([tx.id]),
            generator_id="test-peer",
            weighted_performance_sum=0.1,
            threshold=0.01,
        ),
        transactions=[tx],
    )


@pytest.fixture(params=["json", "sqlite"])
def storage(request, tmp_path):
    if request.param == "json":
        store = Chain(tmp_path)
    else:
        store = ChainDB(tmp_path / "chain.db")
        store.open()
    store.initialize()
    yield store
    if isinstance(store, ChainDB):
        store.close()


def snapshot(store):
    if isinstance(store, ChainDB):
        return list(store._conn.iterdump()), store._conn.total_changes
    return [b.model_dump_json() for b in store.blocks], dict(store._block_index)


def append_many(store, blocks):
    if isinstance(store, ChainDB):
        return store.append_blocks(blocks)
    return store.validate_and_append_blocks(blocks)


@pytest.mark.parametrize(
    "field", ["payload", "nested", "domain_id", "peer_id", "timestamp", "id"]
)
def test_append_rejects_changed_accepted_body_without_writes(storage, field):
    block = mint(storage.get_block(0))
    tx = block.transactions[0]
    if field == "payload":
        tx.payload["verified_performance"] = -999.0
    elif field == "nested":
        tx.payload["parameters"]["seed"] = 99
    elif field == "timestamp":
        tx.timestamp += timedelta(seconds=1)
    else:
        setattr(tx, field, "altered")
    # Rebind the header to isolate body integrity from existing Merkle checks.
    block.header.merkle_root = compute_merkle_root([tx.id])
    block.hash = block.header.compute_hash()
    before = snapshot(storage)
    with pytest.raises((ChainError, ValueError), match="accepted transaction"):
        storage.append_block(block)
    assert snapshot(storage) == before
    assert storage.height == 1
    assert not storage.has_transaction(tx.id)


def test_batch_rejects_first_invalid_body_without_writes(storage):
    block = mint(storage.get_block(0))
    block.transactions[0].payload["parameters"]["seed"] = 99
    before = snapshot(storage)
    assert append_many(storage, [block]) == 0
    assert snapshot(storage) == before


@pytest.mark.parametrize("unhashable", [False, True])
def test_invalid_later_transaction_does_not_partially_write_block(storage, unhashable):
    block = mint(storage.get_block(0))
    tx = accepted_tx()
    tx.timestamp += timedelta(seconds=1)
    tx.id = tx.compute_id()
    block.transactions.append(tx)
    block.header.merkle_root = compute_merkle_root([t.id for t in block.transactions])
    block.hash = block.header.compute_hash()
    tx.payload["parameters"]["seed"] = {99} if unhashable else 99
    before = snapshot(storage)
    with pytest.raises((ChainError, ValueError), match="accepted transaction"):
        storage.append_block(block)
    assert snapshot(storage) == before


def test_batch_preserves_existing_valid_prefix_contract(storage):
    first = mint(storage.get_block(0))
    tx = accepted_tx()
    tx.timestamp += timedelta(seconds=1)
    tx.id = tx.compute_id()
    second = mint(first, tx)
    second.transactions[0].payload["verified_performance"] = -999.0
    assert append_many(storage, [first, second]) == 1
    assert storage.height == 2
    assert storage.get_block(1) == first
    assert storage.get_block(2) is None
    assert not storage.has_transaction(tx.id)


@pytest.mark.parametrize("tx_type", [
    TransactionType.OPTIMAE_ACCEPTED,
    TransactionType.DOMAIN_REGISTERED,
    TransactionType.CANDIDATE_EVALUATED,
])
def test_valid_round_trip_preserves_other_types_custom_ids(storage, tmp_path, tx_type):
    tx = accepted_tx()
    tx.tx_type = tx_type
    tx.id = (
        tx.compute_id() if tx_type == TransactionType.OPTIMAE_ACCEPTED
        else "custom-scheme-id"
    )
    block = mint(storage.get_block(0), tx)
    storage.append_block(block)
    if isinstance(storage, Chain):
        storage.save()
        restored = Chain(tmp_path)
        restored.load()
        assert restored.get_block(1) == block
    else:
        storage.close()
        storage.open()
        assert storage.get_block(1) == block


def test_json_load_invalid_suffix_leaves_existing_chain_untouched(tmp_path):
    chain = Chain(tmp_path)
    chain.initialize("local-genesis")
    first = mint(Block.genesis())
    second = mint(first)
    second.transactions[0].payload["parameters"]["seed"] = 99
    path = tmp_path / "incoming.json"
    incoming = [Block.genesis(), first, second]
    path.write_text(json.dumps([b.model_dump(mode="json") for b in incoming]))
    before = snapshot(chain)
    original = path.read_bytes()
    with pytest.raises(ChainError, match="accepted transaction"):
        chain.load(path)
    assert snapshot(chain) == before
    assert path.read_bytes() == original


@pytest.mark.parametrize("existing", [False, True])
def test_json_save_mutated_body_does_not_write(tmp_path, existing):
    chain = Chain()
    chain.initialize()
    block = mint(chain.tip)
    chain.append_block(block)
    path = tmp_path / "new-directory" / "chain.json"
    if existing:
        chain.save(path)
    before = path.read_bytes() if existing else None
    block.transactions[0].payload["parameters"]["seed"] = 99
    with pytest.raises(ChainError, match="accepted transaction"):
        chain.save(path)
    if existing:
        assert path.read_bytes() == before
    else:
        assert not path.parent.exists()


@pytest.mark.parametrize("reader", [
    "get_block", "get_tip", "get_transactions",
    "get_transactions_by_peer", "get_transactions_by_type",
])
def test_sqlite_rejects_corrupted_persisted_body(tmp_path, reader):
    db = ChainDB(tmp_path / "chain.db")
    db.open()
    try:
        db.initialize()
        block = mint(db.get_block(0))
        db.append_block(block)
        db._conn.execute("UPDATE transactions SET payload = ?", ('{"tampered": true}',))
        before = snapshot(db)
        args = {
            "get_block": (1,), "get_tip": (), "get_transactions": (1,),
            "get_transactions_by_peer": ("test-peer",),
            "get_transactions_by_type": ("optimae_accepted",),
        }
        with pytest.raises(ValueError, match="accepted transaction"):
            getattr(db, reader)(*args[reader])
        assert snapshot(db) == before
    finally:
        db.close()


def test_json_migration_rejects_mutated_body(tmp_path):
    genesis = Block.genesis()
    block = mint(genesis)
    block.transactions[0].payload["parameters"]["seed"] = 99
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        b.model_dump(mode="json") for b in [genesis, block]
    ]))
    destination = tmp_path / "chain.db"
    assert migrate_json_to_sqlite(source, destination) == 1
    db = ChainDB(destination)
    db.open()
    try:
        assert db.height == 1
        assert not db.has_transaction(block.transactions[0].id)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fork_adoption_invalid_suffix_restores_local_branch(
    tmp_path, monkeypatch, caplog
):
    from doin_core.protocol.messages import ChainStatus
    from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig

    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path), storage_backend="sqlite",
        discovery_enabled=False,
        domains=[DomainRole(domain_id="integrity-test", higher_is_better=False)],
    ))
    node.chaindb.open()
    try:
        genesis = node.chaindb.initialize()

        def branch(peer, increment):
            blocks = []
            previous = genesis
            for index in (1, 2):
                tx = accepted_tx()
                tx.peer_id = peer
                tx.timestamp += timedelta(seconds=index)
                tx.payload["optimae_id"] = f"{peer}-{index}"
                tx.payload["verified_performance"] = -float(index)
                tx.payload["effective_increment"] = increment
                tx.id = tx.compute_id()
                previous = mint(previous, tx)
                previous.header.generator_id = peer
                previous.hash = previous.header.compute_hash()
                blocks.append(previous)
            return blocks

        local = branch("local-peer", 1.0)
        peer = branch("remote-peer", 1.1)
        assert node.chaindb.append_blocks(local) == 2
        before = snapshot(node.chaindb)[0]
        peer[-1].transactions[0].payload["parameters"]["seed"] = 99
        endpoint = "test-peer"
        node.sync_manager.update_our_state(3, local[-1].hash)
        node.sync_manager.update_peer_status(endpoint, ChainStatus(
            chain_height=3, tip_hash=peer[-1].hash, tip_index=2,
        ))

        async def common_ancestor(*args):
            return 0

        async def fetch_branch(*args):
            return peer

        monkeypatch.setattr(node, "_find_common_ancestor", common_ancestor)
        monkeypatch.setattr("doin_node.unified.fetch_blocks", fetch_branch)
        assert not await node._resolve_equal_height_fork(object(), endpoint)
        assert "Invalid accepted transaction" in caplog.text
        assert snapshot(node.chaindb)[0] == before
        assert node.chaindb.get_blocks_range(1, 2) == local
    finally:
        node.chaindb.close()


@pytest.fixture
def pruning_db(tmp_path):
    db = ChainDB(tmp_path / "retained.db")
    db.open()
    try:
        genesis = db.initialize()
        tx = accepted_tx()
        tx.payload["optimae_id"] = "retained-optimum"
        tx.id = tx.compute_id()
        first = mint(genesis, tx)
        db.append_block(first)
        other = Transaction(
            id="custom-domain-event-id",
            tx_type=TransactionType.DOMAIN_UPDATED,
            domain_id=tx.domain_id,
            peer_id=tx.peer_id,
            payload={"name": "retained-domain"},
            timestamp=TS,
        )
        db.append_block(mint(first, other))
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("cutoff", [-1, 0, 1, 2, 10000])
@pytest.mark.parametrize("with_snapshot", [False, True])
def test_pruning_is_explicit_noop_preserving_full_blocks(
    pruning_db, cutoff, with_snapshot, caplog
):
    db = pruning_db
    if with_snapshot:
        db.save_snapshot(2, db.tip_hash, {}, {}, {})
    original = db.get_blocks_range(0, db.height - 1)
    before = snapshot(db)
    assert db.prune_transactions_before(cutoff) == 0
    assert db.prune_transactions_before(cutoff) == 0
    assert "Pruning disabled" in caplog.text
    assert snapshot(db) == before
    db.close()
    db.open()
    restored = db.get_blocks_range(0, db.height - 1)
    assert restored == original
    probe = Chain()
    assert probe.initialize() == restored[0]
    for block in restored[1:]:
        probe.append_block(block)
        assert block.header.merkle_root == compute_merkle_root(
            [tx.id for tx in block.transactions]
        )
        assert block.hash == block.header.compute_hash()
        assert all(db.has_transaction(tx.id) for tx in block.transactions)


def test_pruning_preserves_champion_preflight_and_replay_history(pruning_db, tmp_path):
    from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig

    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / "node"), storage_backend="json",
        discovery_enabled=False,
        domains=[DomainRole(domain_id="integrity-test", higher_is_better=False)],
    ))
    node.chaindb = pruning_db

    def successor(ident, score):
        tx = accepted_tx()
        tx.payload.update(optimae_id=ident, verified_performance=score)
        tx.id = tx.compute_id()
        return mint(pruning_db.get_tip(), tx)

    improvement = successor("new-optimum", -0.3)
    regression = successor("inferior-optimum", -0.1)
    replay = successor("retained-optimum", -0.4)
    expected = ({"integrity-test": -0.2}, {("integrity-test", "retained-optimum")})
    assert node._committed_champion_state() == expected
    assert node._preflight_champion_branch([improvement], 2)
    assert pruning_db.prune_transactions_before(10000) == 0
    pruning_db.close()
    pruning_db.open()
    assert node._committed_champion_state() == expected
    before = snapshot(pruning_db)
    assert node._preflight_champion_branch([improvement], 2)
    assert not node._preflight_champion_branch([regression], 2)
    assert not node._preflight_champion_branch([replay], 2)
    assert snapshot(pruning_db) == before


def test_pruning_noop_cannot_recover_historically_deleted_bodies(pruning_db):
    original = pruning_db.get_block(1)
    pruning_db._conn.execute("DELETE FROM transactions WHERE block_index = 1")
    before = snapshot(pruning_db)
    assert pruning_db.prune_transactions_before(10000) == 0
    assert snapshot(pruning_db) == before
    damaged = pruning_db.get_block(1)
    assert damaged.header == original.header
    assert damaged.hash == original.hash
    assert damaged.transactions == []
    probe = Chain()
    probe.initialize()
    with pytest.raises(ChainError, match="Merkle root mismatch"):
        probe.append_block(damaged)
