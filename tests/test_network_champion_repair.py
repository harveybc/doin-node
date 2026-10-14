"""Synthetic multi-node contract checks; no sockets, datasets or model fits."""

import json

import pytest

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType, compute_commitment
from doin_core.protocol.messages import ChainStatus, Message, MessageType, OptimaeCommit, OptimaeReveal
from doin_node.storage.chaindb import ChainDB
from tests.test_network_champion_contract_audit import DOMAIN, make_node


async def reveal_to(nodes, sender, score, ident):
    parameters = {"seed": 42, "width": 8}
    nonce = "nonce-" + ident
    commitment = compute_commitment(parameters, nonce)
    commit = Message(msg_type=MessageType.OPTIMAE_COMMIT, sender_id=sender,
                     payload=json.loads(OptimaeCommit(commitment_hash=commitment,
                                     domain_id=DOMAIN).model_dump_json()))
    reveal = Message(msg_type=MessageType.OPTIMAE_REVEAL, sender_id=sender,
                     payload=json.loads(OptimaeReveal(commitment_hash=commitment,
                                     domain_id=DOMAIN, optimae_id=ident,
                                     parameters=parameters, nonce=nonce,
                                     reported_performance=score).model_dump_json()))
    for node in nodes:
        await node._handle_optimae_commit(commit, "fixture-peer")
        await node._handle_optimae_reveal(reveal, "fixture-peer")


def copy_blocks(origin, receiver):
    blocks = [Block.model_validate_json(b.model_dump_json())
              for b in origin._get_blocks_range(receiver._get_height(), origin._get_height() - 1)]
    assert receiver._validate_and_append_blocks(blocks) == len(blocks)
    receiver._update_domain_best_from_chain()


@pytest.mark.asyncio
async def test_four_nodes_adopt_identical_originator_blocks_and_recover(tmp_path):
    nodes = [make_node(tmp_path, f"node-{i}") for i in range(4)]
    # Deliver the announcement to peers first, then the originator.
    await reveal_to(nodes[1:] + nodes[:1], nodes[0].peer_id, -0.2, "first")
    assert [n._get_height() for n in nodes] == [2, 1, 1, 1]
    for node in nodes[1:]:
        assert node.consensus.weighted_sum == 0
        copy_blocks(nodes[0], node)
        assert await node.try_generate_block() is None
    await reveal_to(nodes, nodes[2].peer_id, -0.3, "second")
    assert [n._get_height() for n in nodes] == [2, 2, 3, 2]
    for i in (0, 1, 3):
        copy_blocks(nodes[2], nodes[i])
    expected = [b.model_dump_json() for b in nodes[0].chain.blocks]
    for node in nodes:
        assert [b.model_dump_json() for b in node.chain.blocks] == expected
        node._domain_best.clear()
        node._update_domain_best_from_chain()
        assert node._domain_best[DOMAIN] == ({"seed": 42, "width": 8}, -0.3)
        assert node.consensus.weighted_sum == 0
    await reveal_to(nodes, nodes[1].peer_id, -0.25, "stale")
    assert all(n._get_height() == 3 for n in nodes)


@pytest.mark.asyncio
async def test_adopted_superior_block_removes_pending_work(tmp_path):
    local, other = make_node(tmp_path, "local"), make_node(tmp_path, "other")
    local.consensus.state.threshold = 100
    await reveal_to([local], local.peer_id, -0.2, "local")
    assert local.consensus.weighted_sum > 0 and local._get_height() == 1
    await reveal_to([other], other.peer_id, -0.3, "remote")
    copy_blocks(other, local)
    local.consensus.state.threshold = 1e-8
    assert await local.try_generate_block() is None
    assert local.consensus.weighted_sum == 0
    assert all(t.tx_type != TransactionType.OPTIMAE_ACCEPTED
               for t in local.consensus.state.pending_transactions)


@pytest.mark.asyncio
async def test_pending_optimum_is_not_network_champion_or_migration(tmp_path):
    node = make_node(tmp_path, "pending-only")
    node.consensus.state.threshold = 100
    migrations = []
    node._optimizer_plugins[DOMAIN] = type("Observer", (), {
        "set_network_champion": lambda self, params: migrations.append(params),
    })()
    await reveal_to([node], node.peer_id, -0.2, "pending-only")
    assert node._get_height() == 1
    assert DOMAIN not in node._domain_best
    assert migrations == []


@pytest.mark.asyncio
async def test_stage_message_does_not_repeat_champion_migration(tmp_path):
    node = make_node(tmp_path, "stage-no-migration")
    migrations = []
    node._optimizer_plugins[DOMAIN] = type("Observer", (), {
        "set_network_champion": lambda self, params: migrations.append(params),
    })()
    await reveal_to([node], node.peer_id, -0.2, "committed")
    assert node._get_height() == 2
    assert len(migrations) == 1
    msg = Message(msg_type=MessageType.STAGE_COMPLETE, sender_id="peer",
                  payload={"domain_id": DOMAIN, "stage": 1, "total_stages": 2,
                           "champion_fitness": -0.2, "champion_params": {"seed": 42}})
    await node._handle_stage_complete(msg, "fixture")
    assert len(migrations) == 1
    node._update_domain_best_from_chain()
    assert len(migrations) == 1


@pytest.mark.asyncio
async def test_pending_improvement_rebases_work_against_new_parent(tmp_path):
    local, other = make_node(tmp_path, "local"), make_node(tmp_path, "other")
    local.consensus.state.threshold = 100
    await reveal_to([local], local.peer_id, -0.2, "pending")
    initial_work = local.consensus.weighted_sum
    await reveal_to([other], other.peer_id, -0.1, "first")
    copy_blocks(other, local)
    local.consensus.state.threshold = 1e-8
    block = await local.try_generate_block()
    assert block is not None
    assert block.header.weighted_performance_sum == pytest.approx(initial_work / 2)
    assert local._domain_best[DOMAIN][1] == -0.2
    assert all(tx.id == tx.compute_id() for tx in block.transactions)


@pytest.mark.asyncio
async def test_stage_hint_does_not_overwrite_champion(tmp_path):
    node = make_node(tmp_path, "node")
    msg = Message(msg_type=MessageType.STAGE_COMPLETE, sender_id="peer",
                  payload={"domain_id": DOMAIN, "stage": 1, "total_stages": 2,
                           "champion_fitness": -100, "champion_params": {"seed": 42}})
    await node._handle_stage_complete(msg, "fixture")
    assert DOMAIN not in node._domain_best


def branch(parent, score, work, ident, peer="origin"):
    tx = Transaction(tx_type=TransactionType.OPTIMAE_ACCEPTED,
                     domain_id=DOMAIN, peer_id=peer,
                     payload={"optimae_id": ident, "verified_performance": score,
                              "effective_increment": work, "parameters": {"seed": 42}})
    return Block(header=BlockHeader(index=parent.header.index + 1, previous_hash=parent.hash,
                 generator_id=peer, merkle_root=compute_merkle_root([tx.id]),
                 weighted_performance_sum=work, threshold=1e-8), transactions=[tx])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_peer_fork_validated_before_rollback(tmp_path, monkeypatch, invalid):
    node = make_node(tmp_path, "node")
    node.chaindb = ChainDB(tmp_path / "chain.sqlite")
    node.chaindb.open()
    genesis = node.chaindb.initialize()
    local = branch(genesis, -0.2, 0.2, "local", node.peer_id)
    node.chaindb.append_block(local)
    peer = branch(genesis, -0.3, 0.3, "peer")
    if invalid:
        peer.transactions[0].payload["verified_performance"] = -999
    node.sync_manager.update_our_state(2, local.hash)
    node.sync_manager.update_peer_status("fixture:1", ChainStatus(
        chain_height=2, tip_hash=peer.hash, tip_index=1))

    async def fetch(_session, _endpoint, first, last):
        return [peer] if first == last == 1 else [genesis]

    monkeypatch.setattr("doin_node.unified.fetch_blocks", fetch)
    try:
        changed = await node._resolve_equal_height_fork(object(), "fixture:1")
        assert changed is not invalid
        assert node._get_tip().hash == (local.hash if invalid else peer.hash)
    finally:
        node.chaindb.close()


@pytest.mark.asyncio
async def test_uncommitted_champion_response_cannot_replace_chain_state(tmp_path):
    node = make_node(tmp_path, "node")
    message = Message(msg_type=MessageType.CHAMPION_RESPONSE, sender_id="uncommitted-peer",
                      payload={"domain_id": DOMAIN, "has_champion": True,
                               "performance": -100, "parameters": {"seed": 42}})
    await node._handle_champion_response(message, "fixture-peer")
    assert DOMAIN not in node._domain_best


@pytest.mark.asyncio
async def test_shared_population_checkpoint_is_not_optimization_work(tmp_path):
    node = make_node(tmp_path, "node")
    await node._store_shared_population_in_chain(DOMAIN, {"generation": 0, "population": []})
    assert node._get_height() == 1
    assert node.consensus.weighted_sum == 0
    assert all(tx.tx_type != TransactionType.OPTIMAE_ACCEPTED
               for tx in node.consensus.state.pending_transactions)
    await reveal_to([node], node.peer_id, -0.2, "genuine-improvement")
    assert node._get_height() == 2
    assert node._recover_shared_state_from_chain(DOMAIN) == {"generation": 0, "population": []}
