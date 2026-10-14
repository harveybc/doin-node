"""Executable requirements for network champion ordering, without live sockets.

These tests intentionally expose the deployed protocol gaps. They run the
real commit/reveal and block construction paths in temporary node directories.
"""

import json

import pytest

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType, compute_commitment
from doin_core.protocol.messages import Message, MessageType, OptimaeCommit, OptimaeReveal
from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig


DOMAIN = "audit-domain"


def make_node(tmp_path, name):
    node = UnifiedNode(UnifiedNodeConfig(
        port=0, data_dir=str(tmp_path / name), storage_backend="json",
        discovery_enabled=False, network_protocol="flooding",
        fee_market_enabled=False, initial_threshold=1e-8,
        domains=[DomainRole(domain_id=DOMAIN, higher_is_better=False,
                            synthetic_data_validation=False)],
    ))
    node.chain.initialize()
    return node


def messages(sender, performance=-0.2):
    params = {"seed": 42, "width": 8}
    nonce = "audit-fixed-nonce"
    commitment = compute_commitment(params, nonce)
    commit = Message(msg_type=MessageType.OPTIMAE_COMMIT, sender_id=sender,
                     payload=json.loads(OptimaeCommit(
                         commitment_hash=commitment, domain_id=DOMAIN).model_dump_json()))
    reveal = Message(msg_type=MessageType.OPTIMAE_REVEAL, sender_id=sender,
                     payload=json.loads(OptimaeReveal(
                         commitment_hash=commitment, domain_id=DOMAIN,
                         optimae_id="audit-optimum", parameters=params,
                         reported_performance=performance, nonce=nonce,
                     ).model_dump_json()))
    return commit, reveal


@pytest.mark.asyncio
async def test_receiver_does_not_mint_its_own_block_for_remote_optimum(tmp_path):
    origin = make_node(tmp_path, "origin")
    receiver = make_node(tmp_path, "receiver")
    commit, reveal = messages(origin.peer_id)
    await receiver._handle_optimae_commit(commit, "audit-peer")
    await receiver._handle_optimae_reveal(reveal, "audit-peer")
    assert receiver.chain.height == 1, (
        "Receiver minted a distinct block for somebody else's optimum; "
        "it must validate/adopt the originator's block."
    )


@pytest.mark.asyncio
async def test_same_reveal_does_not_produce_two_different_blocks(tmp_path):
    origin = make_node(tmp_path, "origin")
    receiver = make_node(tmp_path, "receiver")
    commit, reveal = messages(origin.peer_id)
    for node in (origin, receiver):
        await node._handle_optimae_commit(commit, "audit-peer")
        await node._handle_optimae_reveal(reveal, "audit-peer")
    if receiver.chain.height > 1:
        assert origin.chain.tip.hash == receiver.chain.tip.hash


@pytest.mark.asyncio
async def test_inferior_reveal_is_rejected_when_champion_is_known(tmp_path):
    node = make_node(tmp_path, "node")
    node._domain_best[DOMAIN] = ({"seed": 42, "width": 16}, -0.3)
    commit, reveal = messages("audit-peer", -0.2)
    await node._handle_optimae_commit(commit, "audit-peer")
    await node._handle_optimae_reveal(reveal, "audit-peer")
    assert node.chain.height == 1
    assert node._domain_best[DOMAIN][1] == -0.3
    assert node.consensus.weighted_sum == 0


@pytest.mark.parametrize("storage", ["json", "sqlite"])
def test_imported_block_payload_mutation_is_rejected(tmp_path, storage):
    node = make_node(tmp_path, "node")
    if storage == "sqlite":
        from doin_node.storage.chaindb import ChainDB
        node.chaindb = ChainDB(tmp_path / "audit.sqlite")
        node.chaindb.open()
        node.chaindb.initialize()
    tx = Transaction(tx_type=TransactionType.OPTIMAE_ACCEPTED,
                     domain_id=DOMAIN, peer_id="audit-peer",
                     payload={"verified_performance": -0.2,
                              "parameters": {"seed": 42}})
    block = Block(header=BlockHeader(
        index=1, previous_hash=node.chain.tip.hash,
        merkle_root=compute_merkle_root([tx.id]), generator_id="audit-peer",
        weighted_performance_sum=0.1, threshold=0.01), transactions=[tx])
    tx.payload["verified_performance"] = -999.0
    try:
        assert node._validate_and_append_blocks([block]) == 0, (
            "Mutated transaction body retained its old ID and was accepted."
        )
    finally:
        if node.chaindb:
            node.chaindb.close()


def test_import_rejects_a_block_that_regresses_committed_champion(tmp_path):
    node = make_node(tmp_path, "node")

    def block_with_score(score):
        tx = Transaction(tx_type=TransactionType.OPTIMAE_ACCEPTED,
                         domain_id=DOMAIN, peer_id="audit-peer",
                         payload={"optimae_id": f"opt-{score}",
                                  "verified_performance": score,
                                  "parameters": {"seed": 42}})
        return Block(header=BlockHeader(
            index=node.chain.height, previous_hash=node.chain.tip.hash,
            merkle_root=compute_merkle_root([tx.id]), generator_id="audit-peer",
            weighted_performance_sum=0.1, threshold=0.01), transactions=[tx])

    assert node._validate_and_append_blocks([block_with_score(-0.3)]) == 1
    node._update_domain_best_from_chain()
    assert node._domain_best[DOMAIN][1] == -0.3
    assert node._validate_and_append_blocks([block_with_score(-0.2)]) == 0
