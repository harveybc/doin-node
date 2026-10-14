"""Independent node wiring checks using synthetic blocks and archive bytes only."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doin_core.archive.body import ArchiveRefusal
from doin_core.consensus.fork_choice import ForkChoiceRule
from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader, Transaction, TransactionType
from doin_node.unified import UnifiedNode
from tests.test_archive_champion_projection import DOMAIN, archived
from tests.test_archive_champion_projection import candidate as candidate


TS = datetime(2026, 10, 13, tzinfo=timezone.utc)
PEER = "synthetic-peer"


def accepted(payload, domain=DOMAIN):
    return Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id=domain, peer_id=PEER, timestamp=TS,
        payload=deepcopy(payload),
    )


def block_after(previous, transactions):
    return Block(
        header=BlockHeader(
            index=previous.header.index + 1,
            previous_hash=previous.hash,
            timestamp=TS + timedelta(seconds=previous.header.index + 1),
            merkle_root=compute_merkle_root([tx.id for tx in transactions]),
            generator_id=PEER, weighted_performance_sum=0.05, threshold=0.01,
        ),
        transactions=transactions,
    )


def node_for(blocks, archive=None, *, higher_is_better=False):
    # Bypass startup entirely: exercise real methods over an in-memory chain view.
    node = UnifiedNode.__new__(UnifiedNode)
    node.config = SimpleNamespace(candidate_archive_mode="external_payload", acceptance_tolerance=0.0)
    node._domain_roles = {DOMAIN: SimpleNamespace(higher_is_better=higher_is_better)}
    node._domain_best = {}
    node._domain_champion_metrics = {}
    node._candidate_archive_references = {}
    node._candidate_external_archive = archive
    node._optimizer_plugins = {DOMAIN: Mock(spec=["set_network_champion"])}
    node._get_height = lambda: len(blocks)
    node._get_block = lambda index: blocks[index]
    return node


def pair(payload, *, score=-0.125, increment=0.05, reverse=False):
    transactions = [
        accepted({"optimae_id": payload["optimae_id"], "performance": score, "increment": increment}),
        accepted(payload),
    ]
    return transactions[::-1] if reverse else transactions


def legacy(identity, score, *, domain=DOMAIN):
    return accepted({
        "optimae_id": identity, "verified_performance": score,
        "effective_increment": 0.05, "parameters": {"width": 8, "source": identity},
        "champion_metrics": {"val_mae": 0.8, "train_mae": 0.9, "_model_b64": "not-public"},
    }, domain)


def assert_original_block(block, original):
    assert block.model_dump_json() == original
    assert block.hash == block.header.compute_hash()
    assert block.header.merkle_root == compute_merkle_root([tx.id for tx in block.transactions])
    assert all(tx.id == tx.compute_id() for tx in block.transactions)


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_full_legacy_recovery_uses_domain_direction(higher_is_better):
    blocks = [Block.genesis()]
    blocks.append(block_after(blocks[-1], [legacy("first", -0.1)]))
    blocks.append(block_after(blocks[-1], [legacy("second", -0.3)]))
    originals = [block.model_dump_json() for block in blocks]
    node = node_for(blocks, higher_is_better=higher_is_better)

    node._update_domain_best_from_chain()

    identity, score = ("first", -0.1) if higher_is_better else ("second", -0.3)
    parameters = {"width": 8, "source": identity}
    assert node._domain_best == {DOMAIN: (parameters, score)}
    assert node._domain_champion_metrics == {
        DOMAIN: {"performance": score, "val_mae": 0.8, "train_mae": 0.9},
    }
    node._optimizer_plugins[DOMAIN].set_network_champion.assert_called_once_with(parameters)
    for block, original in zip(blocks, originals):
        assert_original_block(block, original)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("retained_score", [False, True])
def test_hash_only_pair_recovery_and_empty_cache_restart(candidate, reverse, retained_score):
    archive, store, payload = archived(candidate)
    if retained_score:
        payload.update(verified_performance=-0.125, effective_increment=0.05)
    blocks = [Block.genesis()]
    blocks.append(block_after(blocks[-1], pair(payload, reverse=reverse)))
    original = blocks[-1].model_dump_json()
    objects, writes = dict(store.objects), store.writes
    node = node_for(blocks, archive)
    node._update_domain_best_from_chain()

    parameters, score = node._domain_best[DOMAIN]
    assert score == -0.125 != candidate["fitness"]
    assert parameters["core"] == candidate["effective_config"]["core"]
    assert parameters["_i6n_archive_reference"] == payload["candidate_archive_reference"]
    assert parameters["_i6n_genotype_sha256"] == candidate["genotype_canonical_sha256"]
    metrics = node._domain_champion_metrics[DOMAIN]
    assert metrics["performance"] == -0.125
    assert metrics["val_mae"] == metrics["train_mae"] == 0.8
    assert metrics["relative_error_vs_naive"] == pytest.approx(candidate["fitness"])
    node._optimizer_plugins[DOMAIN].set_network_champion.assert_called_once_with(parameters)

    restarted_blocks = [Block.model_validate_json(block.model_dump_json()) for block in blocks]
    restarted = node_for(restarted_blocks, archive)
    assert restarted._domain_best == restarted._domain_champion_metrics == {}
    assert restarted._candidate_archive_references == {}
    restarted._update_domain_best_from_chain()
    assert restarted._domain_best == node._domain_best
    assert restarted._domain_champion_metrics == node._domain_champion_metrics
    assert store.objects == objects and store.writes == writes
    assert_original_block(blocks[-1], original)


@pytest.mark.parametrize("empty_fork", [False, True])
def test_fork_recovery_replaces_stale_better_cache_and_removes_lost_domain(empty_fork):
    blocks = [Block.genesis()]
    blocks.append(block_after(blocks[0], [
        legacy("old-best", -0.9), legacy("lost", 10.0, domain="lost-domain"),
    ]))
    node = node_for(blocks)
    node._update_domain_best_from_chain()
    assert node._domain_best[DOMAIN][1] == -0.9
    assert "lost-domain" in node._domain_best
    node._optimizer_plugins[DOMAIN].set_network_champion.reset_mock()
    blocks[1:] = [] if empty_fork else [block_after(blocks[0], [legacy("new-fork", -0.1)])]

    node._update_domain_best_from_chain()

    assert "lost-domain" not in node._domain_best
    assert "lost-domain" not in node._domain_champion_metrics
    if empty_fork:
        assert node._domain_best == node._domain_champion_metrics == {}
        node._optimizer_plugins[DOMAIN].set_network_champion.assert_not_called()
    else:
        assert node._domain_best == {DOMAIN: ({"width": 8, "source": "new-fork"}, -0.1)}
        assert node._domain_champion_metrics[DOMAIN]["performance"] == -0.1
        node._optimizer_plugins[DOMAIN].set_network_champion.assert_called_once_with(node._domain_best[DOMAIN][0])


@pytest.mark.parametrize("missing", ["body_digest", "manifest_digest"])
def test_missing_bytes_leave_all_caches_and_plugins_untouched(candidate, missing):
    archive, store, payload = archived(candidate)
    blocks = [Block.genesis()]
    blocks.append(block_after(blocks[-1], [legacy("earlier-valid", 0.1)]))
    blocks.append(block_after(blocks[-1], pair(payload)))
    node = node_for(blocks, archive)
    node._domain_best = {DOMAIN: ({"source": "before-call"}, 0.9)}
    node._domain_champion_metrics = {DOMAIN: {"performance": 0.9, "val_mae": 99}}
    node._candidate_archive_references = {DOMAIN: {"existing": {"untouched": True}}}
    caches = (node._domain_best, node._domain_champion_metrics, node._candidate_archive_references)
    before = deepcopy(caches)
    originals = [block.model_dump_json() for block in blocks]
    del store.objects[payload["candidate_archive_reference"][missing]]

    with pytest.raises(ArchiveRefusal, match="MISSING"):
        node._update_domain_best_from_chain()

    assert node._domain_best is caches[0]
    assert node._domain_champion_metrics is caches[1]
    assert node._candidate_archive_references is caches[2]
    assert caches == before
    node._optimizer_plugins[DOMAIN].set_network_champion.assert_not_called()
    for block, original in zip(blocks, originals):
        assert_original_block(block, original)


def test_new_writer_retains_only_reference_identity_and_consensus_fields(candidate):
    archive, _, payload = archived(candidate)
    node = node_for([], archive)
    parameters = {**candidate["effective_config"], "_i6n_archive_reference": payload["candidate_archive_reference"]}
    small = {
        "verified_performance": -0.125, "effective_increment": 0.05,
        "reward_fraction": 0.75, "quorum_agree_fraction": 0.8, "incentive_reason": "verified",
    }
    acceptance = {
        **small, "parameters": parameters, "champion_metrics": {"val_mae": 0.8},
        "val_mae": 0.8, "genotype": candidate["genotype"], "optimae_id": "must-not-override",
    }
    before = deepcopy((parameters, acceptance))

    result = node._archive_only_optimae_payload(DOMAIN, payload["optimae_id"], parameters, acceptance)

    assert result == {**payload, **small}
    assert (parameters, acceptance) == before
    genesis = Block.genesis()
    block = block_after(genesis, pair(result))
    original = block.model_dump_json()
    projection = node._fork_choice_block(block)
    score = ForkChoiceRule().score_chain(block.hash, block.header.index, [projection])
    assert score.cumulative_increment == pytest.approx(0.05)
    assert projection["transactions"][1]["payload"]["effective_increment"] == 0.05
    assert_original_block(block, original)


@pytest.mark.parametrize("reverse", [False, True])
def test_old_pair_restores_work_only_in_fork_choice_projection(candidate, reverse):
    _, store, payload = archived(candidate)
    store.objects.clear()  # Work projection must not need archive availability.
    block = block_after(Block.genesis(), pair(payload, increment=0.075, reverse=reverse))
    original = block.model_dump_json()
    assert all("effective_increment" not in tx.payload for tx in block.transactions)

    projection = UnifiedNode._fork_choice_block(block)

    assert projection["height"] == block.header.index
    assert projection["hash"] == block.hash
    reference_tx = next(tx for tx in projection["transactions"] if "candidate_archive_reference" in tx["payload"])
    work_tx = next(tx for tx in projection["transactions"] if "performance" in tx["payload"])
    assert reference_tx["payload"]["effective_increment"] == 0.075
    assert "effective_increment" not in work_tx["payload"]
    score = ForkChoiceRule().score_chain(block.hash, block.header.index, [projection])
    assert score.cumulative_increment == pytest.approx(0.075)
    assert UnifiedNode._fork_choice_block(block) == projection
    reference_tx["payload"]["candidate_archive_reference"]["domain_id"] = "changed-copy"
    assert_original_block(block, original)


@pytest.mark.parametrize("conflict", ["performance", "increment", "peer"])
def test_conflicting_companion_cannot_recover_or_score(candidate, conflict):
    archive, _, payload = archived(candidate)
    payload.update(verified_performance=-0.125, effective_increment=0.05)
    transactions = pair(payload)
    if conflict == "peer":
        transactions[0].peer_id = "different-peer"
    else:
        transactions[0].payload[conflict] = 0.75
    transactions[0].id = transactions[0].compute_id()
    blocks = [Block.genesis()]
    blocks.append(block_after(blocks[0], transactions))
    node = node_for(blocks, archive)
    original = blocks[-1].model_dump_json()

    with pytest.raises(ValueError, match="CONFLICT"):
        node._update_domain_best_from_chain()
    with pytest.raises(ValueError, match="CONFLICT"):
        node._fork_choice_block(blocks[-1])

    assert node._domain_best == node._domain_champion_metrics == {}
    node._optimizer_plugins[DOMAIN].set_network_champion.assert_not_called()
    assert_original_block(blocks[-1], original)
