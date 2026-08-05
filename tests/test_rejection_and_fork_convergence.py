"""Adversarial tests for AUD-F1-20260805-109 and -111.

DOIN must reject failed candidate results independently of plugin
convenience flags, refuse rejected champions, and resolve equal-height
forks deterministically without crashing on empty divergent ranges or
racing peers.
"""
from __future__ import annotations

import pytest

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block, BlockHeader
from doin_core.protocol.messages import ChainStatus
from doin_core.models.transaction import Transaction, TransactionType
from doin_core.consensus.fork_choice import ForkChoiceRule
from doin_node.unified import (
    UnifiedNode,
    UnifiedNodeConfig,
    _candidate_rejection_reason,
)


class TestIndependentRejection:
    def test_explicit_flag_rejected(self):
        assert _candidate_rejection_reason(
            {"candidate_rejected": True, "fitness": 1.0}
        ) == "candidate_rejected"

    def test_eval_error_rejected_without_flag(self):
        """The audited failure shape: only fitness + evaluation evidence,
        no convenience flag — must still be rejected."""
        for key in ("_eval_error", "evaluation_error", "simulator_error"):
            reason = _candidate_rejection_reason(
                {"fitness": -1e9, key: "boom"})
            assert reason is not None and "boom" in reason

    def test_worst_sentinel_never_eligible(self):
        assert _candidate_rejection_reason(
            {"fitness": -1.0e9}) == "worst_sentinel_fitness"
        assert _candidate_rejection_reason(
            {"fitness": -5.0e9}) == "worst_sentinel_fitness"

    def test_non_finite_and_non_numeric_rejected(self):
        assert "fitness_not_finite" in _candidate_rejection_reason(
            {"fitness": float("nan")})
        assert "fitness_not_finite" in _candidate_rejection_reason(
            {"fitness": float("inf")})
        assert "fitness_not_numeric" in _candidate_rejection_reason(
            {"fitness": None})
        assert "fitness_not_numeric" in _candidate_rejection_reason({})

    def test_clean_result_eligible(self):
        assert _candidate_rejection_reason(
            {"fitness": 12345.0, "hyper_dict": {"a": 1}}) is None

    @pytest.mark.asyncio
    async def test_rejected_champion_broadcast_refused(self, tmp_path):
        """A failed first candidate must not become the initial champion
        via broadcast, whatever its numeric fitness."""
        config = UnifiedNodeConfig(
            port=8499, data_dir=str(tmp_path), discovery_enabled=False)
        node = UnifiedNode(config)
        calls: list = []

        async def record(*args, **kwargs):
            calls.append(args)

        node._handle_optimae_reveal = record
        node._broadcast = record
        await node._broadcast_champion(
            "test-domain", {"p": 1}, -1e9,
            {"fitness": -1e9, "_eval_error": "crashed"}, 0, {})
        assert calls == [], "rejected candidate reached broadcast"


class TestDeterministicTieBreak:
    def test_equal_weight_tie_breaks_by_lowest_hash_any_order(self):
        """Convergence must not depend on local insertion order."""
        chains = {
            "cc": [{"height": 1, "hash": "x", "transactions": [
                {"tx_type": "optimae_accepted",
                 "payload": {"effective_increment": 1.0}}]}],
            "aa": [{"height": 1, "hash": "y", "transactions": [
                {"tx_type": "optimae_accepted",
                 "payload": {"effective_increment": 1.0}}]}],
            "bb": [{"height": 1, "hash": "z", "transactions": [
                {"tx_type": "optimae_accepted",
                 "payload": {"effective_increment": 1.0}}]}],
        }
        for order in (["cc", "aa", "bb"], ["bb", "cc", "aa"],
                      ["aa", "bb", "cc"]):
            rule = ForkChoiceRule()
            for tip in order:
                rule.score_chain(tip, 10, chains[tip])
            assert rule.select_best().tip_hash == "aa", order


def _branch_block(genesis_hash: str, generator: str,
                  increment: float) -> Block:
    transaction = Transaction(
        tx_type=TransactionType.OPTIMAE_ACCEPTED,
        domain_id="test-domain",
        peer_id=generator,
        payload={"effective_increment": increment},
    )
    header = BlockHeader(
        index=1,
        previous_hash=genesis_hash,
        merkle_root=compute_merkle_root([transaction.id]),
        generator_id=generator,
        weighted_performance_sum=increment,
        threshold=0.0,
    )
    return Block(header=header, transactions=[transaction])


class TestEqualHeightForkRepair:
    @pytest.mark.asyncio
    async def test_empty_divergent_range_returns_false(
        self, tmp_path, monkeypatch
    ):
        """AUD-F1-20260805-111: common ancestor == tip index (stale peer
        state) must retry later, not raise IndexError."""
        config = UnifiedNodeConfig(
            port=8501, data_dir=str(tmp_path), discovery_enabled=False)
        node = UnifiedNode(config)
        node.chaindb.open()
        genesis = node.chaindb.initialize("genesis")
        local = _branch_block(genesis.hash, "local", 1.0)
        node.chaindb.append_block(local)
        node.sync_manager.update_our_state(2, local.hash)
        node.sync_manager.update_peer_status(
            "peer:9", ChainStatus(chain_height=2, tip_hash="different",
                                  tip_index=1))

        async def stale_common(_session, _endpoint, our_height):
            return our_height - 1  # ancestor IS the tip: empty range

        monkeypatch.setattr(node, "_find_common_ancestor", stale_common)
        fetches: list = []

        async def must_not_fetch(*args, **kwargs):
            fetches.append(args)
            return []

        monkeypatch.setattr(
            "doin_node.unified.fetch_blocks", must_not_fetch)
        try:
            assert await node._resolve_equal_height_fork(
                object(), "peer:9") is False
            assert fetches == []
            assert node.chaindb.tip_hash == local.hash
        finally:
            node.chaindb.close()

    @pytest.mark.asyncio
    async def test_peer_tip_change_during_fetch_bounded_retry(
        self, tmp_path, monkeypatch
    ):
        """A peer that rolls back mid-fetch triggers exactly one refetch
        and then a clean deferral; the local chain is untouched."""
        config = UnifiedNodeConfig(
            port=8502, data_dir=str(tmp_path), discovery_enabled=False)
        node = UnifiedNode(config)
        node.chaindb.open()
        genesis = node.chaindb.initialize("genesis")
        local = _branch_block(genesis.hash, "local", 1.0)
        stale_peer = _branch_block(genesis.hash, "peer-old", 9.0)
        current_peer = _branch_block(genesis.hash, "peer-new", 2.0)
        node.chaindb.append_block(local)
        node.sync_manager.update_our_state(2, local.hash)
        node.sync_manager.update_peer_status(
            "peer:9", ChainStatus(chain_height=2,
                                  tip_hash=stale_peer.hash, tip_index=1))
        attempts: list = []

        async def racing_fetch(_session, _endpoint, from_i, to_i):
            attempts.append((from_i, to_i))
            return [current_peer]  # never matches the recorded tip

        monkeypatch.setattr("doin_node.unified.fetch_blocks", racing_fetch)
        try:
            assert await node._resolve_equal_height_fork(
                object(), "peer:9") is False
            # One ancestor-search probe + exactly two bounded range
            # fetches; an unbounded loop would spin far past this.
            assert len(attempts) == 3, attempts
            assert node.chaindb.tip_hash == local.hash
            assert node.chaindb.height == 2
        finally:
            node.chaindb.close()
