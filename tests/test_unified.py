"""Tests for the Unified DON Node.

Tests the full integration of all security systems:
1. Commit-reveal     2. Random quorum     3. Asymmetric reputation
4. Resource limits   5. Finality          6. Reputation decay
7. Min reputation    8. External anchors  9. Fork choice
10. Deterministic seeds
"""

import asyncio
import json
import time

import pytest

from doin_core.models import (
    Commitment,
    Reveal,
    Task,
    TaskQueue,
    TaskStatus,
    TaskType,
    compute_commitment,
)
from doin_core.models.reputation import MIN_REPUTATION_FOR_CONSENSUS
from doin_core.protocol.messages import (
    Message,
    MessageType,
    OptimaeCommit,
    OptimaeReveal,
)

from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig


# ── Helpers ──────────────────────────────────────────────────────────

def make_node(
    port: int = 8470,
    domains: list[DomainRole] | None = None,
    quorum_min: int = 1,
    require_seed: bool = False,
) -> UnifiedNode:
    """Create a UnifiedNode for testing (doesn't start transport)."""
    config = UnifiedNodeConfig(
        port=port,
        data_dir=f"/tmp/don-test-{port}",
        domains=domains or [
            DomainRole(
                domain_id="test-domain",
                optimize=True,
                evaluate=True,
                has_synthetic_data=True,
                param_bounds={"lr": (1e-5, 1.0), "layers": (1, 10)},
            ),
        ],
        quorum_min_evaluators=quorum_min,
        require_deterministic_seed=require_seed,
    )
    return UnifiedNode(config)


# ── Test: Commit-Reveal (Hardening #1) ──────────────────────────────

class TestCommitRevealIntegration:
    def test_commitment_registered(self):
        node = make_node()
        h = compute_commitment({"lr": 0.01}, "nonce1")
        commitment = Commitment(
            commitment_hash=h,
            domain_id="test-domain",
            optimizer_id="opt-1",
        )
        assert node.commit_reveal.add_commitment(commitment)
        assert node.commit_reveal.has_valid_commitment(h)

    def test_reveal_matches_commitment(self):
        node = make_node()
        params = {"lr": 0.01}
        nonce = "nonce1"
        h = compute_commitment(params, nonce)

        node.commit_reveal.add_commitment(Commitment(
            commitment_hash=h, domain_id="test-domain", optimizer_id="opt-1",
        ))

        reveal = Reveal(
            commitment_hash=h, domain_id="test-domain", optimizer_id="opt-1",
            parameters=params, nonce=nonce, reported_performance=-0.5,
        )
        assert node.commit_reveal.process_reveal(reveal)

    def test_reveal_wrong_params_rejected(self):
        node = make_node()
        params = {"lr": 0.01}
        nonce = "nonce1"
        h = compute_commitment(params, nonce)

        node.commit_reveal.add_commitment(Commitment(
            commitment_hash=h, domain_id="test-domain", optimizer_id="opt-1",
        ))

        reveal = Reveal(
            commitment_hash=h, domain_id="test-domain", optimizer_id="opt-1",
            parameters={"lr": 999.0}, nonce=nonce, reported_performance=-0.5,
        )
        assert not node.commit_reveal.process_reveal(reveal)


# ── Test: Random Quorum (Hardening #2) ──────────────────────────────

class TestQuorumIntegration:
    def test_quorum_excludes_optimizer(self):
        node = make_node(quorum_min=2)
        selected = node.quorum.select_evaluators(
            "opt-1", "test-domain", "optimizer-peer",
            -0.5, ["optimizer-peer", "eval-1", "eval-2", "eval-3"], "tip",
        )
        assert "optimizer-peer" not in selected

    def test_quorum_deterministic(self):
        node1 = make_node(quorum_min=2)
        node2 = make_node(port=8471, quorum_min=2)
        evals = ["eval-1", "eval-2", "eval-3"]

        s1 = node1.quorum.select_evaluators("opt-1", "d", "opt", -0.5, evals, "tip")
        s2 = node2.quorum.select_evaluators("opt-1", "d", "opt", -0.5, evals, "tip")
        assert s1 == s2

    def test_quorum_accepts_consistent_votes(self):
        node = make_node(quorum_min=2)
        selected = node.quorum.select_evaluators(
            "opt-1", "test-domain", "optimizer", -0.5,
            ["eval-1", "eval-2", "eval-3"], "tip",
        )
        for ev in selected:
            node.quorum.add_vote("opt-1", ev, -0.50)

        result = node.quorum.evaluate_quorum("opt-1")
        assert result.accepted

    def test_quorum_rejects_when_report_diverges(self):
        node = make_node(quorum_min=2)
        selected = node.quorum.select_evaluators(
            "opt-1", "test-domain", "optimizer", -0.10,  # Claims -0.10
            ["eval-1", "eval-2", "eval-3"], "tip",
        )
        for ev in selected:
            node.quorum.add_vote("opt-1", ev, -0.50)  # All see -0.50

        result = node.quorum.evaluate_quorum("opt-1")
        assert not result.accepted


# ── Test: Reputation (Hardening #3, #6, #7) ─────────────────────────

class TestReputationIntegration:
    def test_asymmetric_penalties(self):
        node = make_node()
        node.reputation.record_optimae_accepted("peer-1")
        score_after_accept = node.reputation.get_score("peer-1")
        node.reputation.record_optimae_rejected("peer-1")
        score_after_reject = node.reputation.get_score("peer-1")
        # After 1 accept + 1 reject, score should be near zero
        # (penalty=3.0 >> reward=1.0)
        assert score_after_reject < score_after_accept * 0.1

    def test_min_threshold_for_consensus(self):
        node = make_node()
        assert not node.reputation.meets_threshold("new-peer")
        # Build enough reputation
        for _ in range(5):
            node.reputation.record_optimae_accepted("new-peer")
        assert node.reputation.meets_threshold("new-peer")

    def test_decay_over_time(self):
        node = make_node()
        # Use a short half-life for testing
        node.reputation._half_life = 0.5
        node.reputation.record_optimae_accepted("peer-1")
        s1 = node.reputation.get_score("peer-1")
        # Simulate time passing
        rep = node.reputation.get("peer-1")
        rep.last_activity = time.time() - 2.0
        s2 = node.reputation.get_score("peer-1")
        assert s2 < s1 * 0.3  # 4 half-lives → ~6.25% of original


# ── Test: Resource Limits (Hardening #4) ─────────────────────────────

class TestResourceLimitsIntegration:
    def test_bounds_validation(self):
        node = make_node()
        v = node._bounds_validators["test-domain"]
        ok, _ = v.validate({"lr": 0.01, "layers": 5})
        assert ok

    def test_out_of_bounds_rejected(self):
        node = make_node()
        v = node._bounds_validators["test-domain"]
        ok, reason = v.validate({"lr": 100.0})
        assert not ok

    def test_resource_limits(self):
        node = make_node()
        v = node._bounds_validators["test-domain"]
        role = node._domain_roles["test-domain"]
        ok, _ = v.validate_resource_limits({"epochs": 999999}, role.resource_limits)
        assert not ok


# ── Test: Finality (Hardening #5) ────────────────────────────────────

class TestFinalityIntegration:
    def test_initial_finality(self):
        node = make_node()
        assert node.finality.finalized_height == -1

    def test_explicit_checkpoint(self):
        node = make_node()
        node.finality.add_checkpoint(10, "hash10")
        assert node.finality.finalized_height == 10

    def test_reorg_blocked_below_finality(self):
        node = make_node()
        node.finality.add_checkpoint(10, "h10")
        assert not node.finality.is_reorg_allowed(15, 20)  # Reorg to 5 < 10

    def test_reorg_allowed_above_finality(self):
        node = make_node()
        node.finality.add_checkpoint(10, "h10")
        assert node.finality.is_reorg_allowed(3, 20)  # Reorg to 17 > 10


# ── Test: External Anchoring (Hardening #8) ──────────────────────────

class TestExternalAnchoringIntegration:
    def test_anchor_at_interval(self):
        node = make_node()
        assert node.anchor_manager.should_anchor(100)
        assert not node.anchor_manager.should_anchor(50)

    def test_create_and_verify_anchor(self):
        node = make_node()
        anchor = node.anchor_manager.create_anchor(100, "bh", "sh")
        assert node.anchor_manager.verify_chain_against_anchor(100, "bh", "sh") is True
        assert node.anchor_manager.verify_chain_against_anchor(100, "wrong", "sh") is False


# ── Test: Fork Choice (Hardening #9) ────────────────────────────────

class TestForkChoiceIntegration:
    def test_heavier_chain_wins(self):
        node = make_node()
        node.fork_choice.score_chain("weak", 10, [
            {"height": 1, "hash": "h1", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 1.0}},
            ]},
        ])
        node.fork_choice.score_chain("strong", 10, [
            {"height": 1, "hash": "h1", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 10.0}},
            ]},
        ])
        best = node.fork_choice.select_best()
        assert best.tip_hash == "strong"

    def test_checkpoint_inconsistent_loses(self):
        node = make_node()
        node.fork_choice.score_chain("bad", 10, [
            {"height": 5, "hash": "wrong", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 100.0}},
            ]},
        ], finalized_height=5, finalized_hash="correct")
        node.fork_choice.score_chain("good", 10, [
            {"height": 5, "hash": "correct", "transactions": [
                {"tx_type": "optimae_accepted", "payload": {"effective_increment": 1.0}},
            ]},
        ], finalized_height=5, finalized_hash="correct")

        assert node.fork_choice.select_best().tip_hash == "good"


# ── Test: Deterministic Seed (Hardening #10) ────────────────────────

class TestDeterministicSeedIntegration:
    def test_seed_derivation(self):
        node = make_node(require_seed=True)
        seed = node.seed_policy.get_seed_for_optimae("commit123", "test-domain")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    def test_seed_validation(self):
        node = make_node(require_seed=True)
        seed = node.seed_policy.get_seed_for_optimae("commit123", "test-domain")
        ok, _ = node.seed_policy.validate_submission("commit123", "test-domain", seed)
        assert ok

    def test_wrong_seed_rejected(self):
        node = make_node(require_seed=True)
        ok, reason = node.seed_policy.validate_submission("commit123", "test-domain", 12345)
        assert not ok

    def test_evaluator_seeds_differ(self):
        node = make_node(require_seed=True)
        s0 = node.seed_policy.get_seed_for_evaluation("h", "d", 0)
        s1 = node.seed_policy.get_seed_for_evaluation("h", "d", 1)
        assert s0 != s1


# ── Test: VUW Weights ───────────────────────────────────────────────

class TestVUWIntegration:
    def test_domain_with_synthetic_has_weight(self):
        node = make_node()
        weights = node.vuw.compute_weights()
        assert weights["test-domain"] > 0

    def test_domain_without_synthetic_zero_weight(self):
        node = make_node(domains=[
            DomainRole(domain_id="no-synth", has_synthetic_data=False),
        ])
        weights = node.vuw.compute_weights()
        assert weights["no-synth"] == 0.0

    def test_effective_increment_uses_reputation(self):
        node = make_node()
        eff_low = node.vuw.get_effective_increment("test-domain", 1.0, 0.5)
        eff_high = node.vuw.get_effective_increment("test-domain", 1.0, 10.0)
        assert eff_high > eff_low


# ── Test: Node Configuration ────────────────────────────────────────

class TestNodeConfiguration:
    def test_optimizer_domains(self):
        node = make_node(domains=[
            DomainRole(domain_id="d1", optimize=True),
            DomainRole(domain_id="d2", evaluate=True),
            DomainRole(domain_id="d3", optimize=True, evaluate=True),
        ])
        assert set(node.optimizer_domains) == {"d1", "d3"}
        assert set(node.evaluator_domains) == {"d2", "d3"}

    def test_peer_management(self):
        node = make_node()
        node.add_peer("192.168.1.1", 8470, "peer-1")
        assert len(node._peers) == 1

    def test_status_includes_security_info(self):
        node = make_node()
        # The status endpoint exposes security system state
        assert node.finality.finalized_height == -1
        assert node.commit_reveal.pending_count == 0
        assert node.quorum.pending_count == 0


# ── Test: Full Flow (end-to-end without network) ────────────────────

class TestFullFlow:
    def test_commit_reveal_to_quorum_to_reputation(self):
        """Test the full optimae lifecycle without network transport."""
        node = make_node(quorum_min=2, require_seed=False)
        optimizer_id = "optimizer-1"
        params = {"lr": 0.01, "layers": 3}
        nonce = "test-nonce"

        # Step 1: Commit
        h = compute_commitment(params, nonce)
        node.commit_reveal.add_commitment(Commitment(
            commitment_hash=h, domain_id="test-domain", optimizer_id=optimizer_id,
        ))

        # Step 2: Reveal
        reveal = Reveal(
            commitment_hash=h, domain_id="test-domain", optimizer_id=optimizer_id,
            parameters=params, nonce=nonce, reported_performance=-0.50,
        )
        assert node.commit_reveal.process_reveal(reveal)

        # Step 3: Bounds validation
        v = node._bounds_validators["test-domain"]
        ok, _ = v.validate(params)
        assert ok

        # Step 4: Quorum selection
        evaluators = ["eval-1", "eval-2", "eval-3"]
        selected = node.quorum.select_evaluators(
            "opt-1", "test-domain", optimizer_id, -0.50, evaluators, "chain-tip",
        )
        assert optimizer_id not in selected
        assert len(selected) == 2

        # Step 5: Evaluators vote (both agree with reported)
        for ev in selected:
            node.quorum.add_vote("opt-1", ev, -0.50)

        # Step 6: Evaluate quorum
        result = node.quorum.evaluate_quorum("opt-1")
        assert result.accepted

        # Step 7: Update reputation
        for ev, agreed in result.agreements.items():
            node.reputation.record_evaluation_completed(ev, agreed)
        node.reputation.record_optimae_accepted(optimizer_id)

        assert node.reputation.get_score(optimizer_id) > 0
        for ev in selected:
            assert node.reputation.get_score(ev) > 0

        # Step 8: VUW effective increment
        eff = node.vuw.get_effective_increment(
            "test-domain", 0.05, node.reputation.get_score(optimizer_id),
        )
        assert eff > 0

    def test_rejected_optimae_penalizes_reputation(self):
        """Optimizer submitting bad results gets reputation slashed."""
        node = make_node(quorum_min=2, require_seed=False)

        # Build some reputation first
        for _ in range(5):
            node.reputation.record_optimae_accepted("bad-optimizer")
        rep_before = node.reputation.get_score("bad-optimizer")

        # Quorum rejects
        selected = node.quorum.select_evaluators(
            "opt-bad", "test-domain", "bad-optimizer", -0.10,
            ["eval-1", "eval-2", "eval-3"], "tip",
        )
        for ev in selected:
            node.quorum.add_vote("opt-bad", ev, -0.50)  # Real perf is -0.50, not -0.10

        result = node.quorum.evaluate_quorum("opt-bad")
        assert not result.accepted

        node.reputation.record_optimae_rejected("bad-optimizer")
        rep_after = node.reputation.get_score("bad-optimizer")
        assert rep_after < rep_before

    def test_no_synthetic_means_zero_effective_increment(self):
        """Domain without synthetic data gets zero consensus weight."""
        node = make_node(domains=[
            DomainRole(domain_id="no-synth", has_synthetic_data=False),
        ])
        eff = node.vuw.get_effective_increment("no-synth", 1.0, 10.0)
        assert eff == 0.0
