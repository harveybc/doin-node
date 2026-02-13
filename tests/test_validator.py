"""Tests for optimae validation."""

from doin_core.models.domain import Domain, DomainConfig
from doin_core.models.optimae import Optimae
from doin_node.validation.validator import OptimaeValidator


def _make_domain(domain_id: str = "d1") -> Domain:
    return Domain(
        id=domain_id,
        name="Test Domain",
        performance_metric="accuracy",
        higher_is_better=True,
        weight=1.0,
        config=DomainConfig(
            optimization_plugin="test_opt",
            inference_plugin="test_inf",
        ),
    )


def _make_optimae(
    domain_id: str = "d1",
    performance: float = 0.9,
) -> Optimae:
    return Optimae(
        domain_id=domain_id,
        optimizer_id="optimizer-1",
        parameters={"w": [1, 2, 3]},
        reported_performance=performance,
    )


class TestOptimaeValidator:
    def test_submit_valid_domain(self) -> None:
        validator = OptimaeValidator()
        validator.register_domain(_make_domain())
        optimae = _make_optimae()
        oid = validator.submit_for_validation(optimae)
        assert oid == optimae.id
        assert validator.pending_count == 1

    def test_submit_unknown_domain_raises(self) -> None:
        validator = OptimaeValidator()
        optimae = _make_optimae("nonexistent")
        try:
            validator.submit_for_validation(optimae)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_accept_within_tolerance(self) -> None:
        validator = OptimaeValidator(tolerance=0.05)
        validator.register_domain(_make_domain())
        optimae = _make_optimae(performance=0.90)
        validator.submit_for_validation(optimae)

        result = validator.record_evaluation(optimae.id, 0.89)
        assert result.is_valid
        assert result.verified_performance == 0.89
        assert validator.pending_count == 0

    def test_reject_outside_tolerance(self) -> None:
        validator = OptimaeValidator(tolerance=0.05)
        validator.register_domain(_make_domain())
        optimae = _make_optimae(performance=0.90)
        validator.submit_for_validation(optimae)

        result = validator.record_evaluation(optimae.id, 0.70)
        assert not result.is_valid
        assert validator.pending_count == 0

    def test_performance_increment_calculated(self) -> None:
        validator = OptimaeValidator(tolerance=0.05)
        domain = _make_domain()
        domain.current_best_performance = 0.80
        validator.register_domain(domain)

        optimae = _make_optimae(performance=0.90)
        validator.submit_for_validation(optimae)

        result = validator.record_evaluation(optimae.id, 0.89)
        assert result.is_valid
        # Increment: 0.89 - 0.80 = 0.09
        assert abs(optimae.performance_increment - 0.09) < 1e-6
