"""OptimaeValidator — coordinates verification of reported optimization results.

When an optimizer submits an optimae, nodes must validate the claimed
performance. The validator coordinates with evaluators to verify results,
optionally using synthetic data to prevent overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from doin_core.models.domain import Domain
from doin_core.models.optimae import Optimae

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating an optimae."""

    optimae_id: str
    domain_id: str
    reported_performance: float
    verified_performance: float | None = None
    is_valid: bool = False
    used_synthetic_data: bool = False
    error: str | None = None


class OptimaeValidator:
    """Validates optimae by coordinating with evaluators.

    The validation flow:
    1. Receive optimae announcement from optimizer
    2. Forward parameters to one or more evaluators
    3. Evaluators run inference plugin with the parameters
    4. If synthetic data plugin exists, also validate on synthetic data
    5. Compare verified performance against reported performance
    6. Accept if within tolerance, reject otherwise
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        relative_tolerance: float | None = None,
        require_multiple_evaluators: bool = False,
        min_evaluators: int = 1,
    ) -> None:
        """
        Args:
            tolerance: Absolute acceptable difference between reported and verified.
            relative_tolerance: If set, tolerance = max(absolute, relative * |reported|).
                This handles cases where performance values are large (e.g., -200)
                and synthetic data noise causes proportional differences.
            require_multiple_evaluators: Whether to require multiple evaluators.
            min_evaluators: Minimum number of evaluator confirmations needed.
        """
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.require_multiple = require_multiple_evaluators
        self.min_evaluators = min_evaluators
        self._domains: dict[str, Domain] = {}
        self._pending: dict[str, Optimae] = {}

    def register_domain(self, domain: Domain) -> None:
        """Register a domain for validation."""
        self._domains[domain.id] = domain

    def submit_for_validation(self, optimae: Optimae) -> str:
        """Accept an optimae for validation.

        Args:
            optimae: The optimae to validate.

        Returns:
            The optimae ID for tracking.

        Raises:
            ValueError: If domain is not registered.
        """
        if optimae.domain_id not in self._domains:
            raise ValueError(f"Unknown domain: {optimae.domain_id}")

        self._pending[optimae.id] = optimae
        logger.info(
            "Optimae %s submitted for validation (domain=%s, reported=%.4f)",
            optimae.id[:12],
            optimae.domain_id,
            optimae.reported_performance,
        )
        return optimae.id

    def record_evaluation(
        self,
        optimae_id: str,
        verified_performance: float,
        used_synthetic: bool = False,
    ) -> ValidationResult:
        """Record an evaluation result from an evaluator.

        Args:
            optimae_id: ID of the optimae being validated.
            verified_performance: Performance measured by the evaluator.
            used_synthetic: Whether synthetic data was used.

        Returns:
            ValidationResult with the verdict.
        """
        optimae = self._pending.get(optimae_id)
        if optimae is None:
            return ValidationResult(
                optimae_id=optimae_id,
                domain_id="unknown",
                reported_performance=0.0,
                error="Optimae not found in pending validations",
            )

        domain = self._domains[optimae.domain_id]
        diff = abs(verified_performance - optimae.reported_performance)

        # Compute effective tolerance
        effective_tolerance = self.tolerance
        if self.relative_tolerance is not None:
            relative = self.relative_tolerance * abs(optimae.reported_performance)
            effective_tolerance = max(self.tolerance, relative)

        is_valid = diff <= effective_tolerance

        result = ValidationResult(
            optimae_id=optimae_id,
            domain_id=optimae.domain_id,
            reported_performance=optimae.reported_performance,
            verified_performance=verified_performance,
            is_valid=is_valid,
            used_synthetic_data=used_synthetic,
        )

        if is_valid:
            optimae.verified_performance = verified_performance
            optimae.accepted = True
            # Calculate increment
            if domain.current_best_performance is not None:
                if domain.higher_is_better:
                    optimae.performance_increment = max(
                        0.0, verified_performance - domain.current_best_performance
                    )
                else:
                    optimae.performance_increment = max(
                        0.0, domain.current_best_performance - verified_performance
                    )
            else:
                optimae.performance_increment = abs(verified_performance)

            # Update domain best
            domain.current_best_performance = verified_performance
            del self._pending[optimae_id]
            logger.info(
                "Optimae %s ACCEPTED (verified=%.4f, increment=%.4f)",
                optimae_id[:12],
                verified_performance,
                optimae.performance_increment,
            )
        else:
            del self._pending[optimae_id]
            logger.warning(
                "Optimae %s REJECTED (reported=%.4f, verified=%.4f, diff=%.4f > tol=%.4f)",
                optimae_id[:12],
                optimae.reported_performance,
                verified_performance,
                diff,
                effective_tolerance,
            )

        return result

    @property
    def pending_count(self) -> int:
        """Number of optimae awaiting validation."""
        return len(self._pending)
