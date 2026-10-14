"""Read-only champion projection from already committed acceptance payloads."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from doin_core.archive.body import ArchiveRefusal, canonical_bytes

from doin_node.archive.external import ExternalPayloadArchive


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArchiveRefusal("CHAMPION_VERIFIED_PERFORMANCE_REQUIRED")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArchiveRefusal("CHAMPION_VERIFIED_PERFORMANCE_NONFINITE") from exc
    if not math.isfinite(result):
        raise ArchiveRefusal("CHAMPION_VERIFIED_PERFORMANCE_NONFINITE")
    return result


def resolve_champion_payload(
    payload: Mapping[str, Any],
    domain_id: str,
    archive: ExternalPayloadArchive | None,
    *,
    verified_performance: float | None = None,
) -> dict[str, Any]:
    """Project a legacy or archive-backed OPTIMAE_ACCEPTED payload, or refuse.

    The caller must verify commitment and pair identity before supplying the
    core companion's performance. This function does not establish acceptance
    or derive quorum performance from candidate fitness or increments. Archive
    reads use the supplied resolver; no chain, cache, or input is mutated.
    The I6-N schema validator is needed only for archive-backed payloads.
    """
    if not isinstance(payload, Mapping):
        raise ArchiveRefusal("CHAMPION_PAYLOAD")
    if not isinstance(domain_id, str) or not domain_id:
        raise ArchiveRefusal("DOMAIN_MISMATCH")
    if "domain_id" in payload and payload["domain_id"] != domain_id:
        raise ArchiveRefusal("DOMAIN_MISMATCH")

    score = _score(payload.get("verified_performance", verified_performance))
    if verified_performance is not None and _score(verified_performance) != score:
        raise ArchiveRefusal("CHAMPION_VERIFIED_PERFORMANCE_MISMATCH")

    parameters = payload.get("parameters")
    if "parameters" in payload and not isinstance(parameters, Mapping):
        raise ArchiveRefusal("CHAMPION_PARAMETERS")
    reference = payload.get("candidate_archive_reference")
    embedded = (parameters or {}).get("_i6n_archive_reference")
    has_reference = "candidate_archive_reference" in payload
    has_embedded = "_i6n_archive_reference" in (parameters or {})
    if has_reference and has_embedded and reference != embedded:
        raise ArchiveRefusal("CHAMPION_REFERENCE_MISMATCH")
    if not has_reference and has_embedded:
        reference = embedded

    if has_reference or has_embedded:
        if not isinstance(reference, Mapping) or reference.get("domain_id") != domain_id:
            raise ArchiveRefusal("DOMAIN_MISMATCH")
        if archive is None:
            raise ArchiveRefusal("CHAMPION_ARCHIVE_REQUIRED")
        resolved = archive.resolve(reference)
        if resolved.domain_id != domain_id:
            raise ArchiveRefusal("DOMAIN_MISMATCH")
        try:
            from tools.i6n_search_space import GENOTYPE_SCHEMA
            from tools.i6n_archive.records import (
                ISLAND_CANDIDATE_SCHEMA,
                ArchiveRecordError,
                validate_island_candidate_payload,
            )
        except ImportError as exc:
            raise ArchiveRefusal("CHAMPION_SCHEMA_VALIDATOR_UNAVAILABLE") from exc
        if resolved.payload_schema != ISLAND_CANDIDATE_SCHEMA:
            raise ArchiveRefusal("CHAMPION_PAYLOAD_SCHEMA")
        try:
            candidate = validate_island_candidate_payload(resolved.payload)
        except (ArchiveRecordError, KeyError, TypeError, OverflowError) as exc:
            raise ArchiveRefusal("CHAMPION_BODY_INVALID") from exc
        if candidate["experiment"] != domain_id:
            raise ArchiveRefusal("DOMAIN_MISMATCH")
        effective = candidate["effective_config"]
        migration = {
            "_i6n_genotype_sha256": candidate["genotype_canonical_sha256"],
            "_i6n_origin_island": candidate["island_id"],
            "_i6n_envelope": {
                "schema": "i6n.island.champion.v1",
                "plan_sha256": candidate["plan_sha256"],
                "context_sha256": candidate["contract_sha256"],
                "horizon": candidate["horizon_hours"],
                "training_seed": candidate["seed"],
                "genotype_schema": GENOTYPE_SCHEMA,
            },
        }
        if parameters is not None:
            for key, expected in migration.items():
                if key in parameters and canonical_bytes(parameters[key]) != canonical_bytes(expected):
                    raise ArchiveRefusal(f"CHAMPION_IDENTITY_MISMATCH: {key}")
            supplied = {k: v for k, v in parameters.items()
                        if k != "_i6n_archive_reference" and k not in migration}
            if canonical_bytes(supplied) != canonical_bytes(effective):
                raise ArchiveRefusal("CHAMPION_GENOTYPE_MISMATCH")
        identities = (
            "genotype_canonical_sha256", "effective_config_sha256",
            "search_candidate_sha256", "archive_event_sha256", "plan_sha256",
            "contract_sha256", "checkpoint_identity_sha256",
        )
        for key in identities:
            if key in payload and payload[key] != candidate[key]:
                raise ArchiveRefusal(f"CHAMPION_IDENTITY_MISMATCH: {key}")
        aggregate = candidate["aggregate_metrics"]
        metrics = {
            **aggregate,
            "val_mae": aggregate["inner_mae"],
            "val_naive_mae": aggregate["inner_naive_mae"],
            "metric_schema": candidate["metric_schema"],
            "aggregation": candidate["aggregation"],
            "generation": candidate["generation"],
            "per_fold": {row["fold"]: row["ratio_to_naive"] for row in candidate["per_fold"]},
            **{key: candidate[key] for key in identities},
        }
        parameters = {**effective, **migration, "_i6n_archive_reference": dict(reference)}
    else:
        if parameters is None:
            raise ArchiveRefusal("CHAMPION_PARAMETERS")
        metrics = payload.get("champion_metrics")
        if metrics is None:
            metrics = {key: payload[key] for key in (
                "train_mae", "val_mae", "test_mae",
                "train_naive_mae", "val_naive_mae", "test_naive_mae",
            ) if key in payload}
        if not isinstance(metrics, Mapping):
            raise ArchiveRefusal("CHAMPION_METRICS")

    return deepcopy({
        "parameters": dict(parameters),
        "verified_performance": score,
        "champion_metrics": {
            key: value for key, value in metrics.items()
            if key not in {"_model_b64", "_best_model_b64"} and value is not None
        },
    })
