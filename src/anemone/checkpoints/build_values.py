"""Build-time checkpoint value serialization helpers."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from anemone.node_evaluation.common import canonical_value

from .build_atoms import _serialize_evaluation_atom_for_build
from .payloads import SerializedOverEventPayload, SerializedValuePayload
from .value_serialization import serialize_over_event_with

if TYPE_CHECKING:
    from valanga.evaluations import Value

    from .build_context import _CheckpointBuildContext


def _serialize_optional_value(
    value: Value | None,
    *,
    context: _CheckpointBuildContext,
    value_kind: str,
) -> SerializedValuePayload | None:
    """Serialize one optional Value payload."""
    if value is None:
        return None

    started_at = perf_counter()
    try:
        return _serialize_value_for_build(value, context=context)
    finally:
        elapsed_s = perf_counter() - started_at
        if value_kind == "direct":
            context.metrics.direct_value_serialize_calls += 1
            context.metrics.direct_value_serialize_s += elapsed_s
        else:
            context.metrics.backed_up_value_serialize_calls += 1
            context.metrics.backed_up_value_serialize_s += elapsed_s


def _serialize_value_for_build(
    value: Value,
    *,
    context: _CheckpointBuildContext,
) -> SerializedValuePayload:
    """Serialize one Value with build-local caching and detailed timing."""
    metrics = context.metrics
    metrics.serialize_value_calls += 1

    identity_key = id(value)
    cached_identity_entry = context.value_identity_serialization_cache.get(identity_key)
    if cached_identity_entry is not None and cached_identity_entry[0] is value:
        metrics.serialize_value_cache_hits += 1
        return cached_identity_entry[1]

    metrics.serialize_value_cache_misses += 1
    payload = _serialize_value_uncached_for_build(value, context=context)
    context.value_identity_serialization_cache[identity_key] = (value, payload)
    return payload


def _serialize_value_uncached_for_build(
    value: Value,
    *,
    context: _CheckpointBuildContext,
) -> SerializedValuePayload:
    """Serialize one Value payload while recording validation and atom work."""
    metrics = context.metrics
    started_at = perf_counter()
    try:
        validation_started_at = perf_counter()
        validated_value = canonical_value.validate_value_semantics(value)
        metrics.value_semantic_validation_calls += 1
        metrics.value_semantic_validation_s += perf_counter() - validation_started_at
        return SerializedValuePayload(
            score=validated_value.score,
            certainty=validated_value.certainty.name,
            over_event=_serialize_over_event_for_build(
                validated_value.over_event,
                context=context,
            ),
            line=(
                [
                    _serialize_evaluation_atom_for_build(branch, context=context)
                    for branch in validated_value.line
                ]
                if validated_value.line is not None
                else None
            ),
        )
    finally:
        metrics.serialize_value_total_s += perf_counter() - started_at


def _serialize_over_event_for_build(
    over_event: object | None,
    *,
    context: _CheckpointBuildContext,
) -> SerializedOverEventPayload | None:
    """Serialize over-event metadata while attributing atom work to evaluation."""
    return serialize_over_event_with(
        over_event,
        atom_serializer=lambda atom: _serialize_evaluation_atom_for_build(
            atom,
            context=context,
        ),
    )
