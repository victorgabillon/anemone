"""Build-time checkpoint atom and branch serialization helpers."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from .build_context import _CheckpointBuildContext, _ParentBranchSerialization
from .payloads import CheckpointAtomPayload
from .value_serialization import serialize_checkpoint_atom

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga import BranchKey

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode


def _serialize_branch_collection(
    branches: Iterable[Any],
    *,
    context: _CheckpointBuildContext,
    atom_scope: str = "default",
) -> list[CheckpointAtomPayload]:
    """Serialize a branch collection in deterministic checkpoint-atom order."""
    metrics = context.metrics
    metrics.branch_collection_calls += 1
    started_at = perf_counter()
    try:
        atom_serializer = _serialize_checkpoint_atom_for_build
        if atom_scope == "evaluation":
            atom_serializer = _serialize_evaluation_atom_for_build
        serialized_branches = [
            atom_serializer(branch, context=context) for branch in branches
        ]
        sort_started_at = perf_counter()
        serialized_branches.sort(key=repr)
        metrics.branch_collection_sort_calls += 1
        metrics.branch_collection_sort_s += perf_counter() - sort_started_at
        return serialized_branches
    finally:
        metrics.branch_collection_total_s += perf_counter() - started_at


def _serialize_optional_atom(
    value: object | None,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload | None:
    """Serialize one optional atom with regular build attribution."""
    if value is None:
        return None
    return _serialize_checkpoint_atom_for_build(value, context=context)


def _serialize_optional_evaluation_atom(
    value: object | None,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload | None:
    """Serialize one optional atom while attributing metrics to evaluation work."""
    if value is None:
        return None
    return _serialize_evaluation_atom_for_build(value, context=context)


def _serialize_parent_branches(
    parent_node: AlgorithmNode[Any],
    branches: Iterable[BranchKey],
    *,
    context: _CheckpointBuildContext,
) -> _ParentBranchSerialization:
    """Return stable ordered branch serializations for one parent edge set."""
    serialized_pairs = [
        (
            branch,
            _serialize_checkpoint_atom_for_build(branch, context=context),
        )
        for branch in branches
    ]
    sort_started_at = perf_counter()
    serialized_pairs.sort(key=lambda item: repr(item[1]))
    context.metrics.branch_collection_sort_calls += 1
    context.metrics.branch_collection_sort_s += perf_counter() - sort_started_at
    return _ParentBranchSerialization(
        parent_node=parent_node,
        ordered_branches=tuple(branch for branch, _payload in serialized_pairs),
        serialized_branches=tuple(payload for _branch, payload in serialized_pairs),
    )


def _serialize_checkpoint_atom_for_build(
    value: object,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload:
    """Serialize checkpoint atoms with build-local memoization for hashable atoms."""
    metrics = context.metrics
    metrics.atom_serialize_calls += 1
    try:
        hash(value)
    except TypeError:
        started_at = perf_counter()
        payload = serialize_checkpoint_atom(value)
        metrics.atom_serialize_total_s += perf_counter() - started_at
        return payload

    if value in context.atom_serialization_cache:
        metrics.atom_serialize_cache_hits += 1
        return context.atom_serialization_cache[value]

    metrics.atom_serialize_cache_misses += 1
    started_at = perf_counter()
    payload = serialize_checkpoint_atom(value)
    metrics.atom_serialize_total_s += perf_counter() - started_at
    context.atom_serialization_cache[value] = payload
    return payload


def _serialize_evaluation_atom_for_build(
    value: object,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload:
    """Serialize one atom while attributing metrics to evaluation payload work."""
    metrics = context.metrics
    metrics.evaluation_atom_serialize_calls += 1
    atom_cache_hits_before = metrics.atom_serialize_cache_hits
    atom_cache_misses_before = metrics.atom_serialize_cache_misses
    atom_total_before = metrics.atom_serialize_total_s
    payload = _serialize_checkpoint_atom_for_build(value, context=context)
    metrics.evaluation_atom_serialize_total_s += (
        metrics.atom_serialize_total_s - atom_total_before
    )
    metrics.evaluation_atom_serialize_cache_hits += (
        metrics.atom_serialize_cache_hits - atom_cache_hits_before
    )
    metrics.evaluation_atom_serialize_cache_misses += (
        metrics.atom_serialize_cache_misses - atom_cache_misses_before
    )
    return payload
