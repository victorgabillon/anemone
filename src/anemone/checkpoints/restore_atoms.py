"""Shared small restore helpers for checkpoint node references and atoms."""

# pyright: reportUnusedFunction=false

# ruff: noqa: TC001,TC003

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

from .metadata import CheckpointRestoreError
from .payloads import CheckpointAtomPayload
from .value_serialization import deserialize_checkpoint_atom

if TYPE_CHECKING:
    from valanga import BranchKey


def _require_node(
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
    node_id: int,
) -> AlgorithmNode[Any]:
    """Return one restored node or raise a checkpoint-specific error."""
    try:
        return nodes_by_id[node_id]
    except KeyError as exc:
        raise CheckpointRestoreError.unknown_node_id(node_id) from exc


def _deserialize_optional_branch(
    payload: CheckpointAtomPayload | None,
) -> BranchKey | None:
    """Deserialize an optional branch key payload."""
    if payload is None:
        return None
    return _deserialize_branch(payload)


def _deserialize_branch(payload: CheckpointAtomPayload) -> BranchKey:
    """Deserialize a branch key checkpoint atom."""
    return deserialize_checkpoint_atom(payload)
