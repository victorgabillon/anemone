"""Checkpoint payload helpers for latest tree-expansion records."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001,TC003

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.tree_exploration import TreeExploration
from anemone.tree_manager import TreeExpansion, TreeExpansions

from .payloads import (
    CheckpointAtomPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .restore_atoms import _deserialize_optional_branch, _require_node
from .value_serialization import serialize_checkpoint_atom


def _serialize_optional_atom(value: object | None) -> CheckpointAtomPayload | None:
    """Serialize a small optional atom payload."""
    if value is None:
        return None
    return serialize_checkpoint_atom(value)


def _build_latest_tree_expansions_payload(
    search: TreeExploration[Any],
) -> TreeExpansionsCheckpointPayload:
    """Serialize selector-visible expansion records from the latest iteration."""
    return _build_tree_expansions_payload(search.latest_tree_expansions)


def _build_tree_expansions_payload(
    tree_expansions: TreeExpansions[Any],
) -> TreeExpansionsCheckpointPayload:
    """Serialize one TreeExpansions object by node ids and branch keys."""
    return TreeExpansionsCheckpointPayload(
        expansions_with_node_creation=[
            _build_tree_expansion_payload(tree_expansion)
            for tree_expansion in tree_expansions.expansions_with_node_creation
        ],
        expansions_without_node_creation=[
            _build_tree_expansion_payload(tree_expansion)
            for tree_expansion in tree_expansions.expansions_without_node_creation
        ],
    )


def _build_tree_expansion_payload(
    tree_expansion: TreeExpansion[Any],
) -> TreeExpansionCheckpointPayload:
    """Serialize one TreeExpansion object without state reconstruction data."""
    return TreeExpansionCheckpointPayload(
        child_node_id=tree_expansion.child_node.id,
        parent_node_id=(
            tree_expansion.parent_node.id
            if tree_expansion.parent_node is not None
            else None
        ),
        branch_key=_serialize_optional_atom(tree_expansion.branch_key),
        creation_child_node=tree_expansion.creation_child_node,
    )


def _restore_tree_expansions_runtime_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: TreeExpansionsCheckpointPayload | None,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore selector-visible latest expansion records when present."""
    if payload is not None:
        runtime.latest_tree_expansions = _restore_tree_expansions(
            payload,
            nodes_by_id=nodes_by_id,
        )


def _restore_tree_expansions(
    payload: TreeExpansionsCheckpointPayload,
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> TreeExpansions[AlgorithmNode[Any]]:
    """Restore selector-visible expansion records from node ids."""
    tree_expansions: TreeExpansions[AlgorithmNode[Any]] = TreeExpansions()
    for expansion_payload in payload.expansions_with_node_creation:
        tree_expansions.record_creation(
            _restore_tree_expansion(expansion_payload, nodes_by_id=nodes_by_id)
        )
    for expansion_payload in payload.expansions_without_node_creation:
        tree_expansions.record_connection(
            _restore_tree_expansion(expansion_payload, nodes_by_id=nodes_by_id)
        )
    return tree_expansions


def _restore_tree_expansion(
    payload: TreeExpansionCheckpointPayload,
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> TreeExpansion[AlgorithmNode[Any]]:
    """Restore one selector-visible expansion record."""
    return TreeExpansion(
        child_node=_require_node(nodes_by_id, payload.child_node_id),
        parent_node=(
            _require_node(nodes_by_id, payload.parent_node_id)
            if payload.parent_node_id is not None
            else None
        ),
        state_modifications=None,
        creation_child_node=payload.creation_child_node,
        branch_key=_deserialize_optional_branch(payload.branch_key),
    )
