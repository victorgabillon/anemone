"""Build checkpoint payloads for tree and algorithm nodes."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from ._protocols import IncrementalStateCheckpointCodec
from .build_atoms import (
    _serialize_branch_collection,
    _serialize_checkpoint_atom_for_build,
    _serialize_parent_branches,
)
from .build_context import (
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
    _NodeCheckpointBuildCache,
)
from .build_evaluation_payloads import (
    _build_exploration_index_payload,
    _build_node_evaluation_payload,
)
from .build_state_payloads import (
    _build_checkpoint_state_payload,
    _representative_parent_link,
)
from .payloads import (
    AlgorithmNodeCheckpointPayload,
    LinkedChildCheckpointPayload,
    TreeCheckpointPayload,
)
from .state_handles import checkpoint_payload_for_reuse_or_none

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration


def _build_tree_payload(
    *,
    search: TreeExploration[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
) -> TreeCheckpointPayload:
    """Build the tree payload in stable depth/insertion descendant order."""
    iter_nodes_started_at = perf_counter()
    nodes = _iter_nodes_in_checkpoint_order(search)
    metrics.iter_nodes_s += perf_counter() - iter_nodes_started_at
    metrics.node_count = len(nodes)
    payload_nodes: list[AlgorithmNodeCheckpointPayload] = []
    for node in nodes:
        node_payload_started_at = perf_counter()
        payload_nodes.append(
            _build_node_payload(
                node=node,
                state_codec=state_codec,
                metrics=metrics,
                context=context,
            )
        )
        metrics.node_payload_total_s += perf_counter() - node_payload_started_at
    return TreeCheckpointPayload(
        root_node_id=search.tree.root_node.id,
        nodes=payload_nodes,
    )


def _iter_nodes_in_checkpoint_order(
    search: TreeExploration[Any],
) -> list[AlgorithmNode[Any]]:
    """Return live nodes in stable depth order, then descendant insertion order."""
    descendants = search.tree.descendants
    return [
        node
        for tree_depth in descendants.range()
        for node in descendants[tree_depth].values()
    ]


def _build_node_payload(
    *,
    node: AlgorithmNode[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
) -> AlgorithmNodeCheckpointPayload:
    """Build one node checkpoint payload without mutating runtime state."""
    # Safe cross-generation evaluation payload reuse needs one explicit runtime
    # version that changes whenever any serialized evaluation field changes.
    # Until that exists, restored/checkpoint-backed nodes are only counted as
    # reuse candidates and are always rebuilt to avoid stale payload reuse after
    # reevaluation patches, backup propagation, or frontier/order updates.
    if checkpoint_payload_for_reuse_or_none(node.state_handle) is not None:
        metrics.evaluation_payload_reuse_candidates_missing_version += 1
        metrics.evaluation_payload_reuse_blocked_missing_version += 1

    node_cache = _build_node_checkpoint_cache(node, context=context)
    representative_parent_started_at = perf_counter()
    (
        representative_parent,
        parent_node_id,
        _branch_from_parent,
        branch_from_parent_payload,
    ) = _representative_parent_link(
        node,
        node_cache=node_cache,
        context=context,
    )
    metrics.representative_parent_calls += 1
    metrics.representative_parent_total_s += (
        perf_counter() - representative_parent_started_at
    )
    unopened_branches_started_at = perf_counter()
    unopened_branches = _serialize_branch_collection(
        node.iter_unopened_branches(),
        context=context,
    )
    metrics.unopened_branches_total_s += perf_counter() - unopened_branches_started_at
    linked_children_started_at = perf_counter()
    linked_children = _serialize_linked_children(
        node.iter_child_links(),
        context=context,
    )
    metrics.linked_children_total_s += perf_counter() - linked_children_started_at
    evaluation_started_at = perf_counter()
    evaluation = _build_node_evaluation_payload(node, context=context)
    metrics.evaluation_payload_total_s += perf_counter() - evaluation_started_at
    exploration_index_started_at = perf_counter()
    exploration_index = _build_exploration_index_payload(node)
    metrics.exploration_index_total_s += perf_counter() - exploration_index_started_at
    return AlgorithmNodeCheckpointPayload(
        node_id=node.id,
        parent_node_id=parent_node_id,
        branch_from_parent=branch_from_parent_payload,
        depth=node.tree_depth,
        state_payload=_build_checkpoint_state_payload(
            node=node,
            representative_parent=representative_parent,
            state_codec=state_codec,
            metrics=metrics,
            context=context,
            node_cache=node_cache,
        ),
        generated_all_branches=node.all_branches_generated,
        unopened_branches=unopened_branches,
        linked_children=linked_children,
        evaluation=evaluation,
        exploration_index=exploration_index,
    )


def _build_node_checkpoint_cache(
    node: AlgorithmNode[Any],
    *,
    context: _CheckpointBuildContext,
) -> _NodeCheckpointBuildCache:
    """Precompute parent branch ordering reused across node helper calls."""
    return _NodeCheckpointBuildCache(
        parent_branch_entries=[
            _serialize_parent_branches(
                parent_node,
                branch_keys,
                context=context,
            )
            for parent_node, branch_keys in node.iter_parent_items()
        ]
    )


def _serialize_linked_children(
    child_links: Iterable[tuple[Any, AlgorithmNode[Any] | None]],
    *,
    context: _CheckpointBuildContext,
) -> list[LinkedChildCheckpointPayload]:
    """Serialize linked children in stable checkpoint-atom order."""
    metrics = context.metrics
    metrics.linked_children_calls += 1
    linked_children = [
        LinkedChildCheckpointPayload(
            branch_key=_serialize_checkpoint_atom_for_build(
                branch,
                context=context,
            ),
            child_node_id=child.id,
        )
        for branch, child in child_links
        if child is not None
    ]
    sort_started_at = perf_counter()
    linked_children.sort(key=lambda item: (repr(item.branch_key), item.child_node_id))
    metrics.linked_children_sort_calls += 1
    metrics.linked_children_sort_s += perf_counter() - sort_started_at
    return linked_children
