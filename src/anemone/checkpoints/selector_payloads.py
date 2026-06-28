"""Checkpoint selector payload build and restore helpers."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001,TC003

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, TypeGuard, cast, runtime_checkable

from anemone.node_selector import StatefulNodeSelector
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective
from anemone.tree_exploration import TreeExploration

from .payloads import (
    SelectorCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .restore_atoms import _require_node


@runtime_checkable
class _DepthCursorSelector(Protocol):
    """Selector component with a restorable depth cursor."""

    current_depth_to_expand: int


def _build_selector_state_payload(
    search: TreeExploration[Any],
) -> SelectorCheckpointPayload | None:
    """Serialize optional selector-private checkpoint state when supported."""
    objective = search.tree.root_node.tree_evaluation.required_objective
    if not _is_single_agent_objective(objective):
        return None
    if not isinstance(search.node_selector, StatefulNodeSelector):
        return None
    stateful_selector = cast(
        "StatefulNodeSelector[AlgorithmNode[Any]]",
        search.node_selector,
    )
    stateful_selector.refresh_state_for_checkpoint(
        tree=search.tree,
        objective=objective,
        latest_tree_expansions=search.latest_tree_expansions,
    )
    return stateful_selector.build_checkpoint_payload(objective)


def _is_single_agent_objective(
    objective: object,
) -> TypeGuard[SingleAgentMaxObjective[Any]]:
    """Return whether one objective is a typed single-agent objective."""
    return isinstance(objective, SingleAgentMaxObjective)


def _restore_explicit_selector_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: SelectorCheckpointPayload | None,
) -> None:
    """Restore optional selector-private checkpoint state when supported."""
    if payload is None:
        return

    objective = runtime.tree.root_node.tree_evaluation.required_objective
    if not isinstance(objective, SingleAgentMaxObjective):
        return
    if not isinstance(runtime.node_selector, StatefulNodeSelector):
        return

    stateful_selector = cast(
        "StatefulNodeSelector[AlgorithmNode[Any]]",
        runtime.node_selector,
    )
    stateful_selector.restore_from_checkpoint_payload(
        tree=runtime.tree,
        objective=objective,
        payload=payload,
    )


def _restore_inferred_depth_selector_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: TreeExpansionsCheckpointPayload | None,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore simple depth selector cursors inferred from latest expansions.

    This is intentionally marked as inferred, not exact generic selector
    restoration. Uniform's only local mutable state is the next depth cursor,
    and after a completed Uniform iteration the latest touched child depth is
    exactly that next cursor. Other selector-local state should become explicit
    checkpoint payload in a future selector checkpointing PR.
    """
    if payload is None:
        return
    next_depth = _infer_latest_expansion_depth(
        payload=payload,
        nodes_by_id=nodes_by_id,
    )
    if next_depth is None:
        return

    for selector in _iter_selector_components(runtime.node_selector):
        if isinstance(selector, _DepthCursorSelector):
            selector.current_depth_to_expand = next_depth


def _infer_latest_expansion_depth(
    *,
    payload: TreeExpansionsCheckpointPayload,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> int | None:
    """Infer Uniform's next depth cursor from the latest touched child depth."""
    latest_depths = [
        _require_node(nodes_by_id, expansion.child_node_id).tree_depth
        for expansion in [
            *payload.expansions_with_node_creation,
            *payload.expansions_without_node_creation,
        ]
    ]
    if not latest_depths:
        return None
    return max(latest_depths)


def _iter_selector_components(selector: object) -> Iterable[object]:
    """Yield a composed selector and nested base selectors."""
    current: object | None = selector
    while current is not None:
        yield current
        current = getattr(current, "base", None)
