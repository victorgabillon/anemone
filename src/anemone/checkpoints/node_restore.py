"""Restore checkpoint node shells, runtime state, and tree structure."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from anemone import trees
from anemone._valanga_types import AnyTurnState
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.trees.descendants import RangedDescendants
from anemone.utils.logger import checkpoint_logger

from .evaluation_payloads import _restore_evaluation, _restore_exploration_index
from .metadata import _log_restore_phase
from .payloads import (
    AlgorithmNodeCheckpointPayload,
    CheckpointAtomPayload,
    ExplorationIndexCheckpointPayload,
    LinkedChildCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
)
from .restore_atoms import _deserialize_branch, _require_node

if TYPE_CHECKING:
    from valanga import StateTag

    from anemone.nodes.state_handles import StateHandle


class _CheckpointNodeShell(Protocol):
    """Compact node metadata needed before full runtime-state restoration."""

    node_id: int
    parent_node_id: int | None
    depth: int
    state_summary: object | None


class _CheckpointNodeRuntime(Protocol):
    """Per-node runtime metadata needed after compact node creation."""

    node_id: int
    generated_all_branches: bool
    unopened_branches: list[CheckpointAtomPayload]
    linked_children: list[LinkedChildCheckpointPayload]
    evaluation: NodeEvaluationCheckpointPayload | None
    exploration_index: ExplorationIndexCheckpointPayload | None


def _create_nodes[
    StateT: AnyTurnState,
](
    *,
    node_factory: Any,
    payload: SearchRuntimeCheckpointPayload,
    state_handles_by_id: Mapping[int, StateHandle[StateT]],
) -> dict[int, AlgorithmNode[StateT]]:
    """Create every node without linking parent/child edges yet.

    ``AlgorithmNodeFactory`` forwards ``count`` to ``TreeNode.id_``. Supplying
    the checkpoint id here keeps restored ids stable without relying on any
    hidden global counter.
    """
    return _create_nodes_from_node_payloads(
        node_factory=node_factory,
        node_payloads=payload.tree.nodes,
        state_handles_by_id=state_handles_by_id,
    )


def _create_nodes_from_node_payloads[
    StateT: AnyTurnState,
](
    *,
    node_factory: Any,
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    state_handles_by_id: Mapping[int, StateHandle[StateT]],
) -> dict[int, AlgorithmNode[StateT]]:
    """Create every node from payload metadata without linking edges yet."""
    nodes_by_id: dict[int, AlgorithmNode[StateT]] = {}
    for node_payload in sorted(node_payloads, key=lambda item: item.depth):
        node = node_factory.create(
            state_handle=state_handles_by_id[node_payload.node_id],
            tree_depth=node_payload.depth,
            count=node_payload.node_id,
            parent_node=None,
            branch_from_parent=None,
            modifications=None,
            # Restored checkpoint nodes intentionally keep
            # ``state_representation`` unset. Restore must not force state
            # materialization just to rebuild evaluator-side representations.
            # New nodes created after resume may still build representations
            # through the normal factory path.
            build_state_representation=False,
        )
        nodes_by_id[node_payload.node_id] = node
    return nodes_by_id


def _create_nodes_from_node_shells[
    StateT: AnyTurnState,
](
    *,
    node_factory: Any,
    node_shells: Iterable[_CheckpointNodeShell],
    state_handles_by_id: Mapping[int, StateHandle[StateT]],
) -> dict[int, AlgorithmNode[StateT]]:
    """Create every node from compact metadata without linking edges yet."""
    nodes_by_id: dict[int, AlgorithmNode[StateT]] = {}
    for node_shell in sorted(node_shells, key=lambda item: item.depth):
        node = node_factory.create(
            state_handle=state_handles_by_id[node_shell.node_id],
            tree_depth=node_shell.depth,
            count=node_shell.node_id,
            parent_node=None,
            branch_from_parent=None,
            modifications=None,
            build_state_representation=False,
        )
        nodes_by_id[node_shell.node_id] = node
    return nodes_by_id


def _link_nodes(
    node_payloads: Iterable[_CheckpointNodeRuntime],
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore graph edges from authoritative parent child-link payloads."""
    for node_payload in node_payloads:
        parent_node = nodes_by_id[node_payload.node_id]
        for linked_child in node_payload.linked_children:
            child_node = _require_node(nodes_by_id, linked_child.child_node_id)
            branch = _deserialize_branch(linked_child.branch_key)
            parent_node.set_child_for_branch(branch, child_node)
            child_node.add_parent(branch_key=branch, new_parent_node=parent_node)


def _restore_node_runtime_state(
    node_payloads: Iterable[_CheckpointNodeRuntime],
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore per-node structural flags, evaluation, and exploration index data."""
    for node_payload in node_payloads:
        node = nodes_by_id[node_payload.node_id]
        node.all_branches_generated = node_payload.generated_all_branches
        node.set_unopened_branches(
            _deserialize_branch(branch_payload)
            for branch_payload in node_payload.unopened_branches
        )
        _restore_evaluation(node=node, payload=node_payload.evaluation)
        _restore_exploration_index(node=node, payload=node_payload.exploration_index)


def _build_tree[
    StateT: AnyTurnState,
](
    *,
    payload: SearchRuntimeCheckpointPayload,
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
) -> trees.Tree[AlgorithmNode[StateT]]:
    """Build the restored Tree and descendant registry from restored nodes."""
    return _build_tree_from_node_payloads(
        root_node_id=payload.tree.root_node_id,
        node_payloads=payload.tree.nodes,
        nodes_by_id=nodes_by_id,
    )


def _build_tree_from_node_payloads[
    StateT: AnyTurnState,
](
    *,
    root_node_id: int,
    node_payloads: Sequence[AlgorithmNodeCheckpointPayload],
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
    branch_count: int | None = None,
) -> trees.Tree[AlgorithmNode[StateT]]:
    """Build the restored Tree and descendant registry from node payloads."""
    with _log_restore_phase("build_tree.root_lookup", root_node_id=root_node_id):
        root_node = nodes_by_id[root_node_id]
    with _log_restore_phase("build_tree.descendants_init"):
        descendants = RangedDescendants[AlgorithmNode[StateT]]()
    with _log_restore_phase(
        "build_tree.group_nodes_by_depth", node_count=len(node_payloads)
    ):
        grouped_payloads = _group_node_payloads_by_depth(node_payloads)
    with _log_restore_phase(
        "build_tree.populate_descendants",
        depth_count=len(grouped_payloads),
        node_count=len(node_payloads),
    ):
        _populate_descendants_for_restore(
            descendants=descendants,
            grouped_payloads=grouped_payloads,
            nodes_by_id=nodes_by_id,
        )
    with _log_restore_phase("build_tree.tree_construct", node_count=len(nodes_by_id)):
        restored_tree = trees.Tree(root_node=root_node, descendants=descendants)
    restored_tree.nodes_count = len(nodes_by_id)
    restored_tree.branch_count = (
        branch_count
        if branch_count is not None
        else sum(len(node_payload.linked_children) for node_payload in node_payloads)
    )
    return restored_tree


def _build_tree_from_node_shells[
    StateT: AnyTurnState,
](
    *,
    root_node_id: int,
    node_shells: Sequence[_CheckpointNodeShell],
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
    branch_count: int,
) -> trees.Tree[AlgorithmNode[StateT]]:
    """Build the restored Tree and descendant registry from compact node shells."""
    with _log_restore_phase("build_tree.root_lookup", root_node_id=root_node_id):
        root_node = nodes_by_id[root_node_id]
    with _log_restore_phase("build_tree.descendants_init"):
        descendants = RangedDescendants[AlgorithmNode[StateT]]()
    with _log_restore_phase(
        "build_tree.group_nodes_by_depth", node_count=len(node_shells)
    ):
        grouped_shells = _group_node_shells_by_depth(node_shells)
    with _log_restore_phase(
        "build_tree.populate_descendants",
        depth_count=len(grouped_shells),
        node_count=len(node_shells),
    ):
        _populate_descendants_from_node_shells_for_restore(
            descendants=descendants,
            grouped_shells=grouped_shells,
            nodes_by_id=nodes_by_id,
        )
    with _log_restore_phase("build_tree.tree_construct", node_count=len(nodes_by_id)):
        restored_tree = trees.Tree(root_node=root_node, descendants=descendants)
    restored_tree.nodes_count = len(nodes_by_id)
    restored_tree.branch_count = branch_count
    return restored_tree


def _group_node_payloads_by_depth(
    node_payloads: Sequence[AlgorithmNodeCheckpointPayload],
) -> dict[int, list[AlgorithmNodeCheckpointPayload]]:
    """Group checkpoint node payloads by depth in stable payload order."""
    grouped_payloads: dict[int, list[AlgorithmNodeCheckpointPayload]] = {}
    for node_payload in sorted(node_payloads, key=lambda item: item.depth):
        grouped_payloads.setdefault(node_payload.depth, []).append(node_payload)
    return grouped_payloads


def _group_node_shells_by_depth(
    node_shells: Sequence[_CheckpointNodeShell],
) -> dict[int, list[_CheckpointNodeShell]]:
    """Group checkpoint node shells by depth in stable shell order."""
    grouped_shells: dict[int, list[_CheckpointNodeShell]] = {}
    for node_shell in sorted(node_shells, key=lambda item: item.depth):
        grouped_shells.setdefault(node_shell.depth, []).append(node_shell)
    return grouped_shells


def _populate_descendants_for_restore[
    StateT: AnyTurnState,
](
    *,
    descendants: RangedDescendants[AlgorithmNode[StateT]],
    grouped_payloads: Mapping[int, list[AlgorithmNodeCheckpointPayload]],
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
) -> None:
    """Populate descendants bookkeeping in bulk without incremental updates."""
    if not grouped_payloads:
        return

    summary_tag_count = 0
    fallback_tag_count = 0
    descendants.descendants_at_tree_depth = {}
    descendants.number_of_descendants_at_tree_depth = {}
    descendants.number_of_descendants = 0
    descendants.min_tree_depth = min(grouped_payloads)
    descendants.max_tree_depth = max(grouped_payloads)

    for tree_depth in range(descendants.min_tree_depth, descendants.max_tree_depth + 1):
        payloads_at_depth = grouped_payloads.get(tree_depth)
        if payloads_at_depth is None:
            descendants.descendants_at_tree_depth[tree_depth] = {}
            descendants.number_of_descendants_at_tree_depth[tree_depth] = 0
            continue

        descendants_at_depth: dict[StateTag, AlgorithmNode[StateT]] = {}
        for node_payload in payloads_at_depth:
            node = nodes_by_id[node_payload.node_id]
            state_tag, used_summary_tag = _checkpoint_node_state_tag(
                node=node,
                node_payload=node_payload,
            )
            if used_summary_tag:
                summary_tag_count += 1
            else:
                fallback_tag_count += 1
            assert state_tag not in descendants_at_depth
            descendants_at_depth[state_tag] = node
        descendants.descendants_at_tree_depth[tree_depth] = descendants_at_depth
        descendants.number_of_descendants_at_tree_depth[tree_depth] = len(
            descendants_at_depth
        )
        descendants.number_of_descendants += len(descendants_at_depth)
    checkpoint_logger.debug(
        "[checkpoint-restore] phase=build_tree.populate_descendants_tags "
        "summary_tag_count=%s fallback_tag_count=%s",
        summary_tag_count,
        fallback_tag_count,
    )


def _populate_descendants_from_node_shells_for_restore[
    StateT: AnyTurnState,
](
    *,
    descendants: RangedDescendants[AlgorithmNode[StateT]],
    grouped_shells: Mapping[int, list[_CheckpointNodeShell]],
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
) -> None:
    """Populate descendants bookkeeping from compact checkpoint shell metadata."""
    if not grouped_shells:
        return

    summary_tag_count = 0
    fallback_tag_count = 0
    descendants.descendants_at_tree_depth = {}
    descendants.number_of_descendants_at_tree_depth = {}
    descendants.number_of_descendants = 0
    descendants.min_tree_depth = min(grouped_shells)
    descendants.max_tree_depth = max(grouped_shells)

    for tree_depth in range(descendants.min_tree_depth, descendants.max_tree_depth + 1):
        shells_at_depth = grouped_shells.get(tree_depth)
        if shells_at_depth is None:
            descendants.descendants_at_tree_depth[tree_depth] = {}
            descendants.number_of_descendants_at_tree_depth[tree_depth] = 0
            continue

        descendants_at_depth: dict[StateTag, AlgorithmNode[StateT]] = {}
        for node_shell in shells_at_depth:
            node = nodes_by_id[node_shell.node_id]
            state_tag, used_summary_tag = _checkpoint_node_shell_state_tag(
                node=node,
                node_shell=node_shell,
            )
            if used_summary_tag:
                summary_tag_count += 1
            else:
                fallback_tag_count += 1
            assert state_tag not in descendants_at_depth
            descendants_at_depth[state_tag] = node
        descendants.descendants_at_tree_depth[tree_depth] = descendants_at_depth
        descendants.number_of_descendants_at_tree_depth[tree_depth] = len(
            descendants_at_depth
        )
        descendants.number_of_descendants += len(descendants_at_depth)
    checkpoint_logger.debug(
        "[checkpoint-restore] phase=build_tree.populate_descendants_tags "
        "summary_tag_count=%s fallback_tag_count=%s",
        summary_tag_count,
        fallback_tag_count,
    )


def _checkpoint_node_state_tag(
    *,
    node: AlgorithmNode[Any],
    node_payload: AlgorithmNodeCheckpointPayload,
) -> tuple[StateTag, bool]:
    """Return the checkpoint node tag, preferring summary metadata when available."""
    state_summary = node_payload.state_payload.state_summary
    summary_tag = _checkpoint_summary_tag(state_summary)
    if summary_tag is not None:
        return summary_tag, True
    return node.tag, False


def _checkpoint_node_shell_state_tag(
    *,
    node: AlgorithmNode[Any],
    node_shell: _CheckpointNodeShell,
) -> tuple[StateTag, bool]:
    """Return the checkpoint node tag from compact shell summary metadata."""
    summary_tag = _checkpoint_summary_tag(node_shell.state_summary)
    if summary_tag is not None:
        return summary_tag, True
    return node.tag, False


def _checkpoint_summary_tag(state_summary: object | None) -> StateTag | None:
    """Return a checkpoint summary tag when one is stored explicitly."""
    if state_summary is None:
        return None
    if isinstance(state_summary, Sequence) and not isinstance(
        state_summary, str | bytes | bytearray
    ):
        state_summary_sequence = cast("Sequence[object]", state_summary)
        if len(state_summary_sequence) >= 1:
            summary_tag = state_summary_sequence[0]
            return summary_tag if isinstance(summary_tag, Hashable) else None
        return None
    state_summary_with_attrs: Any = state_summary
    if hasattr(state_summary, "tag"):
        return cast("StateTag", state_summary_with_attrs.tag)
    if hasattr(state_summary, "state_tag"):
        return cast("StateTag", state_summary_with_attrs.state_tag)
    if isinstance(state_summary, Mapping):
        if "tag" in state_summary:
            return cast("StateTag", state_summary["tag"])
        if "state_tag" in state_summary:
            return cast("StateTag", state_summary["state_tag"])
    return None
