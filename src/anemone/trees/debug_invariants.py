"""Opt-in diagnostics for live descendants/checkpoint invariants."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from valanga import StateTag

    from anemone.basics import TreeDepth
    from anemone.nodes import ITreeNode
    from anemone.trees import Tree


DESCENDANTS_INVARIANT_LOG_PREFIX = "[descendants-invariant]"
DESCENDANTS_INVARIANT_ENV_VAR = "ANEMONE_DEBUG_DESCENDANTS_INVARIANTS"
DESCENDANTS_INVARIANT_FAILURE = "descendants invariant failed"
EXPORTED_STATE_TAG_INVARIANT_FAILURE = "exported node state-tag invariant failed"


class _NodeWithStateTag(Protocol):
    """Minimal node surface used by exported checkpoint duplicate checks."""

    id: int
    tree_depth: int
    state: Any


@dataclass(slots=True)
class _Failure:
    """One descendants invariant failure to include in structured debug logs."""

    depth: TreeDepth
    stored_tag: StateTag | None
    node_id: int
    node_tag: StateTag | None
    state_tag: StateTag | None
    node_depth: int
    duplicate_peer_node_ids: list[int] = field(default_factory=list)


def descendants_invariant_debug_enabled() -> bool:
    """Return whether descendants invariant diagnostics should run."""
    return os.environ.get(DESCENDANTS_INVARIANT_ENV_VAR) == "1"


def validate_descendants_tags(
    tree: Tree[ITreeNode[Any]],
    *,
    phase: str,
) -> None:
    """Validate descendants registry keys against live node/state metadata."""
    if not descendants_invariant_debug_enabled():
        return

    failures: list[_Failure] = []
    node_count = 0
    state_tag_groups_by_depth: dict[
        TreeDepth, dict[StateTag | None, list[_NodeWithStateTag]]
    ] = defaultdict(lambda: defaultdict(list))
    depth_count = _depth_count(tree.descendants.range())

    anemone_logger.info(
        "%s phase=%s status=start node_count=0 depth_count=%s "
        "mismatch_count=0 duplicate_state_tag_groups=0",
        DESCENDANTS_INVARIANT_LOG_PREFIX,
        phase,
        depth_count,
    )

    for tree_depth in tree.descendants.range():
        for stored_tag, node in tree.descendants[tree_depth].items():
            node_count += 1
            node_tag = _node_tag(node)
            state_tag = _state_tag(node)
            state_tag_groups_by_depth[tree_depth][state_tag].append(node)
            if (
                stored_tag != node_tag
                or node_tag != state_tag
                or node.tree_depth != tree_depth
            ):
                failures.append(
                    _Failure(
                        depth=tree_depth,
                        stored_tag=stored_tag,
                        node_id=node.id,
                        node_tag=node_tag,
                        state_tag=state_tag,
                        node_depth=node.tree_depth,
                    )
                )

    duplicate_failures = _duplicate_state_tag_failures(state_tag_groups_by_depth)
    failures.extend(duplicate_failures)
    _log_result(
        phase=phase,
        node_count=node_count,
        depth_count=depth_count,
        mismatch_count=len(failures) - len(duplicate_failures),
        duplicate_state_tag_groups=len(duplicate_failures),
        failures=failures,
    )
    if failures:
        raise AssertionError(DESCENDANTS_INVARIANT_FAILURE)


def summarize_duplicate_state_tags(
    tree: Tree[ITreeNode[Any]],
    *,
    phase: str,
    limit: int = 20,
) -> None:
    """Log duplicate ``state.tag`` groups in descendants without raising."""
    if not descendants_invariant_debug_enabled():
        return

    state_tag_groups_by_depth: dict[
        TreeDepth, dict[StateTag | None, list[_NodeWithStateTag]]
    ] = defaultdict(lambda: defaultdict(list))
    node_count = 0
    for tree_depth in tree.descendants.range():
        for node in tree.descendants[tree_depth].values():
            node_count += 1
            state_tag_groups_by_depth[tree_depth][_state_tag(node)].append(node)

    duplicate_failures = _duplicate_state_tag_failures(state_tag_groups_by_depth)
    _log_result(
        phase=phase,
        node_count=node_count,
        depth_count=len(state_tag_groups_by_depth),
        mismatch_count=0,
        duplicate_state_tag_groups=len(duplicate_failures),
        failures=duplicate_failures[:limit],
    )


def validate_exported_node_state_tags(
    nodes: Iterable[_NodeWithStateTag],
    *,
    phase: str,
) -> None:
    """Validate duplicate ``(tree_depth, state.tag)`` groups among exported nodes."""
    if not descendants_invariant_debug_enabled():
        return

    node_list = list(nodes)
    state_tag_groups_by_depth: dict[
        TreeDepth, dict[StateTag | None, list[_NodeWithStateTag]]
    ] = defaultdict(lambda: defaultdict(list))
    for node in node_list:
        state_tag_groups_by_depth[node.tree_depth][_state_tag(node)].append(node)

    duplicate_failures = _duplicate_state_tag_failures(state_tag_groups_by_depth)
    _log_result(
        phase=phase,
        node_count=len(node_list),
        depth_count=len(state_tag_groups_by_depth),
        mismatch_count=0,
        duplicate_state_tag_groups=len(duplicate_failures),
        failures=duplicate_failures,
    )
    if duplicate_failures:
        raise AssertionError(EXPORTED_STATE_TAG_INVARIANT_FAILURE)


def _depth_count(depths: range) -> int:
    """Return the number of depths represented by a descendants range."""
    return len(depths)


def _node_tag(node: _NodeWithStateTag) -> StateTag | None:
    """Return a node tag for diagnostics."""
    return getattr(node, "tag", None)


def _state_tag(node: _NodeWithStateTag) -> StateTag | None:
    """Return a state tag for diagnostics, tolerating partial test doubles."""
    state = getattr(node, "state", None)
    return getattr(state, "tag", None)


def _duplicate_state_tag_failures(
    groups_by_depth: Mapping[
        TreeDepth, Mapping[StateTag | None, list[_NodeWithStateTag]]
    ],
) -> list[_Failure]:
    """Return one representative failure for each duplicate state-tag group."""
    failures: list[_Failure] = []
    for depth, groups in groups_by_depth.items():
        for state_tag, nodes in groups.items():
            if len(nodes) <= 1:
                continue
            first_node = nodes[0]
            failures.append(
                _Failure(
                    depth=depth,
                    stored_tag=None,
                    node_id=first_node.id,
                    node_tag=_node_tag(first_node),
                    state_tag=state_tag,
                    node_depth=first_node.tree_depth,
                    duplicate_peer_node_ids=[node.id for node in nodes[1:]],
                )
            )
    return failures


def _log_result(
    *,
    phase: str,
    node_count: int,
    depth_count: int,
    mismatch_count: int,
    duplicate_state_tag_groups: int,
    failures: list[_Failure],
    limit: int = 20,
) -> None:
    """Emit structured diagnostics for one invariant pass."""
    status = "failed" if failures else "done"
    anemone_logger.info(
        "%s phase=%s status=%s node_count=%s depth_count=%s "
        "mismatch_count=%s duplicate_state_tag_groups=%s",
        DESCENDANTS_INVARIANT_LOG_PREFIX,
        phase,
        status,
        node_count,
        depth_count,
        mismatch_count,
        duplicate_state_tag_groups,
    )
    for failure in failures[:limit]:
        anemone_logger.error(
            "%s phase=%s status=failed depth=%s stored_tag=%r node_id=%s "
            "node_tag=%r state_tag=%r node_depth=%s duplicate_peer_node_ids=%s",
            DESCENDANTS_INVARIANT_LOG_PREFIX,
            phase,
            failure.depth,
            failure.stored_tag,
            failure.node_id,
            failure.node_tag,
            failure.state_tag,
            failure.node_depth,
            failure.duplicate_peer_node_ids,
        )
