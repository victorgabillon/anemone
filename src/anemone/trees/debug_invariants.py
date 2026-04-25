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
DESCENDANTS_INVARIANT_STEP_INTERVAL_ENV_VAR = (
    "ANEMONE_DEBUG_DESCENDANTS_INVARIANTS_STEP_INTERVAL"
)
DEFAULT_DESCENDANTS_INVARIANT_STEP_INTERVAL = 50
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
    first_seen_mismatch: str | None = None
    registered_tag: StateTag | None = None
    registered_state_tag: StateTag | None = None
    registered_depth: TreeDepth | None = None
    registered_node_id: int | None = None
    registered_parent_node_id: int | None = None
    duplicate_peer_node_ids: list[int] = field(default_factory=list)


def descendants_invariant_debug_enabled() -> bool:
    """Return whether descendants invariant diagnostics should run."""
    return os.environ.get(DESCENDANTS_INVARIANT_ENV_VAR) == "1"


def descendants_invariant_runtime_step_should_validate(step_index: int) -> bool:
    """Return whether one runtime step should perform a full descendants scan."""
    if not descendants_invariant_debug_enabled():
        return False
    return step_index % _descendants_invariant_runtime_step_interval() == 0


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
            first_seen_mismatch = _first_seen_mismatch(
                stored_tag=stored_tag,
                node_tag=node_tag,
                state_tag=state_tag,
                node_depth=node.tree_depth,
                tree_depth=tree_depth,
            )
            if first_seen_mismatch is not None:
                failures.append(
                    _Failure(
                        depth=tree_depth,
                        stored_tag=stored_tag,
                        node_id=node.id,
                        node_tag=node_tag,
                        state_tag=state_tag,
                        node_depth=node.tree_depth,
                        first_seen_mismatch=first_seen_mismatch,
                        registered_tag=_debug_registered_tag(node),
                        registered_state_tag=_debug_registered_state_tag(node),
                        registered_depth=_debug_registered_depth(node),
                        registered_node_id=_debug_registered_node_id(node),
                        registered_parent_node_id=_debug_registered_parent_node_id(
                            node
                        ),
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


def _descendants_invariant_runtime_step_interval() -> int:
    """Return the configured runtime-step full-scan interval."""
    raw_interval = os.environ.get(DESCENDANTS_INVARIANT_STEP_INTERVAL_ENV_VAR)
    if raw_interval is None:
        return DEFAULT_DESCENDANTS_INVARIANT_STEP_INTERVAL
    try:
        interval = int(raw_interval)
    except ValueError:
        return DEFAULT_DESCENDANTS_INVARIANT_STEP_INTERVAL
    return max(interval, 1)


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


def _debug_registered_tag(node: _NodeWithStateTag) -> StateTag | None:
    """Return the insertion-time node tag captured by tree expansion."""
    return getattr(node, "_debug_registered_tag", None)


def _debug_registered_state_tag(node: _NodeWithStateTag) -> StateTag | None:
    """Return the insertion-time state tag captured by tree expansion."""
    return getattr(node, "_debug_registered_state_tag", None)


def _debug_registered_depth(node: _NodeWithStateTag) -> TreeDepth | None:
    """Return the insertion-time depth captured by tree expansion."""
    return getattr(node, "_debug_registered_depth", None)


def _debug_registered_node_id(node: _NodeWithStateTag) -> int | None:
    """Return the insertion-time node id captured by tree expansion."""
    return getattr(node, "_debug_registered_node_id", None)


def _debug_registered_parent_node_id(node: _NodeWithStateTag) -> int | None:
    """Return the insertion-time parent node id captured by tree expansion."""
    return getattr(node, "_debug_registered_parent_node_id", None)


def _first_seen_mismatch(
    *,
    stored_tag: StateTag | None,
    node_tag: StateTag | None,
    state_tag: StateTag | None,
    node_depth: int,
    tree_depth: TreeDepth,
) -> str | None:
    """Return the first mismatch category for one descendants entry."""
    if stored_tag != node_tag:
        return "stored_tag!=node.tag"
    if node_tag != state_tag:
        return "node.tag!=node.state.tag"
    if node_depth != tree_depth:
        return "node.tree_depth!=tree_depth"
    return None


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
                    first_seen_mismatch="duplicate_state_tag",
                    registered_tag=_debug_registered_tag(first_node),
                    registered_state_tag=_debug_registered_state_tag(first_node),
                    registered_depth=_debug_registered_depth(first_node),
                    registered_node_id=_debug_registered_node_id(first_node),
                    registered_parent_node_id=_debug_registered_parent_node_id(
                        first_node
                    ),
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
    first_seen_mismatch = failures[0].first_seen_mismatch if failures else None
    anemone_logger.info(
        "%s phase=%s status=%s node_count=%s depth_count=%s "
        "mismatch_count=%s duplicate_state_tag_groups=%s first_seen_mismatch=%s",
        DESCENDANTS_INVARIANT_LOG_PREFIX,
        phase,
        status,
        node_count,
        depth_count,
        mismatch_count,
        duplicate_state_tag_groups,
        first_seen_mismatch,
    )
    for failure in failures[:limit]:
        anemone_logger.error(
            "%s phase=%s status=failed depth=%s stored_tag=%r node_id=%s "
            "registered_tag=%r registered_state_tag=%r current_node_tag=%r "
            "current_state_tag=%r node_depth=%s registered_depth=%r "
            "registered_node_id=%r registered_parent_node_id=%r "
            "first_seen_mismatch=%s duplicate_peer_node_ids=%s",
            DESCENDANTS_INVARIANT_LOG_PREFIX,
            phase,
            failure.depth,
            failure.stored_tag,
            failure.node_id,
            failure.registered_tag,
            failure.registered_state_tag,
            failure.node_tag,
            failure.state_tag,
            failure.node_depth,
            failure.registered_depth,
            failure.registered_node_id,
            failure.registered_parent_node_id,
            failure.first_seen_mismatch,
            failure.duplicate_peer_node_ids,
        )
