"""Best-effort state snapshot helpers for observer-based debug events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeGuard

from anemone.debug.formatting import (
    format_branch_sequence,
    format_over_event,
    format_value,
    safe_getattr,
)


@dataclass(frozen=True, slots=True)
class NodeEvaluationSummary:
    """Compact best-effort representation of node evaluation state."""

    direct_value_repr: str | None
    backed_up_value_repr: str | None
    pv_repr: str | None
    over_repr: str | None


def summarize_node_evaluation(node: Any) -> NodeEvaluationSummary:
    """Summarize the debug-relevant evaluation state of ``node``."""
    evaluation = safe_getattr(node, "tree_evaluation")
    if evaluation is None:
        evaluation = node

    direct_value = safe_getattr(evaluation, "direct_value")
    backed_up_value = safe_getattr(evaluation, "backed_up_value")
    if backed_up_value is None:
        backed_up_value = safe_getattr(evaluation, "minmax_value")

    best_branch_sequence = safe_getattr(evaluation, "best_branch_sequence")
    over_event = safe_getattr(evaluation, "over_event")
    if over_event is None:
        over_event = _safe_get_over_event_candidate(evaluation)

    return NodeEvaluationSummary(
        direct_value_repr=_format_optional_value(direct_value),
        backed_up_value_repr=_format_optional_value(backed_up_value),
        pv_repr=(
            None
            if not best_branch_sequence
            else format_branch_sequence(best_branch_sequence)
        ),
        over_repr=None if over_event is None else format_over_event(over_event),
    )


def snapshot_children(node: Any) -> dict[str, str]:
    """Return the currently linked children as ``branch_key_str -> child_id_str``."""
    branches_children = safe_getattr(node, "branches_children")
    if branches_children is None:
        return {}

    snapshot: dict[str, str] = {}
    for branch_key, child in branches_children.items():
        if child is None:
            continue
        snapshot[str(branch_key)] = str(child.id)

    return snapshot


def diff_new_children(
    before: dict[str, str],
    after: dict[str, str],
) -> list[tuple[str, str]]:
    """Return new or changed branch-to-child links introduced in ``after``."""
    changes: list[tuple[str, str]] = []
    for branch_key, child_id in after.items():
        if before.get(branch_key) != child_id:
            changes.append((branch_key, child_id))

    return sorted(changes)


def collect_unique_nodes_from_opening_instructions(
    opening_instructions: Any,
) -> tuple[Any, ...]:
    """Return unique nodes referenced by an opening-instructions-like object."""
    nodes: list[Any] = []
    seen_nodes: set[int] = set()

    for opening_instruction in collect_opening_instructions(opening_instructions):
        node = safe_getattr(opening_instruction, "node_to_open")
        if node is None:
            continue
        _add_unique_node(node, nodes, seen_nodes)

    return tuple(nodes)


def collect_opening_instructions(opening_instructions: Any) -> tuple[Any, ...]:
    """Return opening instructions from common container shapes in stable order."""
    values = safe_getattr(opening_instructions, "values")
    if callable(values):
        return _iterable_to_any_tuple(values())

    return _iterable_to_any_tuple(opening_instructions)


def collect_nodes_from_tree_expansions(tree_expansions: Any) -> tuple[Any, ...]:
    """Return unique child and parent nodes referenced by ``tree_expansions``."""
    nodes: list[Any] = []
    seen_nodes: set[int] = set()

    expansions = _iterable_to_any_tuple(tree_expansions)

    for tree_expansion in expansions:
        for attribute_name in ("child_node", "parent_node"):
            node = safe_getattr(tree_expansion, attribute_name)
            if node is None:
                continue
            _add_unique_node(node, nodes, seen_nodes)

    return tuple(nodes)


def collect_nodes_and_ancestors(nodes: Iterable[Any]) -> tuple[Any, ...]:
    """Return ``nodes`` plus their reachable ancestors in stable discovery order."""
    observed_nodes: list[Any] = []
    seen_nodes: set[int] = set()
    frontier = list(nodes)

    while frontier:
        node = frontier.pop(0)
        if node is None:
            continue
        node_identity = id(node)
        if node_identity in seen_nodes:
            continue

        seen_nodes.add(node_identity)
        observed_nodes.append(node)

        parent_nodes = safe_getattr(node, "parent_nodes")
        if parent_nodes is None:
            continue

        frontier.extend(parent_nodes.keys())

    return tuple(observed_nodes)


def _safe_get_over_event_candidate(evaluation: Any) -> Any | None:
    """Call ``get_over_event_candidate`` when present."""
    getter = safe_getattr(evaluation, "get_over_event_candidate")
    if not callable(getter):
        return None

    try:
        return getter()
    except (AttributeError, TypeError):
        return None


def _format_optional_value(value: Any | None) -> str | None:
    """Format ``value`` when present."""
    return None if value is None else format_value(value)


def _iterable_to_any_tuple(value: object) -> tuple[Any, ...]:
    """Return ``value`` as ``tuple[Any, ...]`` when it is iterable."""
    if not _is_any_iterable(value):
        return ()

    return tuple(value)


def _is_any_iterable(value: object) -> TypeGuard[Iterable[Any]]:
    """Return whether ``value`` can be treated as ``Iterable[Any]``."""
    return isinstance(value, Iterable)


def _add_unique_node(node: Any, nodes: list[Any], seen_nodes: set[int]) -> None:
    """Append ``node`` only once based on object identity."""
    node_identity = id(node)
    if node_identity in seen_nodes:
        return

    seen_nodes.add(node_identity)
    nodes.append(node)
