"""Compatibility adapters for rollout node status and report fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .report import RolloutStopReason

if TYPE_CHECKING:
    from anemone import nodes as node
    from anemone import trees


def rollout_node_id(node_to_check: node.ITreeNode[Any]) -> str | None:
    """Return a stable string node id when exposed by the node."""
    node_id = getattr(node_to_check, "id", None)
    return None if node_id is None else str(node_id)


def rollout_node_depth(
    *,
    tree: trees.Tree[node.ITreeNode[Any]],
    node_to_check: node.ITreeNode[Any],
) -> int | None:
    """Return the node depth through the tree API when available."""
    node_depth = getattr(tree, "node_depth", None)
    if callable(node_depth):
        depth = node_depth(node_to_check)
        return depth if isinstance(depth, int) else None
    depth = getattr(node_to_check, "tree_depth", None)
    return depth if isinstance(depth, int) else None


def report_terminal_status(node_to_check: node.ITreeNode[Any]) -> bool | None:
    """Return terminal status for reporting from node/state/evaluation adapters."""
    known_false = False

    is_over = getattr(node_to_check, "is_over", None)
    if callable(is_over):
        if bool(is_over()):
            return True
        known_false = True

    state = getattr(node_to_check, "state", None)
    is_game_over = getattr(state, "is_game_over", None)
    if callable(is_game_over):
        if bool(is_game_over()):
            return True
        known_false = True

    state_is_terminal = getattr(state, "is_terminal", None)
    if isinstance(state_is_terminal, bool):
        if state_is_terminal:
            return True
        known_false = True

    tree_evaluation = getattr(node_to_check, "tree_evaluation", None)
    is_terminal = getattr(tree_evaluation, "is_terminal", None)
    if callable(is_terminal):
        if bool(is_terminal()):
            return True
        known_false = True

    return False if known_false else None


def rollout_stop_terminal_status(node_to_check: node.ITreeNode[Any]) -> bool | None:
    """Return terminal status using only rollout-stop compatibility surfaces."""
    is_over = getattr(node_to_check, "is_over", None)
    if callable(is_over):
        return bool(is_over())

    tree_evaluation = getattr(node_to_check, "tree_evaluation", None)
    is_terminal = getattr(tree_evaluation, "is_terminal", None)
    if callable(is_terminal):
        return bool(is_terminal())

    return None


def exact_value_status(node_to_check: node.ITreeNode[Any]) -> bool | None:
    """Return whether the node exposes an exact value status cheaply."""
    tree_evaluation = getattr(node_to_check, "tree_evaluation", None)
    has_exact_value = getattr(tree_evaluation, "has_exact_value", None)
    if callable(has_exact_value):
        return bool(has_exact_value())

    node_has_exact_value = getattr(node_to_check, "has_exact_value", None)
    if callable(node_has_exact_value):
        return bool(node_has_exact_value())

    exact = getattr(node_to_check, "exact", None)
    if isinstance(exact, bool):
        return exact

    return None


def non_opened_branch_count(node_to_check: node.ITreeNode[Any]) -> int | None:
    """Return the node's stored non-opened branch count when available."""
    return node_to_check.unopened_branch_count()


def inverse_optional_bool(value: bool | None) -> bool | None:
    """Return the inverse of an optional boolean."""
    return None if value is None else not value


def no_legal_actions_but_not_terminal(
    *,
    stop_reason: RolloutStopReason,
    end_is_terminal: bool | None,
) -> bool | None:
    """Return whether a no-legal-actions stop was not tagged terminal."""
    if stop_reason is not RolloutStopReason.NO_LEGAL_ACTIONS:
        return False
    if end_is_terminal is None:
        return None
    return not end_is_terminal
