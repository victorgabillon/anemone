"""Tests for rollout node-status compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from anemone.rollouts.node_status import (
    exact_value_status,
    non_opened_branch_count,
    report_terminal_status,
    rollout_node_depth,
    rollout_stop_terminal_status,
)


@dataclass
class _Node:
    """Tiny node shape for adapter tests."""

    id: int = 1
    tree_depth: int = 7
    state: object | None = None
    tree_evaluation: object | None = None
    exact: bool | None = None
    unopened_count: int = 0

    def unopened_branch_count(self) -> int:
        """Return configured unopened branch count."""
        return self.unopened_count


@dataclass
class _IsOverNode(_Node):
    """Node shape exposing the rollout stop ``is_over`` surface."""

    is_over_value: bool = False

    def is_over(self) -> bool:
        """Return configured terminality."""
        return self.is_over_value


@dataclass
class _ExactNode(_Node):
    """Node shape exposing node-level exactness."""

    has_exact_value_value: bool = False

    def has_exact_value(self) -> bool:
        """Return configured exactness."""
        return self.has_exact_value_value


def _as_node(value: object) -> Any:
    """Cast local fakes to the adapter node input type."""
    return cast("Any", value)


def test_rollout_stop_terminal_status_prefers_node_is_over() -> None:
    """Rollout stop status returns immediately from node ``is_over``."""
    node = _IsOverNode(
        is_over_value=False,
        tree_evaluation=SimpleNamespace(is_terminal=lambda: True),
    )

    assert rollout_stop_terminal_status(_as_node(node)) is False


def test_rollout_stop_terminal_status_falls_back_to_tree_evaluation() -> None:
    """Rollout stop status falls back to tree-evaluation terminality."""
    node = _Node(tree_evaluation=SimpleNamespace(is_terminal=lambda: True))

    assert rollout_stop_terminal_status(_as_node(node)) is True


def test_report_terminal_status_returns_true_from_any_surface() -> None:
    """Report terminal status returns true when any report surface is terminal."""
    state = SimpleNamespace(is_game_over=lambda: True)
    node = _IsOverNode(is_over_value=False, state=state)

    assert report_terminal_status(_as_node(node)) is True


def test_report_terminal_status_returns_false_from_known_false_surfaces() -> None:
    """Report terminal status returns false when at least one surface says false."""
    state = SimpleNamespace(is_terminal=False)
    tree_evaluation = SimpleNamespace(is_terminal=lambda: False)
    node = _IsOverNode(
        is_over_value=False,
        state=state,
        tree_evaluation=tree_evaluation,
    )

    assert report_terminal_status(_as_node(node)) is False


def test_report_terminal_status_returns_none_without_known_surface() -> None:
    """Report terminal status returns none when no terminal surface is exposed."""
    assert report_terminal_status(_as_node(SimpleNamespace())) is None


def test_exact_value_status_prefers_tree_evaluation() -> None:
    """Exact status returns tree-evaluation exactness before node exactness."""
    node = _ExactNode(
        has_exact_value_value=True,
        tree_evaluation=SimpleNamespace(has_exact_value=lambda: False),
    )

    assert exact_value_status(_as_node(node)) is False


def test_exact_value_status_falls_back_to_exact_bool() -> None:
    """Exact status falls back to the legacy boolean ``exact`` field."""
    node = _Node(exact=True)

    assert exact_value_status(_as_node(node)) is True


def test_rollout_node_depth_prefers_tree_depth_api() -> None:
    """Rollout node depth uses tree ``node_depth`` when it returns an int."""
    tree = SimpleNamespace(node_depth=lambda node: node.tree_depth - 5)
    node = _Node(tree_depth=9)

    assert rollout_node_depth(tree=cast("Any", tree), node_to_check=_as_node(node)) == 4


def test_rollout_node_depth_falls_back_to_node_tree_depth() -> None:
    """Rollout node depth falls back to node ``tree_depth``."""
    node = _Node(tree_depth=9)

    assert (
        rollout_node_depth(
            tree=cast("Any", SimpleNamespace()), node_to_check=_as_node(node)
        )
        == 9
    )


def test_non_opened_branch_count_delegates_to_node() -> None:
    """Non-opened branch count delegates to the node API."""
    node = _Node(unopened_count=3)

    assert non_opened_branch_count(_as_node(node)) == 3
