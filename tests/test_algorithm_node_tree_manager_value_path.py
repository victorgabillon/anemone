"""Tests for the tree-manager orchestration around value propagation."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from anemone.tree_manager.algorithm_node_tree_manager import (
    AlgorithmNodeTreeManager,
)
from anemone.tree_manager.tree_expander import TreeExpansion, TreeExpansions


@dataclass(eq=False)
class _FakeNode:
    """Tiny node stub for exercising tree-manager orchestration."""

    id: int
    tree_depth: int
    state: Any = field(default_factory=SimpleNamespace)
    tree_evaluation: Any = field(default_factory=SimpleNamespace)
    branches_children: dict[int, "_FakeNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_FakeNode", int] = field(default_factory=dict)


class _SpyValuePropagator:
    """Capture the exact seed set given to value propagation."""

    def __init__(self) -> None:
        self.calls: list[list[_FakeNode]] = []

    def propagate_from_changed_nodes(
        self,
        changed_nodes: list[_FakeNode],
    ) -> set[_FakeNode]:
        """Record one propagation request."""
        self.calls.append(list(changed_nodes))
        return set()


class _SpyIndexManager:
    """Marker object used to assert index refresh is separate."""


def _build_manager(
    *,
    value_propagator: _SpyValuePropagator,
) -> AlgorithmNodeTreeManager[Any]:
    """Create a manager with only the dependencies this test needs."""
    return AlgorithmNodeTreeManager(
        tree_manager=SimpleNamespace(dynamics=SimpleNamespace()),
        algorithm_tree_node_factory=SimpleNamespace(),
        algorithm_node_updater=SimpleNamespace(),
        evaluation_queries=SimpleNamespace(),
        node_evaluator=None,
        index_manager=_SpyIndexManager(),
        value_propagator=value_propagator,
    )


def test_update_backward_routes_expansion_children_through_value_propagator() -> None:
    root = _FakeNode(id=1, tree_depth=0)
    created_child = _FakeNode(id=2, tree_depth=1)
    existing_child = _FakeNode(id=3, tree_depth=1)

    tree_expansions = TreeExpansions()
    tree_expansions.add_creation(
        TreeExpansion(
            child_node=created_child,
            parent_node=root,
            state_modifications=None,
            creation_child_node=True,
            branch_key=0,
        )
    )
    tree_expansions.add_connection(
        TreeExpansion(
            child_node=existing_child,
            parent_node=root,
            state_modifications=None,
            creation_child_node=False,
            branch_key=1,
        )
    )

    value_propagator = _SpyValuePropagator()
    manager = _build_manager(value_propagator=value_propagator)

    manager.update_backward(tree_expansions=tree_expansions)

    assert value_propagator.calls == [[created_child, existing_child]]


def test_refresh_exploration_indices_remains_a_separate_explicit_step(
    monkeypatch: Any,
) -> None:
    value_propagator = _SpyValuePropagator()
    manager = _build_manager(value_propagator=value_propagator)
    recorded_calls: list[tuple[object, object]] = []

    def fake_update_all_indices(*, index_manager: object, tree: object) -> None:
        recorded_calls.append((index_manager, tree))

    monkeypatch.setattr(
        "anemone.tree_manager.algorithm_node_tree_manager.update_all_indices",
        fake_update_all_indices,
    )

    manager.update_backward(tree_expansions=TreeExpansions())
    assert recorded_calls == []

    tree = object()
    manager.refresh_exploration_indices(tree=tree)

    assert recorded_calls == [(manager.index_manager, tree)]
