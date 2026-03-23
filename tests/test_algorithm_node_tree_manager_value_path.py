"""Tests for the tree-manager orchestration around value propagation."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from anemone.node_evaluation.direct.node_direct_evaluator import EvaluationQueries
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
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


class _SpyDepthIndexPropagator:
    """Capture the exact seed set given to depth-index propagation."""

    def __init__(self) -> None:
        self.calls: list[list[_FakeNode]] = []

    def propagate_from_changed_nodes(
        self,
        changed_nodes: list[_FakeNode],
    ) -> set[_FakeNode]:
        """Record one propagation request."""
        self.calls.append(list(changed_nodes))
        return set()


class _SpyNodeEvaluator:
    """Capture which created nodes are staged for direct evaluation."""

    def __init__(self) -> None:
        self.add_calls: list[AlgorithmNode[Any]] = []
        self.evaluate_calls: list[list[AlgorithmNode[Any]]] = []

    def add_evaluation_query(
        self,
        node: AlgorithmNode[Any],
        evaluation_queries: EvaluationQueries[Any],
    ) -> None:
        """Record that one node was staged for direct evaluation."""
        self.add_calls.append(node)
        evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_queried_nodes(
        self,
        evaluation_queries: EvaluationQueries[Any],
    ) -> None:
        """Record the current queued batch and clear it."""
        self.evaluate_calls.append(list(evaluation_queries.not_over_nodes))
        evaluation_queries.clear_queries()


def _build_manager(
    *,
    value_propagator: _SpyValuePropagator,
    depth_index_propagator: _SpyDepthIndexPropagator | None = None,
    node_evaluator: _SpyNodeEvaluator | None = None,
) -> AlgorithmNodeTreeManager[Any]:
    """Create a manager with only the dependencies this test needs."""
    return AlgorithmNodeTreeManager(
        tree_manager=SimpleNamespace(dynamics=SimpleNamespace()),
        evaluation_queries=EvaluationQueries(),
        node_evaluator=node_evaluator,
        index_manager=_SpyIndexManager(),
        value_propagator=value_propagator,
        depth_index_propagator=depth_index_propagator,
    )


def _algorithm_node(node_id: int, *, tree_depth: int) -> AlgorithmNode[Any]:
    """Build a minimal real ``AlgorithmNode`` for direct-evaluation tests."""
    tree_node = SimpleNamespace(
        id=node_id,
        tree_depth_=tree_depth,
        state=SimpleNamespace(tag=f"state-{node_id}"),
        tag=f"tag-{node_id}",
        branches_children={},
        parent_nodes={},
        all_branches_generated=False,
        non_opened_branches=set(),
        add_parent=lambda branch_key, new_parent_node: None,
    )
    tree_evaluation = SimpleNamespace(
        direct_value=None,
        is_terminal=lambda: False,
    )
    return AlgorithmNode(
        tree_node=cast("Any", tree_node),
        tree_evaluation=cast("Any", tree_evaluation),
        exploration_index_data=None,
        state_representation=None,
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


def test_evaluate_expansions_only_stages_newly_created_nodes() -> None:
    root = _FakeNode(id=1, tree_depth=0)
    created_child = _algorithm_node(2, tree_depth=1)
    existing_child = _algorithm_node(3, tree_depth=1)

    tree_expansions = TreeExpansions()
    tree_expansions.add_creation(
        TreeExpansion(
            child_node=created_child,
            parent_node=cast("Any", root),
            state_modifications=None,
            creation_child_node=True,
            branch_key=0,
        )
    )
    tree_expansions.add_connection(
        TreeExpansion(
            child_node=existing_child,
            parent_node=cast("Any", root),
            state_modifications=None,
            creation_child_node=False,
            branch_key=1,
        )
    )

    node_evaluator = _SpyNodeEvaluator()
    manager = _build_manager(
        value_propagator=_SpyValuePropagator(),
        node_evaluator=node_evaluator,
    )

    manager.evaluate_expansions(tree_expansions=tree_expansions)

    assert node_evaluator.add_calls == [created_child]
    assert node_evaluator.evaluate_calls == [[created_child]]


def test_propagate_depth_index_routes_expansion_children_through_depth_propagator() -> (
    None
):
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

    depth_index_propagator = _SpyDepthIndexPropagator()
    manager = _build_manager(
        value_propagator=_SpyValuePropagator(),
        depth_index_propagator=depth_index_propagator,
    )

    manager.propagate_depth_index(tree_expansions=tree_expansions)

    assert depth_index_propagator.calls == [[created_child, existing_child]]


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
