"""Focused tests for ``ValuePropagator`` scheduling semantics."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.types import BackupResult
from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.updates.value_propagator import ValuePropagator


@dataclass(eq=False)
class _FakeEvaluation:
    """Minimal backup surface for testing propagation orchestration."""

    results: list[BackupResult] = field(default_factory=list)
    calls: list[tuple[set[int], set[int]]] = field(default_factory=list)

    def backup_from_children(
        self,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult:
        """Record one backup call and return the next configured result."""
        self.calls.append(
            (
                set(branches_with_updated_value),
                set(branches_with_updated_best_branch_seq),
            )
        )
        if self.results:
            return self.results.pop(0)
        return BackupResult(value_changed=False, pv_changed=False, over_changed=False)


@dataclass(eq=False)
class _FakeNode:
    """Tiny node stub with just the fields the propagator needs."""

    name: str
    tree_depth: int
    branches_children: dict[int, "_FakeNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_FakeNode", int] = field(default_factory=dict)
    tree_evaluation: _FakeEvaluation = field(default_factory=_FakeEvaluation)

    def __repr__(self) -> str:
        """Return a stable name to make assertion failures readable."""
        return self.name


def _connect(parent: _FakeNode, branch: int, child: _FakeNode) -> None:
    """Connect two fake nodes through one branch."""
    parent.branches_children[branch] = child
    child.parent_nodes[parent] = branch


def _as_algorithm_node(node: _FakeNode) -> AlgorithmNode[Any]:
    """Cast a fake node to the production node type used by the propagator."""
    return cast("AlgorithmNode[Any]", node)


def test_single_changed_child_recomputes_parent() -> None:
    leaf = _FakeNode(name="leaf", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    _connect(parent, branch=0, child=leaf)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        recompute_calls.append(cast("_FakeNode", node).name)
        return False

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {parent}
    assert recompute_calls == ["parent"]


def test_two_changed_siblings_recompute_parent_once_before_grandparent() -> None:
    left = _FakeNode(name="left", tree_depth=2)
    right = _FakeNode(name="right", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    grandparent = _FakeNode(name="grandparent", tree_depth=0)
    _connect(parent, branch=0, child=left)
    _connect(parent, branch=1, child=right)
    _connect(grandparent, branch=7, child=parent)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        fake_node = cast("_FakeNode", node)
        recompute_calls.append(fake_node.name)
        return fake_node is parent

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(left), _as_algorithm_node(right)]
    )

    assert affected == {parent, grandparent}
    assert recompute_calls == ["parent", "grandparent"]


def test_changed_child_with_two_parents_schedules_both_parents() -> None:
    shared_child = _FakeNode(name="shared_child", tree_depth=2)
    parent_a = _FakeNode(name="parent_a", tree_depth=1)
    parent_b = _FakeNode(name="parent_b", tree_depth=1)
    _connect(parent_a, branch=0, child=shared_child)
    _connect(parent_b, branch=1, child=shared_child)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        recompute_calls.append(cast("_FakeNode", node).name)
        return False

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(shared_child)]
    )

    assert affected == {parent_a, parent_b}
    assert set(recompute_calls) == {"parent_a", "parent_b"}
    assert len(recompute_calls) == 2


def test_depth_waves_process_deeper_dirty_nodes_before_shallower_ancestors() -> None:
    leaf = _FakeNode(name="leaf", tree_depth=3)
    lower_parent = _FakeNode(name="lower_parent", tree_depth=2)
    upper_parent = _FakeNode(name="upper_parent", tree_depth=1)
    root = _FakeNode(name="root", tree_depth=0)
    _connect(lower_parent, branch=0, child=leaf)
    _connect(upper_parent, branch=1, child=lower_parent)
    _connect(root, branch=2, child=upper_parent)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        fake_node = cast("_FakeNode", node)
        recompute_calls.append(fake_node.name)
        return fake_node in {lower_parent, upper_parent}

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {lower_parent, upper_parent, root}
    assert recompute_calls == ["lower_parent", "upper_parent", "root"]


def test_default_recompute_uses_full_current_child_snapshot() -> None:
    changed_child = _FakeNode(name="changed_child", tree_depth=2)
    sibling = _FakeNode(name="sibling", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    parent.tree_evaluation = _FakeEvaluation(
        results=[BackupResult(value_changed=True, pv_changed=False, over_changed=False)]
    )
    _connect(parent, branch=0, child=changed_child)
    _connect(parent, branch=1, child=sibling)

    propagator = ValuePropagator()
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(changed_child)]
    )

    assert affected == {parent}
    assert parent.tree_evaluation.calls == [({0, 1}, {0, 1})]


@dataclass(eq=False)
class _SemanticNode:
    """Minimal graph node carrying a real minimax evaluation for semantic tests."""

    name: str
    node_id: int
    tree_depth: int
    turn: Color
    branches_children: dict[int, "_SemanticNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_SemanticNode", int] = field(default_factory=dict)
    tree_evaluation: NodeMinmaxEvaluation[Any, Any] = field(init=False)

    def __post_init__(self) -> None:
        tree_node = SimpleNamespace(
            id=self.node_id,
            state=SimpleNamespace(turn=self.turn),
            tree_depth_=self.tree_depth,
            branches_children=self.branches_children,
            parent_nodes=self.parent_nodes,
            all_branches_generated=True,
        )
        self.tree_evaluation = NodeMinmaxEvaluation(
            tree_node=tree_node,
            backup_policy=ExplicitMinimaxBackupPolicy(),
        )
        default_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
        self.tree_evaluation.direct_value = default_value
        self.tree_evaluation.minmax_value = default_value

    @property
    def id(self) -> int:
        return self.node_id

    def __repr__(self) -> str:
        return self.name


def _connect_semantic(parent: _SemanticNode, branch: int, child: _SemanticNode) -> None:
    parent.branches_children[branch] = child
    child.parent_nodes[parent] = branch


def _set_semantic_value(
    node: _SemanticNode,
    *,
    score: float,
    pv_tail: list[int] | None = None,
) -> None:
    value = Value(score=score, certainty=Certainty.ESTIMATE)
    node.tree_evaluation.direct_value = value
    node.tree_evaluation.minmax_value = value
    node.tree_evaluation.set_best_branch_sequence(list(pv_tail or []))


def test_shared_child_semantic_propagation_can_change_one_parent_but_not_the_other() -> (
    None
):
    shared_child = _SemanticNode(
        name="shared_child",
        node_id=3,
        tree_depth=2,
        turn=Color.WHITE,
    )
    parent_a = _SemanticNode(
        name="parent_a",
        node_id=1,
        tree_depth=1,
        turn=Color.WHITE,
    )
    parent_b = _SemanticNode(
        name="parent_b",
        node_id=2,
        tree_depth=1,
        turn=Color.BLACK,
    )
    other_a = _SemanticNode(
        name="other_a",
        node_id=4,
        tree_depth=2,
        turn=Color.WHITE,
    )
    other_b = _SemanticNode(
        name="other_b",
        node_id=5,
        tree_depth=2,
        turn=Color.WHITE,
    )

    _connect_semantic(parent_a, 0, shared_child)
    _connect_semantic(parent_a, 1, other_a)
    _connect_semantic(parent_b, 0, shared_child)
    _connect_semantic(parent_b, 1, other_b)

    _set_semantic_value(shared_child, score=0.4, pv_tail=[40])
    _set_semantic_value(other_a, score=0.5, pv_tail=[50])
    _set_semantic_value(other_b, score=0.2, pv_tail=[20])

    parent_a.tree_evaluation.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq={0, 1},
    )
    parent_b.tree_evaluation.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq={0, 1},
    )
    assert parent_a.tree_evaluation.best_branch() == 1
    assert parent_a.tree_evaluation.best_branch_sequence == [1, 50]
    assert parent_b.tree_evaluation.best_branch() == 1
    assert parent_b.tree_evaluation.best_branch_sequence == [1, 20]

    # The shared child improves enough to become best for the white parent,
    # but the black parent still prefers its lower-scoring alternative.
    _set_semantic_value(shared_child, score=0.6, pv_tail=[60, 61])
    propagator = ValuePropagator()
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(shared_child)]
    )

    assert affected == {parent_a, parent_b}
    assert parent_a.tree_evaluation.best_branch() == 0
    assert parent_a.tree_evaluation.best_branch_sequence == [0, 60, 61]
    assert parent_a.tree_evaluation.minmax_value == Value(
        score=0.6,
        certainty=Certainty.ESTIMATE,
    )
    assert parent_b.tree_evaluation.best_branch() == 1
    assert parent_b.tree_evaluation.best_branch_sequence == [1, 20]
    assert parent_b.tree_evaluation.minmax_value == Value(
        score=0.2,
        certainty=Certainty.ESTIMATE,
    )
