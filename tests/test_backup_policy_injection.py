"""Tests proving injected backup policy semantics remain on the active path."""

from dataclasses import dataclass, field
from typing import Any, cast

from anemone.backup_policies.types import BackupResult
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.updates.value_propagator import ValuePropagator


@dataclass(eq=False)
class _FakePolicy:
    """Capture full-snapshot backup calls made through one parent recomputation."""

    calls: list[tuple[set[int], set[int]]] = field(default_factory=list)
    result: BackupResult[int] = field(
        default_factory=lambda: BackupResult(
            value_changed=True,
            pv_changed=True,
            over_changed=False,
        )
    )

    def backup_from_children(
        self,
        node_eval: object,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult[int]:
        """Record one delegated backup call from the active propagation path."""
        del node_eval
        self.calls.append(
            (
                set(branches_with_updated_value),
                set(branches_with_updated_best_branch_seq),
            )
        )
        return self.result


@dataclass(eq=False)
class _FakeTreeEvaluation:
    """Minimal evaluation object delegating backup work to an injected policy."""

    backup_policy: _FakePolicy

    def backup_from_children(
        self,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult[int]:
        """Forward recomputation to the injected policy."""
        return self.backup_policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )


@dataclass(eq=False)
class _FakeNode:
    """Tiny node stub with the fields ``ValuePropagator`` needs."""

    tree_depth: int
    tree_evaluation: _FakeTreeEvaluation
    branches_children: dict[int, "_FakeNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_FakeNode", set[int]] = field(default_factory=dict)


def _connect(parent: _FakeNode, branch: int, child: _FakeNode) -> None:
    """Connect one parent/child pair through a branch."""
    parent.branches_children[branch] = child
    child.parent_nodes[parent] = {branch}


def _as_algorithm_node(node: _FakeNode) -> AlgorithmNode[Any]:
    """Cast a fake node into the production node protocol the propagator uses."""
    return cast("AlgorithmNode[Any]", node)


def test_value_propagator_uses_injected_backup_policy_for_parent_recompute() -> None:
    """Active value propagation must still delegate to the node's backup policy."""
    policy = _FakePolicy()
    parent = _FakeNode(tree_depth=1, tree_evaluation=_FakeTreeEvaluation(policy))
    changed_child = _FakeNode(
        tree_depth=2,
        tree_evaluation=_FakeTreeEvaluation(_FakePolicy()),
    )
    sibling = _FakeNode(
        tree_depth=2,
        tree_evaluation=_FakeTreeEvaluation(_FakePolicy()),
    )
    _connect(parent, branch=0, child=changed_child)
    _connect(parent, branch=1, child=sibling)

    propagator = ValuePropagator()
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(changed_child)]
    )

    assert affected == {parent}
    assert changed_child.parent_nodes[parent] == {0}
    assert policy.calls == [({0, 1}, {0, 1})]
