"""Regression tests for principal-variation versioning behavior."""

# ruff: noqa: D103
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from tests.fakes_tree_evaluation import set_estimate_value


@dataclass
class _FakeChildNode:
    node_id: int
    tree_evaluation: NodeMinmaxEvaluation[Any, Any]

    @property
    def tree_node(self) -> Any:
        return self.tree_evaluation.tree_node


def _make_leaf_eval(
    *,
    turn: Color,
    value_white: float,
    pv_tail: list[int],
    node_id: int,
) -> NodeMinmaxEvaluation[Any, Any]:
    leaf_tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=turn),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=leaf_tree_node)
    set_estimate_value(ev, score=value_white)
    ev.set_best_branch_sequence(pv_tail[:])
    return ev


def _make_parent_eval() -> tuple[
    NodeMinmaxEvaluation[Any, Any], dict[int, _FakeChildNode]
]:
    children = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children=children,
        all_branches_generated=True,
    )
    parent = NodeMinmaxEvaluation(
        tree_node=parent_tree_node,
        backup_policy=ExplicitMinimaxBackupPolicy(),
    )
    set_estimate_value(parent, score=0.0)
    parent.update_branches_values(branches_to_consider=set(children.keys()))
    return parent, children


def test_pv_version_increments_only_on_real_change() -> None:
    parent, _ = _make_parent_eval()

    first = parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    version_after_first = parent.pv_version
    pv_after_first = parent.best_branch_sequence.copy()

    second = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq=set(),
    )

    assert first.pv_changed
    assert not second.pv_changed
    assert parent.pv_version == version_after_first
    assert parent.best_branch_sequence == pv_after_first


def test_parent_pv_rebuilds_when_best_child_pv_version_changes() -> None:
    parent, children = _make_parent_eval()
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    version_before = parent.pv_version
    children[0].tree_evaluation.set_best_branch_sequence([99])

    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={0},
    )

    assert result.pv_changed
    assert parent.best_branch_sequence == [0, 99]
    assert parent.pv_version == version_before + 1


def test_no_pv_rebuild_for_non_best_child_pv_changes() -> None:
    parent, children = _make_parent_eval()
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    version_before = parent.pv_version
    pv_before = parent.best_branch_sequence.copy()

    children[1].tree_evaluation.set_best_branch_sequence([77])
    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={1},
    )

    assert not result.pv_changed
    assert parent.best_branch_sequence == pv_before
    assert parent.pv_version == version_before


def test_parent_pv_rebuilds_from_empty_parent_pv_on_best_child_notification() -> None:
    parent, children = _make_parent_eval()
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    parent.clear_best_branch_sequence()
    children[0].tree_evaluation.set_best_branch_sequence([88])

    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={0},
    )

    assert result.pv_changed
    assert parent.best_branch_sequence == [0, 88]


def test_update_best_branch_sequence_does_not_fix_corrupted_head() -> None:
    parent, _ = _make_parent_eval()
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    parent.set_best_branch_sequence([1, 5])
    version_before = parent.pv_version
    pv_before = parent.best_branch_sequence.copy()

    changed = parent.update_best_branch_sequence({0})

    assert not changed
    assert parent.best_branch_sequence == pv_before
    assert parent.pv_version == version_before


def test_update_best_branch_sequence_requires_notification() -> None:
    parent, children = _make_parent_eval()
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    version_before = parent.pv_version
    pv_before = parent.best_branch_sequence.copy()

    children[0].tree_evaluation.set_best_branch_sequence([42])
    changed = parent.update_best_branch_sequence(set())

    assert not changed
    assert parent.best_branch_sequence == pv_before
    assert parent.pv_version == version_before


def test_partial_expansion_pv_invariant_helper() -> None:
    parent, _ = _make_parent_eval()
    parent.tree_node.all_branches_generated = False
    parent.tree_node.state.turn = Color.WHITE
    set_estimate_value(parent, score=1e9)
    parent.update_branches_values(branches_to_consider={0, 1})

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 0
    assert parent.best_branch_sequence == []
    parent.assert_pv_invariants()


def test_partial_expansion_pv_invariant_helper_allows_non_empty_pv() -> None:
    parent, _ = _make_parent_eval()
    parent.tree_node.all_branches_generated = False
    parent.tree_node.state.turn = Color.WHITE
    set_estimate_value(parent, score=-1e9)
    parent.update_branches_values(branches_to_consider={0, 1})

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() is not None
    assert parent.best_branch_sequence
    parent.assert_pv_invariants()


def test_partial_expansion_pv_invariant_helper_black_disallows_pv() -> None:
    parent, _ = _make_parent_eval()
    parent.tree_node.all_branches_generated = False
    parent.tree_node.state.turn = Color.BLACK
    set_estimate_value(parent, score=-1e9)
    parent.update_branches_values(branches_to_consider={0, 1})

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() is not None
    assert parent.best_branch_sequence == []
    parent.assert_pv_invariants()


def test_partial_expansion_pv_invariant_helper_black_allows_non_empty_pv() -> None:
    parent, _ = _make_parent_eval()
    parent.tree_node.all_branches_generated = False
    parent.tree_node.state.turn = Color.BLACK
    set_estimate_value(parent, score=1e9)
    parent.update_branches_values(branches_to_consider={0, 1})

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() is not None
    assert parent.best_branch_sequence
    parent.assert_pv_invariants()
