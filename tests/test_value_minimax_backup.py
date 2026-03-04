"""Tests for Value-driven minimax backup in explicit policy."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.backup_policies.explicit_minimax import (
    ExplicitMinimaxBackupPolicy,
    has_value_changed,
)
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.values import DEFAULT_EVALUATION_ORDERING, Certainty, Value


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_winner(self, player: Color) -> bool:
        return self.winner == player


def _make_leaf(node_id: int, value: Value) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=tree_node)
    ev.direct_value = value
    ev.minmax_value = value
    ev.sync_float_views_from_values()
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=ev)


def _make_unvalued_leaf(node_id: int) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=tree_node)
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=ev)


def _make_parent(
    *,
    turn: Color,
    children: dict[int, Any],
    all_generated: bool,
    direct_value: Value,
) -> NodeMinmaxEvaluation[Any, Any]:
    tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=all_generated,
    )
    parent = NodeMinmaxEvaluation(
        tree_node=tree_node,
        backup_policy=ExplicitMinimaxBackupPolicy(),
    )
    parent.direct_value = direct_value
    parent.minmax_value = direct_value
    parent.sync_float_views_from_values()
    return parent


def test_estimate_children_respect_side_to_move_ordering() -> None:
    white_parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=0.2)),
            1: _make_leaf(2, Value(score=0.7)),
        },
        all_generated=True,
        direct_value=Value(score=0.0),
    )
    white_parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert white_parent.minmax_value == Value(score=0.7)
    assert white_parent.value_white_minmax == 0.7

    black_parent = _make_parent(
        turn=Color.BLACK,
        children={
            0: _make_leaf(1, Value(score=0.2)),
            1: _make_leaf(2, Value(score=0.7)),
        },
        all_generated=True,
        direct_value=Value(score=0.0),
    )
    black_parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert black_parent.minmax_value == Value(score=0.2)
    assert black_parent.value_white_minmax == 0.2


def test_partial_expansion_prefers_direct_when_better_for_side_to_move() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.2))},
        all_generated=False,
        direct_value=Value(score=0.5),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.5)
    assert parent.value_white_minmax == 0.5


def test_partial_expansion_prefers_child_for_black_when_child_is_better() -> None:
    parent = _make_parent(
        turn=Color.BLACK,
        children={0: _make_leaf(1, Value(score=0.2))},
        all_generated=False,
        direct_value=Value(score=0.5),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.2)
    assert parent.value_white_minmax == 0.2


def test_partial_expansion_prefers_direct_for_black_when_direct_is_better() -> None:
    parent = _make_parent(
        turn=Color.BLACK,
        children={0: _make_leaf(1, Value(score=0.7))},
        all_generated=False,
        direct_value=Value(score=0.5),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.5)
    assert parent.value_white_minmax == 0.5


def test_semantic_compare_terminal_vs_estimate_is_exposed() -> None:
    forced_win = Value(
        score=-10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    estimate = Value(score=999.0)
    assert (
        DEFAULT_EVALUATION_ORDERING.semantic_compare(
            forced_win,
            estimate,
            side_to_move=Color.WHITE,
        )
        > 0
    )


def test_search_ordering_remains_projection_based_for_large_estimate() -> None:
    forced_win = Value(
        score=-10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    very_high_estimate = Value(score=10.0)

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, forced_win), 1: _make_leaf(2, very_high_estimate)},
        all_generated=True,
        direct_value=Value(score=0.0),
    )
    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 1
    assert parent.minmax_value == very_high_estimate
    assert (
        parent.branches_sorted_by_value_[1][0] < parent.branches_sorted_by_value_[0][0]
    )


def test_backup_result_value_changed_tracks_score_certainty_and_over_event() -> None:
    assert has_value_changed(
        value_before=Value(score=0.1),
        value_after=Value(score=0.2),
    )
    assert has_value_changed(
        value_before=Value(score=0.1, certainty=Certainty.ESTIMATE),
        value_after=Value(score=0.1, certainty=Certainty.FORCED),
    )
    assert has_value_changed(
        value_before=Value(score=0.1, over_event=None),
        value_after=Value(score=0.1, over_event=_FakeOverEvent(draw=True)),
    )


def test_direct_eval_change_without_minmax_change_reports_no_value_change() -> None:
    child_value = Value(score=0.7)
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, child_value)},
        all_generated=False,
        direct_value=Value(score=0.5),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )
    parent.direct_value = Value(score=0.6)
    parent.sync_float_views_from_values()

    result = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == child_value
    assert not result.value_changed


def test_partial_expansion_pv_does_not_depend_on_float_field() -> None:
    child_value = Value(score=0.9)
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, child_value)},
        all_generated=False,
        direct_value=Value(score=0.8),
    )
    parent.set_best_branch_sequence([0])

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == child_value
    assert parent.best_branch_sequence
    assert parent.best_branch_sequence[0] == 0


def test_partial_expansion_unvalued_best_child_clears_pv() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_unvalued_leaf(1)},
        all_generated=False,
        direct_value=Value(score=0.8),
    )
    parent.set_best_branch_sequence([0])

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == parent.direct_value
    assert parent.best_branch_sequence == []
