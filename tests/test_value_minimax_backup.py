"""Tests for Value-driven minimax backup in explicit policy."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import Color, FloatyStateEvaluation

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
            0: _make_leaf(1, Value(score=0.2, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(score=0.7, certainty=Certainty.ESTIMATE)),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )
    white_parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert white_parent.minmax_value == Value(score=0.7, certainty=Certainty.ESTIMATE)
    assert white_parent.get_score() == 0.7

    black_parent = _make_parent(
        turn=Color.BLACK,
        children={
            0: _make_leaf(1, Value(score=0.2, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(score=0.7, certainty=Certainty.ESTIMATE)),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )
    black_parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert black_parent.minmax_value == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert black_parent.get_score() == 0.2


def test_partial_expansion_prefers_direct_when_better_for_side_to_move() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.2, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.5, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.5


def test_partial_expansion_prefers_child_for_black_when_child_is_better() -> None:
    parent = _make_parent(
        turn=Color.BLACK,
        children={0: _make_leaf(1, Value(score=0.2, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.2


def test_partial_expansion_prefers_direct_for_black_when_direct_is_better() -> None:
    parent = _make_parent(
        turn=Color.BLACK,
        children={0: _make_leaf(1, Value(score=0.7, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.5, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.5


def test_semantic_compare_terminal_vs_estimate_is_exposed() -> None:
    forced_win = Value(
        score=-10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    estimate = Value(score=999.0, certainty=Certainty.ESTIMATE)
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
    very_high_estimate = Value(score=10.0, certainty=Certainty.ESTIMATE)

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, forced_win), 1: _make_leaf(2, very_high_estimate)},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
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
        value_before=Value(score=0.1, certainty=Certainty.ESTIMATE),
        value_after=Value(score=0.2, certainty=Certainty.ESTIMATE),
    )
    assert has_value_changed(
        value_before=Value(score=0.1, certainty=Certainty.ESTIMATE),
        value_after=Value(score=0.1, certainty=Certainty.FORCED),
    )
    assert has_value_changed(
        value_before=Value(score=0.1, certainty=Certainty.ESTIMATE, over_event=None),
        value_after=Value(
            score=0.1,
            certainty=Certainty.ESTIMATE,
            over_event=_FakeOverEvent(draw=True),
        ),
    )


def test_direct_eval_change_without_minmax_change_reports_no_value_change() -> None:
    child_value = Value(score=0.7, certainty=Certainty.ESTIMATE)
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, child_value)},
        all_generated=False,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )
    parent.direct_value = Value(score=0.6, certainty=Certainty.ESTIMATE)
    parent.sync_float_views_from_values()

    result = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == child_value
    assert not result.value_changed


def test_partial_expansion_pv_does_not_depend_on_float_field() -> None:
    child_value = Value(score=0.9, certainty=Certainty.ESTIMATE)
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, child_value)},
        all_generated=False,
        direct_value=Value(score=0.8, certainty=Certainty.ESTIMATE),
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
        direct_value=Value(score=0.8, certainty=Certainty.ESTIMATE),
    )
    parent.set_best_branch_sequence([0])

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == parent.direct_value
    assert parent.best_branch_sequence == []


def test_partial_expansion_requires_direct_value() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.4, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    parent.direct_value = None

    try:
        parent.backup_from_children(
            branches_with_updated_value={0},
            branches_with_updated_best_branch_seq=set(),
        )
    except AssertionError as exc:
        assert "direct_value" in str(exc)
    else:
        raise AssertionError


def test_terminal_value_syncs_parent_over_event_and_reports_over_changed() -> None:
    terminal = Value(
        score=0.2,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, terminal)},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    result = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == terminal
    assert parent.over_event == terminal.over_event
    assert result.over_changed


def test_best_branch_tail_update_propagates_without_head_rebuild() -> None:
    child = _make_leaf(1, Value(score=0.8, certainty=Certainty.ESTIMATE))
    child.tree_evaluation.set_best_branch_sequence([9])

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: child},
        all_generated=True,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq={0},
    )
    assert parent.best_branch_sequence == [0, 9]

    child.tree_evaluation.set_best_branch_sequence([8, 7])
    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={0},
    )

    assert parent.best_branch_sequence == [0, 8, 7]
    assert result.pv_changed


def test_best_branch_selection_uses_semantic_compare_terminal_over_estimate() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=100.0, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(
                score=0.0,
                certainty=Certainty.TERMINAL,
                over_event=_FakeOverEvent(winner=Color.WHITE),
            )),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 1


def test_branch_ordering_for_search_uses_projection_and_stable_tie_breakers() -> None:
    child_a = _make_leaf(1, Value(score=0.5, certainty=Certainty.ESTIMATE))
    child_b = _make_leaf(2, Value(score=0.5, certainty=Certainty.ESTIMATE))

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: child_a, 1: child_b},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert list(parent.branches_sorted_by_value.keys()) == [0, 1]


def test_evaluate_does_not_emit_forced_outcome_for_non_terminal_estimate_with_over_event() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={},
        all_generated=True,
        direct_value=Value(
            score=0.25,
            certainty=Certainty.ESTIMATE,
            over_event=_FakeOverEvent(winner=Color.WHITE),
        ),
    )

    result = parent.evaluate()

    assert isinstance(result, FloatyStateEvaluation)
    assert result.value_white == 0.25
