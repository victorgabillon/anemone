"""Tests for Value-driven minimax backup in explicit policy."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import BranchKey, Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.common import (
    ProofClassification,
    SelectedValue,
    has_value_changed,
)
from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common import FieldChange
from anemone.node_evaluation.common.canonical_value import ValueSemanticsError
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BestBranchEquivalenceMode,
)
from anemone.values import DEFAULT_EVALUATION_ORDERING


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: Color) -> bool:
        return self.winner == role

    def is_loss_for(self, role: Color) -> bool:
        return self.winner is not None and self.winner != role


class _SelectDirectAggregationPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def select_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        self.calls += 1
        assert branches_with_updated_value
        return SelectedValue(value=node_eval.direct_value, from_child=False)


class _ForcedProofPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def classify_selected_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selection: SelectedValue,
    ) -> ProofClassification | None:
        self.calls += 1
        del node_eval
        assert selection.value is not None
        return ProofClassification(
            certainty=Certainty.FORCED,
            over_event=_FakeOverEvent(winner=Color.WHITE),
        )


class _FlipAllBranchesGeneratedPolicy:
    def __init__(self, *, new_value: bool) -> None:
        self.new_value = new_value

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult[BranchKey]:
        del branches_with_updated_value
        del branches_with_updated_best_branch_seq
        node_eval.tree_node.all_branches_generated = self.new_value
        return BackupResult(
            value_changed=False,
            pv_changed=False,
            over_changed=False,
        )


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
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=ev)


def _make_unvalued_leaf(node_id: int) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=tree_node)
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=ev)


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
    return parent


def _set_leaf_value(child: Any, value: Value) -> None:
    child.tree_evaluation.direct_value = value
    child.tree_evaluation.minmax_value = value


def _assert_minimax_ordering(
    parent: NodeMinmaxEvaluation[Any, Any],
    *,
    best_branch: int,
    second_best_branch: int,
    ordered_branches: list[int],
    score: float,
) -> None:
    assert parent.best_branch() == best_branch
    assert parent.second_best_branch() == second_best_branch
    assert parent.decision_ordered_branches() == ordered_branches
    assert parent.minmax_value == Value(score=score, certainty=Certainty.ESTIMATE)


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


def test_partial_expansion_uses_best_child_backed_up_value_for_white() -> None:
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

    assert parent.minmax_value == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.2


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


def test_partial_expansion_uses_best_child_backed_up_value_for_black() -> None:
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

    assert parent.minmax_value == Value(score=0.7, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.7


def test_explicit_minimax_backup_uses_injected_aggregation_policy() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.9, certainty=Certainty.ESTIMATE))},
        all_generated=True,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    aggregation_policy = _SelectDirectAggregationPolicy()
    parent.backup_policy = ExplicitMinimaxBackupPolicy(
        aggregation_policy=aggregation_policy
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert aggregation_policy.calls == 1
    assert parent.minmax_value == Value(score=0.1, certainty=Certainty.ESTIMATE)
    assert parent.best_branch_sequence == []


def test_explicit_minimax_backup_uses_injected_proof_policy() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.9, certainty=Certainty.ESTIMATE))},
        all_generated=True,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    aggregation_policy = _SelectDirectAggregationPolicy()
    proof_policy = _ForcedProofPolicy()
    parent.backup_policy = ExplicitMinimaxBackupPolicy(
        aggregation_policy=aggregation_policy,
        proof_policy=proof_policy,
    )

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert proof_policy.calls == 1
    assert parent.minmax_value == Value(
        score=0.1,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    assert parent.best_branch_sequence == []


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

    ordered_branches = list(parent.decision_ordering.branch_ordering_keys)
    assert ordered_branches.index(1) < ordered_branches.index(0)
    assert parent.best_branch() == 0


def test_minimax_best_equivalent_branches_use_family_semantics_not_id_tie_break() -> (
    None
):
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            2: _make_leaf(3, Value(score=0.497, certainty=Certainty.ESTIMATE)),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.update_branches_values({0, 1, 2})

    assert parent.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0, 1]
    assert parent.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0, 1]
    assert parent.best_equivalent_branches(BestBranchEquivalenceMode.ALMOST_EQUAL) == [
        0,
        1,
        2,
    ]
    assert parent.best_equivalent_branches(
        BestBranchEquivalenceMode.ALMOST_EQUAL_LOGISTIC
    ) == [0, 1, 2]


def test_minimax_equal_respects_pv_tie_break_without_using_node_id() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            2: _make_leaf(3, Value(score=0.5, certainty=Certainty.ESTIMATE)),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.tree_node.branches_children[0].tree_evaluation.set_best_branch_sequence([8])
    parent.tree_node.branches_children[1].tree_evaluation.set_best_branch_sequence([9])
    parent.tree_node.branches_children[2].tree_evaluation.set_best_branch_sequence(
        [7, 6]
    )

    parent.update_branches_values({0, 1, 2})

    assert parent.best_branch() == 0
    assert parent.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0, 1]
    assert parent.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0, 1, 2]
    assert parent.best_equivalent_branches(
        BestBranchEquivalenceMode.ALMOST_EQUAL_LOGISTIC
    ) == [0, 1, 2]


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

    result = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == child_value
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()


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

    assert parent.minmax_value is None
    assert parent.get_value() == parent.direct_value
    assert parent.best_branch_sequence == []


def test_partial_expansion_does_not_require_direct_value_when_child_value_exists() -> (
    None
):
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.4, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    parent.direct_value = None

    parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value == Value(score=0.4, certainty=Certainty.ESTIMATE)
    assert parent.get_score() == 0.4


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

    assert parent.minmax_value == Value(
        score=0.2,
        certainty=Certainty.FORCED,
        over_event=terminal.over_event,
    )
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
    version_before = parent.pv_version
    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={0},
    )

    assert parent.best_branch_sequence == [0, 8, 7]
    assert result.pv_changed
    assert not result.value_changed
    assert not result.over_changed
    assert result.node_delta.value is None
    assert result.node_delta.best_branch is None
    assert result.node_delta.all_branches_generated is None
    assert result.node_delta.pv_version == FieldChange(
        old=version_before,
        new=version_before + 1,
    )


def test_backup_delta_reports_all_branches_generated_change() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, Value(score=0.3, certainty=Certainty.ESTIMATE))},
        all_generated=False,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
    )
    parent.backup_policy = _FlipAllBranchesGeneratedPolicy(new_value=True)

    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.get_value() == Value(score=0.5, certainty=Certainty.ESTIMATE)
    assert parent.tree_node.all_branches_generated is True
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.value is None
    assert result.node_delta.pv_version is None
    assert result.node_delta.best_branch is None
    assert result.node_delta.all_branches_generated == FieldChange(
        old=False,
        new=True,
    )


def test_best_branch_selection_uses_semantic_compare_terminal_over_estimate() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=100.0, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(
                2,
                Value(
                    score=0.0,
                    certainty=Certainty.TERMINAL,
                    over_event=_FakeOverEvent(winner=Color.WHITE),
                ),
            ),
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

    assert list(parent.decision_ordering.branch_ordering_keys) == [0, 1]
    assert parent.decision_ordered_branches() == [0, 1]
    assert parent.second_best_branch() == 1


def test_minimax_decision_ordering_refreshes_when_child_value_changes() -> None:
    child_a = _make_leaf(1, Value(score=0.2, certainty=Certainty.ESTIMATE))
    child_b = _make_leaf(2, Value(score=0.7, certainty=Certainty.ESTIMATE))

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: child_a, 1: child_b},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.update_branches_values({0, 1})
    assert parent.decision_ordered_branches() == [1, 0]
    assert parent.best_branch() == 1

    child_a.tree_evaluation.direct_value = Value(
        score=0.9,
        certainty=Certainty.ESTIMATE,
    )
    child_a.tree_evaluation.minmax_value = Value(
        score=0.9,
        certainty=Certainty.ESTIMATE,
    )

    parent.update_branches_values({0})

    assert parent.decision_ordered_branches() == [0, 1]
    assert parent.best_branch() == 0


def test_non_terminal_estimate_with_over_event_is_rejected() -> None:
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

    try:
        parent.get_value()
    except ValueSemanticsError as exc:
        assert "ESTIMATE" in str(exc)
    else:
        raise AssertionError


def test_best_branch_prefers_estimate_over_forced_loss() -> None:
    forced_loss = Value(
        score=10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    slight_estimate = Value(score=0.1, certainty=Certainty.ESTIMATE)

    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, forced_loss), 1: _make_leaf(2, slight_estimate)},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 1


def test_best_branch_draw_baseline_vs_estimate() -> None:
    draw_baseline = DEFAULT_EVALUATION_ORDERING.draw_score
    forced_draw = Value(
        score=999.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(draw=True),
    )

    parent_above = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, forced_draw),
            1: _make_leaf(
                2, Value(score=draw_baseline + 0.1, certainty=Certainty.ESTIMATE)
            ),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )
    parent_above.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert parent_above.best_branch() == 1

    parent_below = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, forced_draw),
            1: _make_leaf(
                2, Value(score=draw_baseline - 0.1, certainty=Certainty.ESTIMATE)
            ),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )
    parent_below.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert parent_below.best_branch() == 0


def test_exact_win_child_keeps_winning_over_event_on_forced_parent() -> None:
    forced_draw = Value(
        score=0.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(draw=True),
    )
    forced_win = Value(
        score=0.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    parent = _make_parent(
        turn=Color.WHITE,
        children={0: _make_leaf(1, forced_draw), 1: _make_leaf(2, forced_win)},
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 1
    assert parent.minmax_value == Value(
        score=0.0,
        certainty=Certainty.FORCED,
        over_event=forced_win.over_event,
    )


def test_minimax_best_and_second_best_transition_sequence_with_three_children() -> None:
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(1, Value(score=0.9, certainty=Certainty.ESTIMATE)),
            1: _make_leaf(2, Value(score=0.6, certainty=Certainty.ESTIMATE)),
            2: _make_leaf(3, Value(score=0.2, certainty=Certainty.ESTIMATE)),
        },
        all_generated=True,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    parent.backup_from_children(
        branches_with_updated_value={0, 1, 2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[1],
        Value(score=0.8, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[1],
        Value(score=1.0, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=1,
        second_best_branch=0,
        ordered_branches=[1, 0, 2],
        score=1.0,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[1],
        Value(score=0.85, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[2],
        Value(score=0.88, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=0,
        second_best_branch=2,
        ordered_branches=[0, 2, 1],
        score=0.9,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[2],
        Value(score=0.9, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=0,
        second_best_branch=2,
        ordered_branches=[0, 2, 1],
        score=0.9,
    )

    _set_leaf_value(
        parent.tree_node.branches_children[2],
        Value(score=0.91, certainty=Certainty.ESTIMATE),
    )
    parent.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_minimax_ordering(
        parent,
        best_branch=2,
        second_best_branch=0,
        ordered_branches=[2, 0, 1],
        score=0.91,
    )


def test_minimax_exactness_reverses_when_winning_child_loses_and_recovers_proof() -> (
    None
):
    parent = _make_parent(
        turn=Color.WHITE,
        children={
            0: _make_leaf(
                1,
                Value(
                    score=1.0,
                    certainty=Certainty.TERMINAL,
                    over_event=_FakeOverEvent(winner=Color.WHITE),
                ),
            ),
            1: _make_leaf(2, Value(score=0.2, certainty=Certainty.ESTIMATE)),
        },
        all_generated=False,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
    )

    first = parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert parent.minmax_value == Value(
        score=1.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    assert parent.has_exact_value()
    assert parent.has_over_event()
    assert first.over_changed

    _set_leaf_value(
        parent.tree_node.branches_children[0],
        Value(score=1.0, certainty=Certainty.ESTIMATE),
    )
    second = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )
    assert parent.minmax_value == Value(score=1.0, certainty=Certainty.ESTIMATE)
    assert not parent.has_exact_value()
    assert not parent.has_over_event()
    assert second.over_changed

    _set_leaf_value(
        parent.tree_node.branches_children[0],
        Value(
            score=1.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=Color.WHITE),
        ),
    )
    third = parent.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )
    assert parent.minmax_value == Value(
        score=1.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    assert parent.has_exact_value()
    assert parent.has_over_event()
    assert third.over_changed
