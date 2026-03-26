"""Tests for the single-agent max evaluation family."""

# ruff: noqa: D103

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

from valanga import BranchKey
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.common import ProofClassification, SelectedValue
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BestBranchEquivalenceMode,
)
from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.objectives import SingleAgentMaxObjective
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic


class _SoloRole(Enum):
    SOLO = "solo"


@dataclass(frozen=True)
class _FakeOverEvent:
    done: bool = True
    winner: object | None = None
    loser: object | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.done

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: object) -> bool:
        return self.winner == role

    def is_loss_for(self, role: object) -> bool:
        if self.draw:
            return False
        if self.loser is not None:
            return self.loser == role
        if self.winner is not None:
            return self.winner != role
        return False


class _SelectDirectAggregationPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def select_value(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
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
        node_eval: NodeMaxEvaluation[Any],
        selection: SelectedValue,
    ) -> ProofClassification | None:
        self.calls += 1
        del node_eval
        assert selection.value is not None
        return ProofClassification(
            certainty=Certainty.FORCED,
            over_event=_FakeOverEvent(),
        )


def _state() -> Any:
    return SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent")


def _node(
    *,
    node_id: int,
    objective: SingleAgentMaxObjective[Any] | None = None,
    direct_value: Value | None = None,
    backed_up_value: Value | None = None,
    children: dict[int, Any] | None = None,
    all_branches_generated: bool = True,
) -> NodeMaxEvaluation[Any]:
    tree_node = SimpleNamespace(
        id=node_id,
        state=_state(),
        branches_children=children or {},
        all_branches_generated=all_branches_generated,
    )
    if objective is None:
        evaluation = NodeMaxEvaluation(tree_node=tree_node)
    else:
        evaluation = NodeMaxEvaluation(tree_node=tree_node, objective=objective)
    evaluation.direct_value = direct_value
    evaluation.backed_up_value = backed_up_value
    if children:
        evaluation.update_branches_values(
            {
                branch_key
                for branch_key, child in children.items()
                if child is not None
                and child.tree_evaluation.get_value_candidate() is not None
            }
        )
    return evaluation


def _child(
    node_id: int,
    value: Value | None,
    *,
    best_branch_sequence: list[int] | None = None,
) -> Any:
    tree_node = SimpleNamespace(id=node_id, state=_state(), branches_children={})
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = value
    evaluation.backed_up_value = value
    evaluation.set_best_branch_sequence(list(best_branch_sequence or []))
    return SimpleNamespace(
        id=node_id,
        tree_node=tree_node,
        tree_evaluation=evaluation,
    )


def _set_child_value(child: Any, value: Value) -> None:
    child.tree_evaluation.direct_value = value
    child.tree_evaluation.backed_up_value = value


def _assert_single_agent_ordering(
    node: NodeMaxEvaluation[Any],
    *,
    best_branch: int,
    second_best_branch: int,
    ordered_branches: list[int],
    score: float,
) -> None:
    assert node.best_branch() == best_branch
    assert node.second_best_branch() == second_best_branch
    assert node.decision_ordered_branches() == ordered_branches
    assert node.backed_up_value == Value(score=score, certainty=Certainty.ESTIMATE)


def test_single_agent_objective_uses_raw_score_without_turn() -> None:
    objective = SingleAgentMaxObjective(terminal_score_value=-7.0)
    state = _state()

    assert (
        objective.evaluate_value(Value(score=3.5, certainty=Certainty.ESTIMATE), state)
        == 3.5
    )
    assert (
        objective.semantic_compare(
            Value(score=2.0, certainty=Certainty.ESTIMATE),
            Value(score=1.0, certainty=Certainty.ESTIMATE),
            state,
        )
        > 0
    )
    assert objective.terminal_score(_FakeOverEvent(), state) == -7.0


def test_single_agent_objective_prefers_terminal_on_equal_score() -> None:
    objective = SingleAgentMaxObjective()
    state = _state()
    terminal = Value(
        score=1.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(),
    )
    estimate = Value(score=1.0, certainty=Certainty.ESTIMATE)

    assert objective.semantic_compare(terminal, estimate, state) > 0


def test_node_max_defaults_to_explicit_max_backup_policy() -> None:
    node = _node(node_id=0, direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE))

    assert isinstance(node.backup_policy, ExplicitMaxBackupPolicy)


def test_node_max_get_value_prefers_backed_up_value() -> None:
    direct = Value(score=0.1, certainty=Certainty.ESTIMATE)
    backed_up = Value(score=0.9, certainty=Certainty.ESTIMATE)
    node = _node(node_id=0, direct_value=direct, backed_up_value=backed_up)

    assert node.get_value() == backed_up
    assert node.get_score() == 0.9


def test_node_max_get_value_falls_back_to_direct_value() -> None:
    direct = Value(score=0.4, certainty=Certainty.ESTIMATE)
    node = _node(node_id=0, direct_value=direct, backed_up_value=None)

    assert node.get_value() == direct
    assert node.get_score() == 0.4


def test_explicit_max_backup_chooses_highest_scoring_child_and_updates_pv() -> None:
    children = {
        0: _child(
            10, Value(score=0.2, certainty=Certainty.ESTIMATE), best_branch_sequence=[7]
        ),
        1: _child(
            11,
            Value(score=0.9, certainty=Certainty.ESTIMATE),
            best_branch_sequence=[8, 9],
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )
    node.backup_policy = ExplicitMaxBackupPolicy()

    result = node.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq={1},
    )

    assert result.value_changed
    assert result.pv_changed
    assert node.backed_up_value == Value(score=0.9, certainty=Certainty.ESTIMATE)
    assert node.best_branch() == 1
    assert node.best_branch_sequence == [1, 8, 9]


def test_explicit_max_backup_uses_best_child_backed_up_value_in_partial_mode() -> None:
    children = {
        0: _child(10, Value(score=0.2, certainty=Certainty.ESTIMATE)),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.5, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=False,
    )
    node.backup_policy = ExplicitMaxBackupPolicy()

    node.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert node.backed_up_value == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert node.get_value() == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert node.best_branch_sequence == [0]


def test_explicit_max_backup_leaves_backed_up_value_absent_without_child_value() -> (
    None
):
    children = {
        0: _child(10, None),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.3, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=False,
    )
    node.set_best_branch_sequence([0])

    node.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert node.backed_up_value is None
    assert node.get_value() == Value(score=0.3, certainty=Certainty.ESTIMATE)
    assert node.best_branch() is None
    assert node.best_branch_sequence == []


def test_explicit_max_backup_updates_pv_tail_for_best_child_changes() -> None:
    children = {
        0: _child(10, Value(score=0.2, certainty=Certainty.ESTIMATE)),
        1: _child(
            11,
            Value(score=0.9, certainty=Certainty.ESTIMATE),
            best_branch_sequence=[8],
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    node.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq={1},
    )
    assert node.best_branch_sequence == [1, 8]

    children[1].tree_evaluation.set_best_branch_sequence([10, 11])
    result = node.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={1},
    )

    assert result.pv_changed
    assert node.best_branch_sequence == [1, 10, 11]


def test_explicit_max_backup_uses_injected_aggregation_policy() -> None:
    children = {
        0: _child(10, Value(score=0.9, certainty=Certainty.ESTIMATE)),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.2, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )
    aggregation_policy = _SelectDirectAggregationPolicy()
    node.backup_policy = ExplicitMaxBackupPolicy(aggregation_policy=aggregation_policy)

    node.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert aggregation_policy.calls == 1
    assert node.backed_up_value == Value(score=0.2, certainty=Certainty.ESTIMATE)
    assert node.best_branch_sequence == []


def test_explicit_max_backup_uses_injected_proof_policy() -> None:
    children = {
        0: _child(10, Value(score=0.9, certainty=Certainty.ESTIMATE)),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.2, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )
    aggregation_policy = _SelectDirectAggregationPolicy()
    proof_policy = _ForcedProofPolicy()
    node.backup_policy = ExplicitMaxBackupPolicy(
        aggregation_policy=aggregation_policy,
        proof_policy=proof_policy,
    )

    node.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert proof_policy.calls == 1
    assert node.backed_up_value == Value(
        score=0.2,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(),
    )
    assert node.best_branch_sequence == []


def test_single_agent_decision_ordered_branches_are_public_and_deterministic() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(20, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            1: _child(10, Value(score=0.5, certainty=Certainty.ESTIMATE)),
        },
    )

    assert node.decision_ordered_branches() == [1, 0]
    assert node.best_branch() == node.decision_ordered_branches()[0]


def test_single_agent_uses_shared_decision_ordering_state_cache() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(20, Value(score=0.5, certainty=Certainty.ESTIMATE)),
            1: _child(10, Value(score=0.5, certainty=Certainty.ESTIMATE)),
        },
    )

    assert node.decision_ordered_branches() == [1, 0]
    assert node.decision_ordering.branch_ordering_keys == {
        1: BranchOrderingKey(
            primary_score=0.5,
            tactical_tiebreak=0,
            stable_tiebreak_id=10,
        ),
        0: BranchOrderingKey(
            primary_score=0.5,
            tactical_tiebreak=0,
            stable_tiebreak_id=20,
        ),
    }


def test_single_agent_considered_equal_respects_exactness_tie_break() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(
                    score=0.9,
                    certainty=Certainty.FORCED,
                    over_event=_FakeOverEvent(),
                ),
            ),
            1: _child(11, Value(score=0.9, certainty=Certainty.ESTIMATE)),
        },
    )

    assert node.best_branch() == 0
    assert node.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0]
    assert node.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0]


def test_single_agent_best_equivalent_branches_support_generic_modes() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(10, Value(score=0.9, certainty=Certainty.ESTIMATE)),
            1: _child(11, Value(score=0.9, certainty=Certainty.ESTIMATE)),
            2: _child(12, Value(score=0.895, certainty=Certainty.ESTIMATE)),
        },
    )

    assert node.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0, 1]
    assert node.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0, 1]
    assert node.best_equivalent_branches(BestBranchEquivalenceMode.ALMOST_EQUAL) == [
        0,
        1,
        2,
    ]
    assert node.best_equivalent_branches(
        BestBranchEquivalenceMode.ALMOST_EQUAL_LOGISTIC
    ) == [0, 1]


def test_single_agent_equal_uses_exact_line_quality_without_using_id() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(score=0.9, certainty=Certainty.FORCED),
                best_branch_sequence=[8],
            ),
            1: _child(
                11,
                Value(score=0.9, certainty=Certainty.FORCED),
                best_branch_sequence=[9],
            ),
            2: _child(
                12,
                Value(score=0.9, certainty=Certainty.FORCED),
                best_branch_sequence=[7, 6],
            ),
        },
    )

    assert node.best_branch() == 0
    assert node.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0, 1]
    assert node.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0, 1, 2]


def test_single_agent_equal_uses_shared_pv_length_tie_break_for_estimates() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(score=0.9, certainty=Certainty.ESTIMATE),
                best_branch_sequence=[8],
            ),
            1: _child(
                11,
                Value(score=0.9, certainty=Certainty.ESTIMATE),
                best_branch_sequence=[7, 6],
            ),
        },
    )

    assert node.best_branch() == 0
    assert node.best_equivalent_branches(BestBranchEquivalenceMode.EQUAL) == [0]
    assert node.best_equivalent_branches(
        BestBranchEquivalenceMode.CONSIDERED_EQUAL
    ) == [0, 1]


def test_single_agent_branch_order_prefers_shorter_favorable_exact_lines() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(
                    score=1.0,
                    certainty=Certainty.FORCED,
                    over_event=_FakeOverEvent(winner=_SoloRole.SOLO),
                ),
                best_branch_sequence=[8, 7, 6],
            ),
            1: _child(
                11,
                Value(
                    score=1.0,
                    certainty=Certainty.FORCED,
                    over_event=_FakeOverEvent(winner=_SoloRole.SOLO),
                ),
                best_branch_sequence=[9],
            ),
        },
    )

    assert node.best_branch() == 1


def test_single_agent_branch_order_prefers_longer_unfavorable_exact_lines() -> None:
    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(
                    score=0.0,
                    certainty=Certainty.FORCED,
                    over_event=_FakeOverEvent(loser=_SoloRole.SOLO),
                ),
                best_branch_sequence=[8],
            ),
            1: _child(
                11,
                Value(
                    score=0.0,
                    certainty=Certainty.FORCED,
                    over_event=_FakeOverEvent(loser=_SoloRole.SOLO),
                ),
                best_branch_sequence=[9, 7, 6],
            ),
        },
    )

    assert node.best_branch() == 1


def test_single_agent_exact_draw_like_outcomes_keep_default_shorter_policy() -> None:
    @dataclass(frozen=True)
    class _DrawOverEvent:
        def is_over(self) -> bool:
            return True

        def is_draw(self) -> bool:
            return True

        def is_win_for(self, role: object) -> bool:
            del role
            return False

        def is_loss_for(self, role: object) -> bool:
            del role
            return False

    node = _node(
        node_id=0,
        children={
            0: _child(
                10,
                Value(
                    score=0.0,
                    certainty=Certainty.FORCED,
                    over_event=_DrawOverEvent(),
                ),
                best_branch_sequence=[8],
            ),
            1: _child(
                11,
                Value(
                    score=0.0,
                    certainty=Certainty.FORCED,
                    over_event=_DrawOverEvent(),
                ),
                best_branch_sequence=[9, 7],
            ),
        },
    )

    assert node.best_branch() == 0


def test_almost_equal_logistic_recommender_uses_generic_evaluation_capability() -> None:
    tree_evaluation = _node(
        node_id=0,
        children={
            0: _child(10, Value(score=0.9, certainty=Certainty.ESTIMATE)),
            1: _child(11, Value(score=0.895, certainty=Certainty.ESTIMATE)),
            2: _child(12, Value(score=0.2, certainty=Certainty.ESTIMATE)),
        },
    )
    root_node = SimpleNamespace(
        tree_evaluation=tree_evaluation,
        branches_children=tree_evaluation.tree_node.branches_children,
    )
    rule = AlmostEqualLogistic(type="almost_equal_logistic", temperature=1.0)

    policy = rule.policy(root_node)

    assert policy.probs == {0: 1.0}


def test_single_agent_exact_value_and_terminality_are_distinct() -> None:
    node = _node(
        node_id=0,
        direct_value=Value(
            score=5.0,
            certainty=Certainty.FORCED,
            over_event=_FakeOverEvent(),
        ),
    )

    assert node.has_exact_value()
    assert not node.is_terminal()
    assert node.has_over_event()
    assert node.over_event is not None


def test_single_agent_pv_state_tracks_version_and_cache() -> None:
    children = {
        1: _child(
            11,
            Value(score=0.9, certainty=Certainty.ESTIMATE),
            best_branch_sequence=[8, 9],
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    assert node.set_best_branch_sequence([1, 8, 9])
    assert node.pv_version == 1
    assert node.pv_cached_best_child_version == children[1].tree_evaluation.pv_version

    assert not node.set_best_branch_sequence([1, 8, 9])
    assert node.pv_version == 1

    assert node.clear_best_branch_sequence()
    assert node.best_branch_sequence == []
    assert node.pv_version == 2
    assert node.pv_cached_best_child_version is None


def test_single_agent_incremental_pv_update_uses_shared_state() -> None:
    children = {
        1: _child(
            11,
            Value(score=0.9, certainty=Certainty.ESTIMATE),
            best_branch_sequence=[8, 9],
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    assert node.set_best_branch_sequence([1, 8, 9])
    version_before = node.pv_version

    assert not node.update_best_branch_sequence(set())
    assert node.pv_version == version_before

    children[1].tree_evaluation.set_best_branch_sequence([10, 11])

    assert node.update_best_branch_sequence({1})
    assert node.best_branch_sequence == [1, 10, 11]
    assert node.pv_version == version_before + 1
    assert node.pv_cached_best_child_version == children[1].tree_evaluation.pv_version


def test_single_agent_pv_invariant_helper_is_available_from_shared_state() -> None:
    children = {
        1: _child(
            11,
            Value(score=0.9, certainty=Certainty.ESTIMATE),
            best_branch_sequence=[8, 9],
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.1, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    node.set_best_branch_sequence([1, 8, 9])

    node.assert_pv_invariants()


def test_single_agent_best_and_second_best_transition_sequence_with_three_children() -> (
    None
):
    children = {
        0: _child(10, Value(score=0.9, certainty=Certainty.ESTIMATE)),
        1: _child(20, Value(score=0.6, certainty=Certainty.ESTIMATE)),
        2: _child(30, Value(score=0.2, certainty=Certainty.ESTIMATE)),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    node.backup_from_children(
        branches_with_updated_value={0, 1, 2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    # Non-best improves but remains second-best.
    _set_child_value(children[1], Value(score=0.8, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    # Second-best overtakes the current best.
    _set_child_value(children[1], Value(score=1.0, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=1,
        second_best_branch=0,
        ordered_branches=[1, 0, 2],
        score=1.0,
    )

    # Current best worsens below the former second-best.
    _set_child_value(children[1], Value(score=0.85, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=0,
        second_best_branch=1,
        ordered_branches=[0, 1, 2],
        score=0.9,
    )

    # Third child becomes second-best without becoming best.
    _set_child_value(children[2], Value(score=0.88, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=0,
        second_best_branch=2,
        ordered_branches=[0, 2, 1],
        score=0.9,
    )

    # A tie appears and stable tie-breaking keeps the lower-id child first.
    _set_child_value(children[2], Value(score=0.9, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=0,
        second_best_branch=2,
        ordered_branches=[0, 2, 1],
        score=0.9,
    )

    # The tie disappears and the former third child becomes best.
    _set_child_value(children[2], Value(score=0.91, certainty=Certainty.ESTIMATE))
    node.backup_from_children(
        branches_with_updated_value={2},
        branches_with_updated_best_branch_seq=set(),
    )
    _assert_single_agent_ordering(
        node,
        best_branch=2,
        second_best_branch=0,
        ordered_branches=[2, 0, 1],
        score=0.91,
    )


def test_single_agent_exactness_reverses_when_child_exactness_and_generation_change() -> (
    None
):
    children = {
        0: _child(
            10,
            Value(
                score=1.0,
                certainty=Certainty.TERMINAL,
                over_event=_FakeOverEvent(winner=_SoloRole.SOLO),
            ),
        ),
        1: _child(
            11,
            Value(
                score=0.2,
                certainty=Certainty.TERMINAL,
                over_event=_FakeOverEvent(loser=_SoloRole.SOLO),
            ),
        ),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.0, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=True,
    )

    first = node.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert node.backed_up_value == Value(
        score=1.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=_SoloRole.SOLO),
    )
    assert node.has_exact_value()
    assert node.has_over_event()
    assert first.over_changed

    # Losing one exact child makes the parent fall back to an estimate.
    _set_child_value(children[1], Value(score=0.2, certainty=Certainty.ESTIMATE))
    second = node.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert node.backed_up_value == Value(score=1.0, certainty=Certainty.ESTIMATE)
    assert not node.has_exact_value()
    assert not node.has_over_event()
    assert second.over_changed

    # Restoring child exactness is not enough if generation completeness is broken.
    _set_child_value(
        children[1],
        Value(
            score=0.2,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(loser=_SoloRole.SOLO),
        ),
    )
    node.tree_node.all_branches_generated = False
    third = node.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert node.backed_up_value == Value(score=1.0, certainty=Certainty.ESTIMATE)
    assert not node.has_exact_value()
    assert not node.has_over_event()
    assert not third.over_changed

    # Re-enabling full generation restores the exact parent result.
    node.tree_node.all_branches_generated = True
    fourth = node.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq=set(),
    )
    assert node.backed_up_value == Value(
        score=1.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=_SoloRole.SOLO),
    )
    assert node.has_exact_value()
    assert node.has_over_event()
    assert fourth.over_changed
