"""Tests for the single-agent max evaluation family."""

# ruff: noqa: D103

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import BranchKey
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.common import ProofClassification, SelectedValue
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.objectives import SingleAgentMaxObjective


@dataclass(frozen=True)
class _FakeOverEvent:
    done: bool = True

    def is_over(self) -> bool:
        return self.done

    def is_draw(self) -> bool:
        return False

    def is_winner(self, player: object) -> bool:
        del player
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
    return SimpleNamespace(phase="single-agent")


def _node(
    *,
    node_id: int,
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
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = direct_value
    evaluation.backed_up_value = backed_up_value
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
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=evaluation)


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


def test_explicit_max_backup_falls_back_to_direct_value_when_direct_is_better() -> None:
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

    assert node.backed_up_value == Value(score=0.5, certainty=Certainty.ESTIMATE)
    assert node.best_branch_sequence == []


def test_explicit_max_backup_falls_back_to_direct_when_no_child_value_exists() -> None:
    children = {
        0: _child(10, None),
    }
    node = _node(
        node_id=0,
        direct_value=Value(score=0.3, certainty=Certainty.ESTIMATE),
        children=children,
        all_branches_generated=False,
    )

    node.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert node.backed_up_value == Value(score=0.3, certainty=Certainty.ESTIMATE)
    assert node.best_branch() is None


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
