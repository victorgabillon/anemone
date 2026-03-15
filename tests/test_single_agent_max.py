"""Tests for the single-agent max evaluation family."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.objectives import SingleAgentMaxObjective
from valanga.evaluations import Certainty, Value


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
    evaluation.best_branch_sequence = list(best_branch_sequence or [])
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
