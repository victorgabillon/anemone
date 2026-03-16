"""Focused cross-family invariants for node-evaluation families."""

from types import SimpleNamespace
from typing import Any

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies import ExplicitMaxBackupPolicy, ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.objectives import AdversarialZeroSumObjective, SingleAgentMaxObjective


class _FakeOverEvent:
    def is_over(self) -> bool:
        return True

    def is_draw(self) -> bool:
        return False

    def is_winner(self, player: object) -> bool:
        del player
        return False


def _adversarial_tree_node() -> Any:
    return SimpleNamespace(
        id=1,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )


def _single_agent_tree_node() -> Any:
    return SimpleNamespace(
        id=2,
        state=SimpleNamespace(phase="single-agent"),
        branches_children={},
        all_branches_generated=True,
    )


def test_canonical_value_semantics_align_across_families() -> None:
    adversarial = NodeMinmaxEvaluation(tree_node=_adversarial_tree_node())
    single_agent = NodeMaxEvaluation(tree_node=_single_agent_tree_node())

    direct = Value(score=0.2, certainty=Certainty.ESTIMATE)
    backed_up = Value(score=0.9, certainty=Certainty.ESTIMATE)

    adversarial.direct_value = direct
    adversarial.backed_up_value = backed_up
    single_agent.direct_value = direct
    single_agent.backed_up_value = backed_up

    assert adversarial.get_value() == backed_up
    assert single_agent.get_value() == backed_up

    adversarial.backed_up_value = None
    single_agent.backed_up_value = None

    assert adversarial.get_value() == direct
    assert single_agent.get_value() == direct


def test_exact_value_and_terminality_semantics_align_across_families() -> None:
    terminal_value = Value(
        score=1.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(),
    )
    adversarial = NodeMinmaxEvaluation(tree_node=_adversarial_tree_node())
    single_agent = NodeMaxEvaluation(tree_node=_single_agent_tree_node())
    adversarial.direct_value = terminal_value
    single_agent.direct_value = terminal_value

    assert adversarial.has_exact_value()
    assert single_agent.has_exact_value()
    assert not adversarial.is_terminal()
    assert not single_agent.is_terminal()
    assert adversarial.over_event is not None
    assert single_agent.over_event is not None


def test_default_construction_is_intentional_in_both_families() -> None:
    adversarial_factory = NodeTreeMinmaxEvaluationFactory()
    single_agent_factory = NodeMaxEvaluationFactory()

    adversarial_eval = adversarial_factory.create(_adversarial_tree_node())
    single_agent_eval = single_agent_factory.create(_single_agent_tree_node())

    assert isinstance(adversarial_eval.objective, AdversarialZeroSumObjective)
    assert isinstance(adversarial_eval.backup_policy, ExplicitMinimaxBackupPolicy)
    assert isinstance(single_agent_eval.objective, SingleAgentMaxObjective)
    assert isinstance(single_agent_eval.backup_policy, ExplicitMaxBackupPolicy)


def test_family_specific_objective_semantics_remain_distinct() -> None:
    adversarial_objective = AdversarialZeroSumObjective()
    single_agent_objective = SingleAgentMaxObjective()
    left = Value(score=0.2, certainty=Certainty.ESTIMATE)
    right = Value(score=0.8, certainty=Certainty.ESTIMATE)

    assert (
        adversarial_objective.semantic_compare(
            left,
            right,
            SimpleNamespace(turn=Color.BLACK),
        )
        > 0
    )
    assert (
        single_agent_objective.semantic_compare(
            left,
            right,
            SimpleNamespace(phase="single-agent"),
        )
        < 0
    )
