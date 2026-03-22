"""Tests for default backup policy wiring."""

from enum import Enum
from types import SimpleNamespace

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies import (
    BestChildAggregationPolicy,
    ExplicitMaxBackupPolicy,
    ExplicitMinimaxBackupPolicy,
    MaxProofPolicy,
    MinimaxProofPolicy,
)
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


class _SoloRole(Enum):
    SOLO = "solo"


def _leaf(score: float) -> SimpleNamespace:
    tree_node = SimpleNamespace(
        id=1,
        state=SimpleNamespace(turn=Color.BLACK),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=tree_node)
    ev.direct_value = Value(score=score, certainty=Certainty.ESTIMATE)
    ev.minmax_value = Value(score=score, certainty=Certainty.ESTIMATE)
    return SimpleNamespace(id=tree_node.id, tree_node=tree_node, tree_evaluation=ev)


def _single_agent_tree_node() -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children={},
        all_branches_generated=True,
    )


def test_node_minmax_evaluation_backup_default_is_explicit_policy() -> None:
    """Backup fallback policy should initialize to explicit minimax."""
    child = _leaf(0.6)
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={0: child},
        all_branches_generated=True,
    )
    node_eval = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    assert isinstance(node_eval.backup_policy, ExplicitMinimaxBackupPolicy)
    node_eval.direct_value = Value(score=0.1, certainty=Certainty.ESTIMATE)

    node_eval.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert isinstance(node_eval.objective, AdversarialZeroSumObjective)


def test_factory_defaults_to_explicit_policy() -> None:
    """Factory should default to explicit backup policy."""
    explicit_factory = NodeTreeMinmaxEvaluationFactory()
    explicit_eval = explicit_factory.create(_leaf(0.2).tree_node)
    assert isinstance(explicit_eval, NodeMinmaxEvaluation)
    assert isinstance(explicit_eval.backup_policy, ExplicitMinimaxBackupPolicy)
    assert isinstance(explicit_eval.objective, AdversarialZeroSumObjective)


def test_single_agent_factory_defaults_to_explicit_policy() -> None:
    """Single-agent factory should default to explicit max backup policy."""
    factory = NodeMaxEvaluationFactory()
    evaluation = factory.create(_single_agent_tree_node())

    assert isinstance(evaluation, NodeMaxEvaluation)
    assert isinstance(evaluation.backup_policy, ExplicitMaxBackupPolicy)
    assert isinstance(evaluation.objective, SingleAgentMaxObjective)


def test_factories_keep_injected_family_dependencies() -> None:
    """Factory wiring should forward explicitly-provided family dependencies unchanged."""
    minimax_backup_policy = ExplicitMinimaxBackupPolicy()
    minimax_objective = AdversarialZeroSumObjective()
    minimax_eval = NodeTreeMinmaxEvaluationFactory(
        backup_policy=minimax_backup_policy,
        objective=minimax_objective,
    ).create(_leaf(0.2).tree_node)

    single_agent_backup_policy = ExplicitMaxBackupPolicy()
    single_agent_objective = SingleAgentMaxObjective(terminal_score_value=-3.0)
    single_agent_eval = NodeMaxEvaluationFactory(
        backup_policy=single_agent_backup_policy,
        objective=single_agent_objective,
    ).create(_single_agent_tree_node())

    assert minimax_eval.backup_policy is minimax_backup_policy
    assert minimax_eval.objective is minimax_objective
    assert single_agent_eval.backup_policy is single_agent_backup_policy
    assert single_agent_eval.objective is single_agent_objective


def test_explicit_backup_policies_default_to_family_policy_objects() -> None:
    """Explicit backup policies should install the matching aggregation/proof policies."""
    assert isinstance(
        ExplicitMaxBackupPolicy().aggregation_policy, BestChildAggregationPolicy
    )
    assert isinstance(ExplicitMaxBackupPolicy().proof_policy, MaxProofPolicy)
    assert isinstance(
        ExplicitMinimaxBackupPolicy().aggregation_policy,
        BestChildAggregationPolicy,
    )
    assert isinstance(ExplicitMinimaxBackupPolicy().proof_policy, MinimaxProofPolicy)
