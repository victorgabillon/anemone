"""Tests for default backup policy wiring."""

from types import SimpleNamespace

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies import (
    ExplicitMaxBackupPolicy,
    ExplicitMinimaxBackupPolicy,
    MaxAggregationPolicy,
    MinimaxAggregationPolicy,
)
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.objectives import AdversarialZeroSumObjective


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
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=ev)


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
    node_eval.direct_value = Value(score=0.1, certainty=Certainty.ESTIMATE)

    node_eval.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert isinstance(node_eval.backup_policy, ExplicitMinimaxBackupPolicy)
    assert isinstance(node_eval.objective, AdversarialZeroSumObjective)


def test_factory_defaults_to_explicit_policy() -> None:
    """Factory should default to explicit backup policy."""
    explicit_factory = NodeTreeMinmaxEvaluationFactory()
    explicit_eval = explicit_factory.create(_leaf(0.2).tree_node)
    assert isinstance(explicit_eval.backup_policy, ExplicitMinimaxBackupPolicy)
    assert isinstance(explicit_eval.objective, AdversarialZeroSumObjective)


def test_explicit_backup_policies_default_to_family_aggregation_policies() -> None:
    """Explicit backup policies should install the matching aggregation policy."""
    assert isinstance(ExplicitMaxBackupPolicy().aggregation_policy, MaxAggregationPolicy)
    assert isinstance(
        ExplicitMinimaxBackupPolicy().aggregation_policy,
        MinimaxAggregationPolicy,
    )
