"""Tests for default backup policy wiring."""

from types import SimpleNamespace

from valanga import Color

from anemone.backup_policies import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.values import Certainty, Value


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
    ev.sync_over_from_values()
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


def test_factory_defaults_to_explicit_policy() -> None:
    """Factory should default to explicit backup policy."""
    explicit_factory = NodeTreeMinmaxEvaluationFactory()
    explicit_eval = explicit_factory.create(_leaf(0.2).tree_node)
    assert isinstance(explicit_eval.backup_policy, ExplicitMinimaxBackupPolicy)

