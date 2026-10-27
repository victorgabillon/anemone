"""Fail-closed certificate checks for evaluator-only refreshes.

Values carry certainty and an evaluator version, not external proof validity.
The caller must validate direct FORCED origins; tree proofs must be reconstructible
with the existing built-in backup rule. No certificate registry is retained.
"""

from functools import cmp_to_key
from typing import TYPE_CHECKING, Any

from anemone.backup_policies.aggregation import BestChildAggregationPolicy
from anemone.backup_policies.common import (
    SelectedValue,
    make_value_from_selection_and_proof,
)
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.backup_policies.explicit_tree import ExplicitTreeBackupPolicy
from anemone.backup_policies.proof import MaxProofPolicy, MinimaxProofPolicy
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluationState
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from collections.abc import Callable


def _refusal(node: AlgorithmNode[Any], reason: str) -> ValueError:
    return ValueError(f"Cannot refresh node {node.id}: {reason}.")


def validate_effective_certificate(node: AlgorithmNode[Any]) -> None:
    """Refuse contradictions or effective selection that hides a direct proof."""
    evaluation = node.tree_evaluation
    direct = evaluation.direct_value
    if not canonical_value.is_exact_value(direct):
        return
    tree = evaluation.tree_value
    if (
        canonical_value.is_exact_value(tree) and tree != direct
    ) or evaluation.effective_value != direct:
        raise _refusal(node, "direct certificate conflicts with tree/effective value")


def validate_refresh_node[NodeT: AlgorithmNode[Any]](
    node: NodeT,
    validate_direct_certificate: Callable[[NodeT], bool] | None,
) -> None:
    """Validate existing proofs and supported backup rules before any mutation."""
    evaluation = node.tree_evaluation
    direct = evaluation.direct_value
    if canonical_value.is_terminal_value(direct) and not node.state.is_game_over():
        raise _refusal(node, "TERMINAL direct value on a nonterminal state")
    if canonical_value.is_forced_value(direct) and (
        validate_direct_certificate is None or not validate_direct_certificate(node)
    ):
        raise _refusal(node, "unverified direct FORCED certificate")
    validate_effective_certificate(node)

    validate_tree_certificate(node)


def validate_tree_certificate(node: AlgorithmNode[Any]) -> None:
    """Require a tree certificate to be reconstructible under its current rule."""
    evaluation = node.tree_evaluation
    tree = evaluation.tree_value
    if not node.has_child_links() and not canonical_value.is_exact_value(tree):
        return
    if not isinstance(evaluation, NodeTreeEvaluationState):
        raise _refusal(node, "unsupported tree evaluation")
    policy = evaluation.backup_policy
    if (
        not isinstance(policy, ExplicitTreeBackupPolicy)
        or type(policy)
        not in (
            ExplicitTreeBackupPolicy,
            ExplicitMaxBackupPolicy,
            ExplicitMinimaxBackupPolicy,
        )
        # Subclasses may implement different proof rules: exact types are intentional.
        or type(policy.aggregation_policy) is not BestChildAggregationPolicy  # pylint: disable=unidiomatic-typecheck
        or type(policy.proof_policy) not in (MaxProofPolicy, MinimaxProofPolicy)
    ):
        raise _refusal(node, "unsupported backup/proof policy")
    if not canonical_value.is_exact_value(tree):
        return

    # Reuse canonical comparison and proof classification, without backing up or
    # changing branch caches. A partial max proof cannot pass this classification.
    branches = [
        branch for branch, child in node.iter_child_links() if child is not None
    ]
    best = max(
        branches, key=cmp_to_key(evaluation.compare_child_branches), default=None
    )
    selection = SelectedValue(
        value=evaluation.child_value_candidate(best) if best is not None else None,
        from_child=best is not None,
    )
    proof = policy.proof_policy.classify_selected_value(
        node_eval=evaluation, selection=selection
    )
    if make_value_from_selection_and_proof(selection=selection, proof=proof) != tree:
        raise _refusal(node, "unverifiable tree certificate under current backup rule")
