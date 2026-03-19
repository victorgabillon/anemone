"""Compatibility aliases for the shared decision-ordering module."""

from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
    BranchOrderingKeyGetter,
    ChildValueCandidateGetter,
    DecisionOrderingState,
    SemanticCompare,
    make_branch_ordering_keys_factory,
)

BranchSortValue = BranchOrderingKey
BranchSortValueGetter = BranchOrderingKeyGetter
MinmaxDecisionOrderingState = DecisionOrderingState
make_branches_sorted_by_value_factory = make_branch_ordering_keys_factory

__all__ = [
    "BranchSortValue",
    "BranchSortValueGetter",
    "ChildValueCandidateGetter",
    "MinmaxDecisionOrderingState",
    "SemanticCompare",
    "make_branches_sorted_by_value_factory",
]
