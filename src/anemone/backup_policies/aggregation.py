"""Aggregation policies for selecting parent candidates during backup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from anemone.backup_policies.common import (
    SelectedValue,
    select_value_from_best_child_and_direct,
)

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )
    from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
        NodeMaxEvaluation,
    )


class AggregationPolicy[NodeEvalT](Protocol):
    """Protocol for selecting the winning parent candidate value.

    Aggregation policies choose only which Value candidate wins for the parent.
    They must remain compatible with each family's existing best-branch and PV
    semantics, which still live outside this policy boundary.
    """

    def select_value(
        self,
        *,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        """Return the selected child/direct candidate for one node backup."""


class MaxAggregationPolicy:
    """Aggregation policy for single-agent max candidate selection."""

    def select_value(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        """Select the parent candidate using single-agent max semantics."""
        del branches_with_updated_value
        best_branch_key = node_eval.best_branch()
        best_child_value = (
            node_eval.child_value_candidate(best_branch_key)
            if best_branch_key is not None
            else None
        )
        return select_value_from_best_child_and_direct(
            best_child_value=best_child_value,
            direct_value=node_eval.direct_value,
            all_branches_generated=node_eval.tree_node.all_branches_generated,
            child_beats_direct=lambda child, direct: (
                node_eval.objective.semantic_compare(
                    child,
                    direct,
                    node_eval.tree_node.state,
                )
                >= 0
            ),
        )


class MinimaxAggregationPolicy:
    """Aggregation policy for adversarial minimax candidate selection."""

    def select_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        """Select the parent candidate using minimax ordering semantics."""
        if branches_with_updated_value:
            self._update_branch_ordering(
                node_eval=node_eval,
                branches_to_consider=branches_with_updated_value,
            )

        best_child_value = self._best_child_value_for_selection(node_eval=node_eval)
        if best_child_value is None:
            assert node_eval.tree_node.branches_children, (
                "Cannot compute minimax value: no children."
            )
            direct_value = node_eval.direct_value
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )
            return SelectedValue(value=direct_value, from_child=False)

        direct_value = node_eval.direct_value
        if not node_eval.tree_node.all_branches_generated:
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )

        return select_value_from_best_child_and_direct(
            best_child_value=best_child_value,
            direct_value=direct_value,
            all_branches_generated=node_eval.tree_node.all_branches_generated,
            child_beats_direct=lambda child, direct: (
                node_eval.child_is_better_than_direct(
                    child,
                    direct,
                    side_to_move=node_eval.tree_node.state.turn,
                )
            ),
        )

    def _best_child_value_for_selection(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> Value | None:
        """Return the current best child candidate, refreshing ordering if needed."""
        best_branch_key = node_eval.best_branch()
        if best_branch_key is None and node_eval.tree_node.branches_children:
            self._update_branch_ordering(
                node_eval=node_eval,
                branches_to_consider=set(node_eval.tree_node.branches_children),
            )
            best_branch_key = node_eval.best_branch()

        if best_branch_key is None:
            return None

        best_child_value = node_eval.child_value_candidate(best_branch_key)
        assert best_child_value is not None
        return best_child_value

    def _update_branch_ordering(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_to_consider: set[BranchKey],
    ) -> None:
        """Refresh minimax branch ordering from the currently-known child values."""
        node_eval.update_branches_values(
            {
                branch_key
                for branch_key in branches_to_consider
                if node_eval.child_value_candidate(branch_key) is not None
            }
        )
