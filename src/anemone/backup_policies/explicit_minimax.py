"""Explicit minimax backup policy with separated value/PV responsibilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.types import BackupResult
from anemone.utils.my_value_sorted_dict import sort_dic

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )
    from anemone.values import Value


class ExplicitMinimaxBackupPolicy:
    """Backup policy that computes minimax value/PV/over explicitly from Value."""

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute value/PV/over from updated children and return change flags."""
        assert (
            not node_eval.tree_node.all_branches_generated
            or node_eval.tree_node.branches_children
        ), "Cannot compute minimax value: no children."

        value_before_update = node_eval.minmax_value
        pv_before_update = node_eval.best_branch_sequence.copy()
        over_before_update = node_eval.over_event
        best_branch_before_update = node_eval.best_branch()

        if branches_with_updated_value:
            self._update_branches_values_value_only(
                node_eval=node_eval,
                branches_to_consider=branches_with_updated_value,
            )

        self._update_value_value_only(node_eval=node_eval)
        best_branch_after_update = node_eval.best_branch()

        if best_branch_after_update is None:
            node_eval.clear_best_branch_sequence()
        elif node_eval.tree_node.all_branches_generated:
            should_rebuild = (
                best_branch_before_update != best_branch_after_update
                or not node_eval.best_branch_sequence
                or node_eval.best_branch_sequence[0] != best_branch_after_update
                or best_branch_after_update in branches_with_updated_best_branch_seq
            )
            if should_rebuild:
                node_eval.one_of_best_children_becomes_best_next_node()
        else:
            direct_value = self._direct_value_candidate(node_eval)
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )
            best_child_value = node_eval.child_value_candidate(best_branch_after_update)
            assert best_child_value is not None
            if node_eval.child_is_better_than_direct(
                best_child_value,
                direct_value,
                side_to_move=node_eval.tree_node.state.turn,
            ):
                should_rebuild = (
                    best_branch_before_update != best_branch_after_update
                    or not node_eval.best_branch_sequence
                    or node_eval.best_branch_sequence[0] != best_branch_after_update
                    or best_branch_after_update in branches_with_updated_best_branch_seq
                )
                if should_rebuild:
                    node_eval.one_of_best_children_becomes_best_next_node()
            else:
                node_eval.clear_best_branch_sequence()

        if (
            best_branch_after_update is not None
            and node_eval.best_branch_sequence
            and node_eval.best_branch_sequence[0] == best_branch_after_update
            and branches_with_updated_best_branch_seq
        ):
            node_eval.update_best_branch_sequence(branches_with_updated_best_branch_seq)

        node_eval.sync_float_views_from_values()

        value_changed = has_value_changed(
            value_before=value_before_update,
            value_after=node_eval.minmax_value,
        )
        pv_changed = pv_before_update != node_eval.best_branch_sequence
        over_changed = over_before_update != node_eval.over_event

        return BackupResult(
            value_changed=value_changed,
            pv_changed=pv_changed,
            over_changed=over_changed,
        )

    def _update_value_value_only(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> None:
        """Update ``minmax_value`` from Value-first branch ordering semantics."""
        best_branch_key = node_eval.best_branch()
        if best_branch_key is None and node_eval.tree_node.branches_children:
            self._update_branches_values_value_only(
                node_eval=node_eval,
                branches_to_consider=set(node_eval.tree_node.branches_children),
            )
            best_branch_key = node_eval.best_branch()

        if best_branch_key is None:
            assert node_eval.tree_node.branches_children, (
                "Cannot compute minimax value: no children."
            )
            direct_value = self._direct_value_candidate(node_eval)
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )
            node_eval.minmax_value = direct_value
            return

        best_child_value = node_eval.child_value_candidate(best_branch_key)
        assert best_child_value is not None

        if node_eval.tree_node.all_branches_generated:
            value_after_update = best_child_value
        else:
            direct_value = self._direct_value_candidate(node_eval)
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )
            if node_eval.child_is_better_than_direct(
                best_child_value,
                direct_value,
                side_to_move=node_eval.tree_node.state.turn,
            ):
                value_after_update = best_child_value
            else:
                value_after_update = direct_value

        node_eval.minmax_value = value_after_update

    def _update_branches_values_value_only(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_to_consider: set[BranchKey],
    ) -> None:
        """Update branch ordering keys using Value semantics instead of float-only keys."""
        for branch_key in branches_to_consider:
            child = node_eval.tree_node.branches_children[branch_key]
            assert child is not None
            child_value = node_eval.child_value_candidate(branch_key)
            if child_value is None:
                continue

            subjective_sort_value = node_eval.evaluation_ordering.search_sort_key(
                child_value,
                side_to_move=node_eval.tree_node.state.turn,
            )

            if node_eval.is_terminal_candidate():
                node_eval.branches_sorted_by_value_[branch_key] = (
                    subjective_sort_value,
                    -len(child.tree_evaluation.best_branch_sequence),
                    child.tree_node.id,
                )
            else:
                node_eval.branches_sorted_by_value_[branch_key] = (
                    subjective_sort_value,
                    len(child.tree_evaluation.best_branch_sequence),
                    child.tree_node.id,
                )

        node_eval.branches_sorted_by_value_ = sort_dic(
            node_eval.branches_sorted_by_value_
        )

    def _direct_value_candidate(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> Value | None:
        return node_eval.direct_value

    def _is_child_value_better_than_direct_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        child_value: Value | None,
    ) -> bool:
        """Return whether child value dominates direct eval for the side to move."""
        if child_value is None:
            return False
        direct_value = self._direct_value_candidate(node_eval)
        assert direct_value is not None, (
            "Explicit minimax requires direct_value for partially expanded nodes."
        )
        return node_eval.child_is_better_than_direct(
            child_value,
            direct_value,
            side_to_move=node_eval.tree_node.state.turn,
        )


def has_value_changed(*, value_before: Value | None, value_after: Value | None) -> bool:
    """Return whether score/certainty/over metadata changed between two values."""
    if value_before is None or value_after is None:
        return value_before != value_after
    return (
        value_before.score != value_after.score
        or value_before.certainty != value_after.certainty
        or value_before.over_event != value_after.over_event
    )
