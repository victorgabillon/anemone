"""Aggregation policies for computing tree-derived backup candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anemone.backup_policies.common import (
    SelectedValue,
)

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value


class AggregationPolicy[NodeEvalT](Protocol):
    """Protocol for selecting one tree-derived candidate for ``backed_up_value``.

    Aggregation policies compute only the backup candidate derived from child or
    subtree information. They do not define the node's canonical value; the
    canonical-value helpers still prefer ``backed_up_value`` over ``direct_value``
    when a backed-up candidate exists.
    """

    def select_value(
        self,
        *,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        """Return one candidate for ``backed_up_value`` for one node backup."""
        ...


class BestChildAggregationSource(Protocol):
    """Minimal node-evaluation surface needed by the shared best-child policy."""

    def best_branch(self) -> BranchKey | None:
        """Return the current best child branch."""
        ...

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the current candidate value for one child branch."""
        ...


class BranchOrderingPreparationSource(Protocol):
    """Minimal surface for refreshing child ordering before aggregation."""

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the current candidate value for one child branch."""
        ...

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Refresh ordering keys for the provided child branches."""
        ...


def prepare_best_child_aggregation(
    *,
    node_eval: BranchOrderingPreparationSource,
    branches_to_consider: set[BranchKey],
) -> None:
    """Refresh ordering keys for child branches that currently expose a value."""
    if not branches_to_consider:
        return

    node_eval.update_branches_values(
        {
            branch_key
            for branch_key in branches_to_consider
            if node_eval.child_value_candidate(branch_key) is not None
        }
    )


class BestChildAggregationPolicy[NodeEvalT: BestChildAggregationSource]:
    """Aggregation policy that returns the current best known child candidate."""

    def select_value(
        self,
        *,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
    ) -> SelectedValue:
        """Return the best currently-known child-derived backup candidate."""
        del branches_with_updated_value
        best_branch_key = node_eval.best_branch()
        if best_branch_key is None:
            return SelectedValue(value=None, from_child=False)

        best_child_value = node_eval.child_value_candidate(best_branch_key)
        if best_child_value is None:
            return SelectedValue(value=None, from_child=False)
        return SelectedValue(value=best_child_value, from_child=True)
