"""Small shared helpers for explicit backup policy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from valanga import BranchKey, OverEvent
    from valanga.evaluations import Value


@dataclass(frozen=True, slots=True)
class SelectedValue:
    """One value candidate plus whether it was chosen from a child branch."""

    value: Value | None
    from_child: bool


class ChildValueReader(Protocol):
    """Protocol for node-evaluation objects that can expose child values."""

    tree_node: Any

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the candidate value for one child branch."""


def all_child_values_exact(node_eval: ChildValueReader) -> bool:
    """Return True when every existing child branch has an exact Value."""
    if not node_eval.tree_node.branches_children:
        return False
    for branch_key in node_eval.tree_node.branches_children:
        child_value = node_eval.child_value_candidate(branch_key)
        if not canonical_value.is_exact_value(child_value):
            return False
    return True


def has_value_changed(*, value_before: Value | None, value_after: Value | None) -> bool:
    """Return whether effective value semantics changed between two values.

    Principal-variation or line changes are intentionally excluded here and are
    reported separately through ``pv_changed``.
    """
    if value_before is None or value_after is None:
        return value_before != value_after
    return (
        value_before.score != value_after.score
        or value_before.certainty != value_after.certainty
        or value_before.over_event != value_after.over_event
    )


class BackupPipelineNode(Protocol):
    """Protocol for evaluations that participate in the shared backup pipeline."""

    @property
    def backed_up_value(self) -> Value | None:
        """Return the node's current generic backed-up value."""
        ...

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Store the node's current generic backed-up value."""
        ...

    @property
    def best_branch_sequence(self) -> Sequence[BranchKey]:
        """Return the current principal variation as read-only sequence data."""
        ...

    @property
    def over_event(self) -> OverEvent | None:
        """Return exact-outcome metadata derived from the current effective value."""
        ...

    def sync_branch_frontier(self, branches_to_refresh: set[BranchKey]) -> None:
        """Refresh frontier bookkeeping after a backup."""
        ...


@dataclass(frozen=True, slots=True)
class BackupSnapshot:
    """Pre-backup state captured for generic change reporting."""

    value_before: Value | None
    pv_before: tuple[BranchKey, ...]
    over_before: OverEvent | None

    @classmethod
    def capture(cls, node_eval: BackupPipelineNode) -> BackupSnapshot:
        """Capture the generic state touched by all explicit backup policies."""
        return cls(
            value_before=node_eval.backed_up_value,
            pv_before=tuple(node_eval.best_branch_sequence),
            over_before=node_eval.over_event,
        )


def run_backup_pipeline(
    *,
    node_eval: BackupPipelineNode,
    value_after: Value | None,
    branches_with_updated_value: set[BranchKey],
    update_pv: Callable[[], bool],
) -> BackupResult:
    """Run the shared backup orchestration around family-specific selection logic."""
    snapshot = BackupSnapshot.capture(node_eval)

    node_eval.backed_up_value = value_after
    node_eval.sync_branch_frontier(branches_with_updated_value)
    pv_changed = update_pv()

    return BackupResult(
        value_changed=has_value_changed(
            value_before=snapshot.value_before,
            value_after=node_eval.backed_up_value,
        ),
        pv_changed=pv_changed
        or snapshot.pv_before != tuple(node_eval.best_branch_sequence),
        over_changed=snapshot.over_before != node_eval.over_event,
    )
