"""Small shared helpers for explicit backup policy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, assert_never

from valanga.evaluations import Certainty

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone._valanga_types import AnyOverEvent


@dataclass(frozen=True, slots=True)
class SelectedValue:
    """One candidate for ``backed_up_value`` plus whether it came from a child."""

    value: Value | None
    from_child: bool


@dataclass(frozen=True, slots=True)
class ProofClassification:
    """Exactness/outcome classification for the resulting parent value."""

    certainty: Certainty
    over_event: AnyOverEvent | None

    @classmethod
    def from_value(cls, value: Value) -> ProofClassification:
        """Mirror the certainty/outcome already carried by an existing Value."""
        return cls(
            certainty=value.certainty,
            over_event=value.over_event,
        )


class ChildValueReader(Protocol):
    """Protocol for node-evaluation objects that can expose child values."""

    tree_node: Any

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the candidate value for one child branch."""


def _missing_terminal_over_event_error() -> ValueError:
    return ValueError(
        "Cannot construct a TERMINAL Value without an over_event in "
        "ProofClassification."
    )


def all_child_values_exact(node_eval: ChildValueReader) -> bool:
    """Return True when every existing child branch has an exact Value."""
    if not node_eval.tree_node.branches_children:
        return False
    for branch_key in node_eval.tree_node.branches_children:
        child_value = node_eval.child_value_candidate(branch_key)
        if not canonical_value.is_exact_value(child_value):
            return False
    return True


def select_value_from_best_child_and_direct(
    *,
    best_child_value: Value | None,
    direct_value: Value | None,
    all_branches_generated: bool,
    child_beats_direct: Callable[[Value, Value], bool],
) -> SelectedValue:
    """Select the winner of one hybrid best-child-versus-direct comparison.

    This is a special helper for policies that intentionally mix direct and
    child-derived estimates. It is not the default aggregation shape: the shared
    default aggregation now computes only child/tree-derived ``backed_up_value``
    candidates and leaves canonical direct fallback to ``canonical_value``.
    """
    if best_child_value is None:
        return SelectedValue(value=direct_value, from_child=False)
    if direct_value is None:
        return SelectedValue(value=best_child_value, from_child=True)
    if all_branches_generated:
        return SelectedValue(value=best_child_value, from_child=True)
    if child_beats_direct(best_child_value, direct_value):
        return SelectedValue(value=best_child_value, from_child=True)
    return SelectedValue(value=direct_value, from_child=False)


def has_value_changed(*, value_before: Value | None, value_after: Value | None) -> bool:
    """Return whether effective value semantics changed between two values.

    Principal-variation or line changes are intentionally excluded here and are
    reported separately through ``pv_changed``.
    """
    if value_before is None or value_after is None:
        return value_before != value_after
    return bool(
        value_before.score != value_after.score
        or value_before.certainty != value_after.certainty
        or value_before.over_event != value_after.over_event
    )


def make_value_from_proof_classification(
    *,
    score: float,
    proof: ProofClassification,
) -> Value:
    """Construct a Value from one score and one proof/exactness classification."""
    if proof.certainty == Certainty.ESTIMATE:
        return canonical_value.make_estimate_value(score=score)
    if proof.certainty == Certainty.FORCED:
        return canonical_value.make_forced_value(
            score=score,
            over_event=proof.over_event,
        )
    if proof.certainty == Certainty.TERMINAL:
        over_event = proof.over_event
        if over_event is None:
            raise _missing_terminal_over_event_error()
        return canonical_value.make_terminal_value(
            score=score,
            over_event=over_event,
        )
    assert_never(proof.certainty)


def make_value_from_selection_and_proof(
    *,
    selection: SelectedValue,
    proof: ProofClassification | None,
) -> Value | None:
    """Construct the resulting parent value from selection plus proof classification."""
    chosen_value = selection.value
    if chosen_value is None or proof is None:
        return None
    return make_value_from_proof_classification(
        score=chosen_value.score,
        proof=proof,
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
    def over_event(self) -> AnyOverEvent | None:
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
    over_before: AnyOverEvent | None

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
) -> BackupResult[BranchKey]:
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


def finalize_selection_with_proof(
    *,
    node_eval: BackupPipelineNode,
    selection: SelectedValue,
    proof: ProofClassification | None,
    branches_with_updated_value: set[BranchKey],
    update_pv: Callable[[], bool],
) -> BackupResult[BranchKey]:
    """Finalize one selected candidate once proof classification is known."""
    value_after = make_value_from_selection_and_proof(
        selection=selection,
        proof=proof,
    )
    return run_backup_pipeline(
        node_eval=node_eval,
        value_after=value_after,
        branches_with_updated_value=branches_with_updated_value,
        update_pv=update_pv,
    )
