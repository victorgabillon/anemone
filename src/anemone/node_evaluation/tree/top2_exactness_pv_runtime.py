"""Conservative parent-side runtime cache for child-delta handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from valanga import BranchKey

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from collections.abc import Mapping

    from valanga.evaluations import Value

    from anemone.node_evaluation.common.node_delta import FieldChange, NodeDelta
    from anemone.node_evaluation.tree.backup_runtime_protocol import (
        BackupRuntimeNodeEvaluation,
    )


@dataclass(slots=True, frozen=True)
class _ResolvedTop2Update[BranchKeyT: BranchKey]:
    """One locally-resolved top-two cache update."""

    best_branch: BranchKeyT | None
    second_best_branch: BranchKeyT | None


@dataclass(slots=True)
class Top2ExactnessPvRuntime[BranchKeyT: BranchKey]:
    """Conservative cache for top-2, exactness, and PV-related child updates."""

    best_branch: BranchKeyT | None = None
    second_best_branch: BranchKeyT | None = None
    exact_child_count: int = 0
    selected_child_pv_version: int | None = None
    is_initialized: bool = False

    def refresh_from_node_eval(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
    ) -> None:
        """Refresh cached state from the node after a full local recompute."""
        self.best_branch = node_eval.best_branch()
        self.second_best_branch = self._current_second_best_branch(node_eval=node_eval)
        self.exact_child_count = self._count_exact_children(node_eval=node_eval)
        self.selected_child_pv_version = node_eval.pv_cached_best_child_version
        self.is_initialized = True

    def try_apply_child_deltas(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        branches_with_updated_value: set[BranchKeyT],
        branches_with_updated_best_branch_seq: set[BranchKeyT],
        child_deltas: Mapping[BranchKeyT, NodeDelta[BranchKeyT]] | None,
    ) -> BackupResult[BranchKeyT] | None:
        """Apply one conservative runtime fast path or request fallback.

        ``None`` means "fall back to the existing full local recompute path".
        """
        if child_deltas is None:
            return None

        updated_branches = (
            set(branches_with_updated_value) | set(branches_with_updated_best_branch_seq)
        )
        if not updated_branches.issubset(child_deltas):
            return None

        if not child_deltas:
            self.selected_child_pv_version = node_eval.pv_cached_best_child_version
            return self._empty_backup_result()

        if not self.is_initialized:
            return None

        value_delta_branches = {
            branch_key
            for branch_key, delta in child_deltas.items()
            if delta.value is not None
        }
        if not value_delta_branches.issubset(branches_with_updated_value):
            return None
        if len(value_delta_branches) > 1:
            return None
        if value_delta_branches:
            changed_branch = next(iter(value_delta_branches))
            return self._try_apply_single_value_delta(
                node_eval=node_eval,
                changed_branch=changed_branch,
                delta=child_deltas[changed_branch],
                branches_with_updated_value=branches_with_updated_value,
                branches_with_updated_best_branch_seq=(
                    branches_with_updated_best_branch_seq
                ),
            )

        return self._try_apply_non_value_deltas(
            node_eval=node_eval,
            child_deltas=child_deltas,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

    def _empty_backup_result(self) -> BackupResult[BranchKeyT]:
        """Return a typed empty backup result for runtime-managed no-op paths."""
        return cast(
            "BackupResult[BranchKeyT]",
            BackupResult(
                value_changed=False,
                pv_changed=False,
                over_changed=False,
            ),
        )

    def _try_apply_non_value_deltas(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        child_deltas: Mapping[BranchKeyT, NodeDelta[BranchKeyT]],
        branches_with_updated_best_branch_seq: set[BranchKeyT],
    ) -> BackupResult[BranchKeyT] | None:
        """Handle pure PV / child-observable churn without full fallback."""
        best_branch = self.best_branch
        best_delta = child_deltas.get(best_branch) if best_branch is not None else None

        for branch_key, delta in child_deltas.items():
            if (
                delta.pv_version is not None
                and branch_key not in branches_with_updated_best_branch_seq
            ):
                return None
            if (
                delta.best_branch is not None
                and branch_key not in branches_with_updated_best_branch_seq
            ):
                return None
            if (
                branch_key == best_branch
                and delta.best_branch is not None
                and delta.pv_version is None
            ):
                return None

        if best_delta is None:
            self.selected_child_pv_version = node_eval.pv_cached_best_child_version
            return self._empty_backup_result()

        if best_branch not in branches_with_updated_best_branch_seq:
            self.selected_child_pv_version = node_eval.pv_cached_best_child_version
            return self._empty_backup_result()

        if best_delta.pv_version is None and best_delta.best_branch is None:
            self.selected_child_pv_version = node_eval.pv_cached_best_child_version
            return self._empty_backup_result()

        pv_changed = node_eval.update_best_branch_sequence(
            set(branches_with_updated_best_branch_seq)
        )
        self.selected_child_pv_version = node_eval.pv_cached_best_child_version
        return BackupResult(
            value_changed=False,
            pv_changed=pv_changed,
            over_changed=False,
        )

    def _try_apply_single_value_delta(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        changed_branch: BranchKeyT,
        delta: NodeDelta[BranchKeyT],
        branches_with_updated_value: set[BranchKeyT],
        branches_with_updated_best_branch_seq: set[BranchKeyT],
    ) -> BackupResult[BranchKeyT] | None:
        """Handle one single-child value delta when top-two reasoning is sufficient."""
        if not node_eval.supports_runtime_best_child_selection():
            return None

        value_delta = delta.value
        if value_delta is None:
            return None
        if delta.pv_version is not None and changed_branch not in (
            branches_with_updated_best_branch_seq
        ):
            return None
        if delta.best_branch is not None and changed_branch not in (
            branches_with_updated_best_branch_seq
        ):
            return None

        best_branch_before_update = self.best_branch
        second_best_before_update = self.second_best_branch
        new_exact_child_count = self._exact_child_count_after_value_delta(value_delta)

        if value_delta.new is not None:
            node_eval.update_branches_values({changed_branch})

        resolved_top2 = self._resolve_single_value_delta_top2(
            node_eval=node_eval,
            changed_branch=changed_branch,
            old_best_branch=best_branch_before_update,
            old_second_best_branch=second_best_before_update,
            old_value=value_delta.old,
            new_value=value_delta.new,
        )
        if resolved_top2 is None:
            return None

        backup_result = node_eval.finalize_runtime_best_child_selection(
            best_branch_before_update=best_branch_before_update,
            exact_child_count=new_exact_child_count,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=(
                branches_with_updated_best_branch_seq
            ),
        )
        self.best_branch = resolved_top2.best_branch
        self.second_best_branch = resolved_top2.second_best_branch
        self.exact_child_count = new_exact_child_count
        self.selected_child_pv_version = node_eval.pv_cached_best_child_version
        return backup_result

    def _resolve_single_value_delta_top2(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        changed_branch: BranchKeyT,
        old_best_branch: BranchKeyT | None,
        old_second_best_branch: BranchKeyT | None,
        old_value: Value | None,
        new_value: Value | None,
    ) -> _ResolvedTop2Update[BranchKeyT] | None:
        """Resolve one single-child value delta against the cached top-two order."""
        if changed_branch == old_best_branch:
            return self._resolve_best_branch_value_delta(
                node_eval=node_eval,
                changed_branch=changed_branch,
                old_second_best_branch=old_second_best_branch,
                new_value=new_value,
            )

        if changed_branch == old_second_best_branch:
            assert old_second_best_branch is not None
            return self._resolve_second_best_value_delta(
                node_eval=node_eval,
                changed_branch=changed_branch,
                old_best_branch=old_best_branch,
                old_second_best_branch=old_second_best_branch,
                old_value=old_value,
                new_value=new_value,
            )

        return self._resolve_other_branch_value_delta(
            node_eval=node_eval,
            changed_branch=changed_branch,
            old_best_branch=old_best_branch,
            old_second_best_branch=old_second_best_branch,
            new_value=new_value,
        )

    def _resolve_best_branch_value_delta(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        changed_branch: BranchKeyT,
        old_second_best_branch: BranchKeyT | None,
        new_value: Value | None,
    ) -> _ResolvedTop2Update[BranchKeyT] | None:
        """Resolve one value change on the currently selected best branch."""
        if old_second_best_branch is None:
            if new_value is None:
                return _ResolvedTop2Update(best_branch=None, second_best_branch=None)
            return _ResolvedTop2Update(
                best_branch=changed_branch,
                second_best_branch=None,
            )

        if new_value is None:
            return None

        if (
            node_eval.compare_child_branches(changed_branch, old_second_best_branch)
            > 0
        ):
            return _ResolvedTop2Update(
                best_branch=changed_branch,
                second_best_branch=old_second_best_branch,
            )
        return None

    def _resolve_second_best_value_delta(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        changed_branch: BranchKeyT,
        old_best_branch: BranchKeyT | None,
        old_second_best_branch: BranchKeyT,
        old_value: Value | None,
        new_value: Value | None,
    ) -> _ResolvedTop2Update[BranchKeyT] | None:
        """Resolve one value change on the cached second-best branch."""
        if old_best_branch is None:
            return None

        if new_value is None:
            return None

        if node_eval.compare_child_branches(changed_branch, old_best_branch) > 0:
            return _ResolvedTop2Update(
                best_branch=changed_branch,
                second_best_branch=old_best_branch,
            )

        if self._compare_value_change(
            node_eval=node_eval,
            old_value=old_value,
            new_value=new_value,
        ) >= 0:
            return _ResolvedTop2Update(
                best_branch=old_best_branch,
                second_best_branch=old_second_best_branch,
            )
        return None

    def _resolve_other_branch_value_delta(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        changed_branch: BranchKeyT,
        old_best_branch: BranchKeyT | None,
        old_second_best_branch: BranchKeyT | None,
        new_value: Value | None,
    ) -> _ResolvedTop2Update[BranchKeyT] | None:
        """Resolve one value change on a branch outside the cached top-two."""
        if old_best_branch is None:
            if new_value is None:
                return _ResolvedTop2Update(best_branch=None, second_best_branch=None)
            return _ResolvedTop2Update(
                best_branch=changed_branch,
                second_best_branch=None,
            )

        if new_value is None:
            return _ResolvedTop2Update(
                best_branch=old_best_branch,
                second_best_branch=old_second_best_branch,
            )

        if node_eval.compare_child_branches(changed_branch, old_best_branch) > 0:
            return _ResolvedTop2Update(
                best_branch=changed_branch,
                second_best_branch=old_best_branch,
            )

        if old_second_best_branch is None:
            return _ResolvedTop2Update(
                best_branch=old_best_branch,
                second_best_branch=changed_branch,
            )

        if (
            node_eval.compare_child_branches(changed_branch, old_second_best_branch)
            > 0
        ):
            return _ResolvedTop2Update(
                best_branch=old_best_branch,
                second_best_branch=changed_branch,
            )

        return _ResolvedTop2Update(
            best_branch=old_best_branch,
            second_best_branch=old_second_best_branch,
        )

    def _compare_value_change(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        old_value: Value | None,
        new_value: Value | None,
    ) -> int:
        """Return whether one single-branch value change improved or worsened."""
        if old_value is None:
            return 0 if new_value is None else 1
        if new_value is None:
            return -1
        return node_eval.compare_candidate_values(new_value, old_value)

    def _exact_child_count_after_value_delta(
        self,
        value_delta: FieldChange[Value | None],
    ) -> int:
        """Update the cached exact-child count from one field-level value delta."""
        exact_child_count = self.exact_child_count
        if canonical_value.is_exact_value(value_delta.old):
            exact_child_count -= 1
        if canonical_value.is_exact_value(value_delta.new):
            exact_child_count += 1
        return exact_child_count

    def _current_second_best_branch(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
    ) -> BranchKeyT | None:
        """Return the current second-best branch when one exists."""
        try:
            return node_eval.second_best_branch()
        except RuntimeError:
            return None

    def _count_exact_children(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
    ) -> int:
        """Count child branches whose current canonical value is exact."""
        exact_child_count = 0
        for branch_key, child in node_eval.tree_node.branches_children.items():
            if child is None:
                continue
            if canonical_value.is_exact_value(node_eval.child_value_candidate(branch_key)):
                exact_child_count += 1
        return exact_child_count
