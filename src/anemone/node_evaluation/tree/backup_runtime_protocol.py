"""Protocol for specialized parent-side backup runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from valanga import BranchKey

if TYPE_CHECKING:
    from collections.abc import Mapping

    from valanga.evaluations import Value

    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.common.node_delta import NodeDelta


class BackupRuntimeNodeEvaluation[BranchKeyT: BranchKey](Protocol):
    """Minimal node-evaluation surface shared by parent backup runtimes."""

    tree_node: Any

    @property
    def pv_cached_best_child_version(self) -> int | None:
        """Return the stored PV version for the currently selected child."""
        ...

    def best_branch(self) -> BranchKeyT | None:
        """Return the current best child branch."""
        ...

    def second_best_branch(self) -> BranchKeyT:
        """Return the current second-best child branch."""
        ...

    def child_value_candidate(self, branch_key: BranchKeyT) -> Value | None:
        """Return the current canonical child value for one branch."""
        ...

    def compare_candidate_values(self, left: Value, right: Value) -> int:
        """Compare two candidate values under this node's current semantics."""
        ...

    def compare_child_branches(
        self,
        left_branch: BranchKeyT,
        right_branch: BranchKeyT,
    ) -> int:
        """Compare two child branches under current node-local ordering semantics."""
        ...

    def update_branches_values(self, branches_to_consider: set[BranchKeyT]) -> None:
        """Refresh ordering keys for the provided child branches."""
        ...

    def update_best_branch_sequence(
        self,
        branches_with_updated_best_branch_seq: set[BranchKeyT],
    ) -> bool:
        """Refresh the stored PV tail when the selected child changed."""
        ...

    def supports_runtime_best_child_selection(self) -> bool:
        """Return whether runtime best-child value updates preserve policy semantics."""
        ...

    def finalize_runtime_best_child_selection(
        self,
        *,
        best_branch_before_update: BranchKeyT | None,
        exact_child_count: int,
        branches_with_updated_value: set[BranchKeyT],
        branches_with_updated_best_branch_seq: set[BranchKeyT],
    ) -> BackupResult[BranchKeyT]:
        """Finalize one runtime-managed best-child selection update."""
        ...


class BackupRuntime[BranchKeyT: BranchKey](Protocol):
    """Specialized parent-side runtime attached to a node tree evaluation."""

    def refresh_from_node_eval(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
    ) -> None:
        """Refresh runtime caches from the current node state."""
        ...

    def try_apply_child_deltas(
        self,
        *,
        node_eval: BackupRuntimeNodeEvaluation[BranchKeyT],
        branches_with_updated_value: set[BranchKeyT],
        branches_with_updated_best_branch_seq: set[BranchKeyT],
        child_deltas: Mapping[BranchKeyT, NodeDelta[BranchKeyT]] | None,
    ) -> BackupResult[BranchKeyT] | None:
        """Return a runtime-managed result, or None to request fallback."""
        ...
