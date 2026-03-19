"""Provide shared tree-search evaluation state plus the public protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import log
from typing import TYPE_CHECKING, Any, Protocol, assert_never

from valanga import BranchKey, OverEvent, State

from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
from anemone.node_evaluation.common.branch_ordering import DecisionOrderedEvaluation
from anemone.node_evaluation.common.node_value_evaluation import NodeValueEvaluation
from anemone.node_evaluation.common.principal_variation import (
    PrincipalVariationState,
)
from anemone.nodes.itree_node import ITreeNode
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga.evaluations import Value

    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.tree.decision_ordering import (
        BranchOrderingKey,
        DecisionOrderingState,
    )
    from anemone.nodes.tree_node import TreeNode


def make_branch_frontier_factory() -> BranchFrontierState:
    """Create the generic frontier bookkeeping helper for one node."""
    return BranchFrontierState()


def make_principal_variation_state_factory() -> PrincipalVariationState:
    """Create the generic principal-variation bookkeeping helper for one node."""
    return PrincipalVariationState()


class BestBranchEquivalenceMode(StrEnum):
    """Modes for collecting branches considered equivalent to the best branch."""

    EQUAL = "equal"
    CONSIDERED_EQUAL = "considered_equal"
    ALMOST_EQUAL = "almost_equal"
    ALMOST_EQUAL_LOGISTIC = "almost_equal_logistic"


class TreeEvaluationChild[StateT: State = State](ITreeNode[StateT], Protocol):
    """Minimal child-node protocol needed by the shared tree-evaluation state."""

    @property
    def tree_evaluation(self) -> ChildTreeEvaluation:
        """Return the child node's tree evaluation."""
        ...


class ChildTreeEvaluation(Protocol):
    """Minimal child-evaluation protocol needed by shared tree-state helpers."""

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the child's principal-variation branch sequence."""
        ...

    def get_value_candidate(self) -> Value | None:
        """Return the child's current best available candidate Value."""
        ...

    def has_exact_value(self) -> bool:
        """Return whether the child currently has an exact candidate Value."""
        ...


@dataclass(slots=True)
class NodeTreeEvaluationState[
    NodeT: TreeEvaluationChild[Any] = TreeEvaluationChild[Any],
    StateT: State = State,
]:
    """Shared concrete state/helper base for tree-evaluation families.

    This class owns only the family-neutral state and helper methods that both
    current tree-evaluation families already share. Decision ordering,
    family-specific backup entrypoints, and the value semantics behind
    best-branch equivalence remain in the concrete families.
    """

    tree_node: TreeNode[NodeT, StateT]
    direct_value: Value | None = None
    _backed_up_value: Value | None = None
    pv_state: PrincipalVariationState = field(
        default_factory=make_principal_variation_state_factory
    )
    branch_frontier: BranchFrontierState = field(
        default_factory=make_branch_frontier_factory
    )

    @property
    def backed_up_value(self) -> Value | None:
        """Return the canonical backed-up value for this node."""
        return self._backed_up_value

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Set the canonical backed-up value for this node."""
        self._backed_up_value = value

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the current principal-variation branch sequence."""
        return self.pv_state.best_branch_sequence

    @property
    def pv_version(self) -> int:
        """Return the version of the current PV content."""
        return self.pv_state.pv_version

    @property
    def pv_cached_best_child_version(self) -> int | None:
        """Cached pv_version of the best child when PV was last materialized."""
        return self.pv_state.cached_best_child_version

    def get_value_candidate(self) -> Value | None:
        """Return backed-up value when available, else direct value."""
        return canonical_value.get_value_candidate(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_value(self) -> Value:
        """Return the canonical Value for this node."""
        return canonical_value.get_value(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_score(self) -> float:
        """Return the canonical scalar score for this node."""
        return canonical_value.get_score(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def has_exact_value(self) -> bool:
        """Return True when the candidate Value is exact."""
        return canonical_value.is_exact_value(self.get_value_candidate())

    def is_terminal(self) -> bool:
        """Return True when the candidate Value says this node's own state is terminal."""
        return canonical_value.is_terminal_value(self.get_value_candidate())

    def has_over_event(self) -> bool:
        """Return True when the candidate Value carries exact outcome metadata."""
        return canonical_value.has_over_event(self.get_value_candidate())

    def get_over_event_candidate(self) -> OverEvent | None:
        """Return exact outcome metadata from the candidate Value when present."""
        return canonical_value.get_over_event_candidate(self.get_value_candidate())

    @property
    def over_event(self) -> OverEvent | None:
        """Return exact outcome metadata when present on the canonical candidate."""
        return self.get_over_event_candidate()

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the best available Value candidate for a child branch."""
        child_tree_evaluation = self.child_tree_evaluation(branch_key)
        if child_tree_evaluation is None:
            return None
        return child_tree_evaluation.get_value_candidate()

    def child_tree_evaluation(
        self,
        branch_key: BranchKey,
    ) -> ChildTreeEvaluation | None:
        """Return one child evaluation when the branch currently exists."""
        child = self.tree_node.branches_children.get(branch_key)
        if child is None:
            return None
        return child.tree_evaluation

    def best_branch_line_from_child(self, branch_key: BranchKey) -> list[BranchKey]:
        """Return the PV line that starts by taking one concrete child branch."""
        child_tree_evaluation = self.child_tree_evaluation(branch_key)
        assert child_tree_evaluation is not None, (
            f"Cannot build PV line: missing child for branch {branch_key!r}."
        )
        return [branch_key, *child_tree_evaluation.best_branch_sequence]

    def on_branch_opened(self, branch: BranchKey) -> None:
        """Record that a child branch has entered the frontier."""
        self.branch_frontier.on_branch_opened(branch)

    def has_frontier_branches(self) -> bool:
        """Return whether some child branches remain search-relevant."""
        return self.branch_frontier.has_frontier_branches()

    def ordered_frontier_branches_from(
        self,
        ordered_candidate_branches: Iterable[BranchKey],
    ) -> list[BranchKey]:
        """Project frontier membership onto one caller-supplied branch ordering."""
        return self.branch_frontier.ordered_frontier_branches(
            ordered_candidate_branches
        )

    def _branch_is_frontier_relevant(self, branch_key: BranchKey) -> bool:
        """Return whether a child branch can still affect future search results."""
        child_tree_evaluation = self.child_tree_evaluation(branch_key)
        return (
            child_tree_evaluation is not None
            and not child_tree_evaluation.has_exact_value()
        )

    def _child_pv_version(self, child: Any) -> int:
        """Return one child's PV version with a conservative fallback."""
        return int(getattr(child.tree_evaluation, "pv_version", 0))

    def _child_best_branch_sequence(self, child: Any) -> list[BranchKey]:
        """Return one child's current PV sequence."""
        return list(child.tree_evaluation.best_branch_sequence)

    def _pv_child_version_for_sequence(self, sequence: list[BranchKey]) -> int | None:
        """Return the cached best-child PV version for a given PV sequence head."""
        if not sequence:
            return None

        best_child_tree_evaluation = self.child_tree_evaluation(sequence[0])
        if best_child_tree_evaluation is None:
            return None

        return int(getattr(best_child_tree_evaluation, "pv_version", 0))

    def sync_branch_frontier(self, branches_to_refresh: set[BranchKey]) -> None:
        """Refresh frontier bookkeeping from updated child values."""
        if self.has_exact_value():
            self.branch_frontier.clear()
            return

        self.branch_frontier.sync_with_current_state(
            branches_to_refresh=branches_to_refresh,
            should_remain_in_frontier=self._branch_is_frontier_relevant,
        )

    def frontier_branches_in_order(self) -> list[BranchKey]:
        """Return frontier branches ordered by family-defined search semantics."""
        return self.ordered_frontier_branches_from(
            self._ordered_candidate_branches_for_frontier()
        )

    def _ordered_candidate_branches_for_frontier(self) -> Iterable[BranchKey]:
        """Return candidate branches in the family's current search order."""
        raise NotImplementedError

    def _ensure_decision_ordering_ready(self) -> None:
        """Ensure the family's cached decision ordering is ready to consume."""
        raise NotImplementedError

    def _decision_ordering_state(self) -> DecisionOrderingState:
        """Return the family's shared decision-ordering state helper."""
        raise NotImplementedError

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using the family's current decision semantics."""
        raise NotImplementedError

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by current family decision semantics."""
        self._ensure_decision_ordering_ready()
        return self._decision_ordering_state().decision_ordered_branches(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch."""
        self._ensure_decision_ordering_ready()
        return self._decision_ordering_state().best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def second_best_branch(self) -> BranchKey:
        """Return the current second-best branch."""
        self._ensure_decision_ordering_ready()
        return self._decision_ordering_state().second_best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def best_equivalent_branches(
        self,
        mode: BestBranchEquivalenceMode = BestBranchEquivalenceMode.EQUAL,
    ) -> list[BranchKey]:
        """Return branches equivalent to the current best branch under one mode."""
        best_branch_key = self.best_branch()
        if best_branch_key is None:
            return []

        return [
            branch
            for branch in self._ordered_candidate_branches_for_best_equivalence()
            if self._branch_is_equivalent_to_best(
                branch=branch,
                best_branch=best_branch_key,
                mode=mode,
            )
        ]

    def _ordered_candidate_branches_for_best_equivalence(
        self,
    ) -> tuple[BranchKey, ...]:
        """Return candidate branches in family-defined best-equivalence order."""
        return tuple(self.decision_ordered_branches())

    def _branch_ordering_key(self, branch: BranchKey) -> BranchOrderingKey:
        """Return the shared branch-ordering key for one branch."""
        return self._decision_ordering_state().branch_ordering_keys[branch]

    def _branch_is_equivalent_to_best(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
        mode: BestBranchEquivalenceMode,
    ) -> bool:
        """Return whether one branch is equivalent to the best branch under one mode."""
        if mode is BestBranchEquivalenceMode.EQUAL:
            return self._branch_values_are_equal(
                branch=branch,
                best_branch=best_branch,
            )
        if mode is BestBranchEquivalenceMode.CONSIDERED_EQUAL:
            return (
                self._branch_ordering_key(branch)[:2]
                == self._branch_ordering_key(best_branch)[:2]
            )
        if mode is BestBranchEquivalenceMode.ALMOST_EQUAL:
            return self._are_almost_equal_scores(
                self._branch_equivalence_score(branch),
                self._branch_equivalence_score(best_branch),
            )
        if mode is BestBranchEquivalenceMode.ALMOST_EQUAL_LOGISTIC:
            return self._are_almost_equal_scores(
                self._branch_logistic_equivalence_score(branch),
                self._branch_logistic_equivalence_score(best_branch),
            )
        assert_never(mode)

    def _branch_values_are_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two minimax branch ordering keys are exactly equal."""
        return self._branch_ordering_key(branch) == self._branch_ordering_key(
            best_branch
        )

    def _branch_equivalence_score(self, branch: BranchKey) -> float:
        """Return the family's primary scalar score for branch equivalence."""
        return self._branch_ordering_key(branch)[0]

    def _branch_logistic_equivalence_score(self, branch: BranchKey) -> float:
        """Return the score used by logistic-style best-branch equivalence."""
        return self._logistic_equivalence_score(self._branch_equivalence_score(branch))

    def _are_almost_equal_scores(self, left: float, right: float) -> bool:
        """Return whether two scalar branch-equivalence scores are close enough."""
        epsilon = 0.01
        return left > right - epsilon and right > left - epsilon

    def _logistic_equivalence_score(self, x: float) -> float:
        """Apply the shared logistic-style transform used for branch equivalence."""
        y = min(max(x * 0.5 + 0.5, 0.000000000000000000000001), 0.9999999999999999)
        return log(y / (1 - y)) * max(
            1,
            abs(x),
        )  # the * min(1,x) is a hack to prioritize game over

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the current principal variation from the best child when needed."""
        best_branch_key = self.best_branch()
        best_child = (
            self.tree_node.branches_children.get(best_branch_key)
            if best_branch_key is not None
            else None
        )
        return self.pv_state.try_update_from_best_child(
            best_branch_key=best_branch_key,
            best_child=best_child,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
            child_pv_version_getter=self._child_pv_version,
            child_best_branch_sequence_getter=self._child_best_branch_sequence,
        )

    def set_best_branch_sequence(self, new_seq: list[BranchKey]) -> bool:
        """Replace PV content and update child-version tracking when it changes."""
        return self.pv_state.set_sequence(
            new_seq,
            current_best_child_version=self._pv_child_version_for_sequence(new_seq),
        )

    def clear_best_branch_sequence(self) -> bool:
        """Clear the stored principal variation content."""
        return self.pv_state.clear()

    def assert_pv_invariants(self) -> None:
        """Assert family-neutral principal-variation invariants."""
        best_branch_key = self.best_branch()

        if best_branch_key is None:
            assert not self.best_branch_sequence, (
                "PV must be empty when no best branch exists."
            )
            return

        if self.best_branch_sequence:
            assert self.best_branch_sequence[0] == best_branch_key, (
                "PV head must match best_branch()."
            )
            child_tree_evaluation = self.child_tree_evaluation(best_branch_key)
            assert child_tree_evaluation is not None, (
                "PV is non-empty but best child is missing from branches_children."
            )

        # NOTE: partial-expansion PV/value policy is owned by backup policies.

    def print_best_line(self) -> None:
        """Log the current best line by following the stored PV through children."""
        info_string = f"Best line from node {self.tree_node.id!s}: "
        node_eval: Any = self
        for branch in self.best_branch_sequence:
            child = node_eval.tree_node.branches_children[branch]
            assert child is not None
            info_string += f"{branch} ({child.tree_node.id!s}) "
            node_eval = child.tree_evaluation
        anemone_logger.info(info_string)


class NodeTreeEvaluation[StateT: State = State](
    NodeValueEvaluation,
    DecisionOrderedEvaluation,
    Protocol,
):
    """Shared tree-search evaluation surface used by generic search orchestration."""

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the current principal-variation branch sequence."""
        ...

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the principal variation from changed child branches."""
        ...

    def best_equivalent_branches(
        self,
        mode: BestBranchEquivalenceMode = BestBranchEquivalenceMode.EQUAL,
    ) -> list[BranchKey]:
        """Return branches equivalent to the best branch under one mode."""
        ...

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Run family-specific backup after child updates."""
        ...
