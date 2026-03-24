"""Provide the generic tree-evaluation engine plus its shared public protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, assert_never, cast

from valanga import BranchKey, State

from anemone._valanga_types import AnyOverEvent, AnyTurnState
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
from anemone.node_evaluation.common.branch_ordering import DecisionOrderedEvaluation
from anemone.node_evaluation.common.node_value_evaluation import NodeValueEvaluation
from anemone.node_evaluation.common.principal_variation import (
    PrincipalVariationState,
)
from anemone.node_evaluation.tree import (
    best_branch_equivalence,
    branch_ordering_runtime,
    debug_printing,
    principal_variation_runtime,
    value_access,
)
from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
    DecisionOrderingState,
)
from anemone.nodes.itree_node import ITreeNode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga.evaluations import Value

    from anemone.backup_policies.types import BackupResult
    from anemone.dynamics import SearchDynamics
    from anemone.nodes.tree_node import TreeNode
    from anemone.objectives.objective import Objective


BackupPolicyFactory = Callable[[], BackupPolicy[Any]]


def _missing_objective_error(node_id: object) -> RuntimeError:
    return RuntimeError(
        f"Cannot use tree evaluation for node {node_id}: no objective is configured."
    )


def _missing_pv_child_for_branch_error(
    *,
    node_id: object,
    branch_key: BranchKey,
) -> RuntimeError:
    return RuntimeError(
        "Cannot build PV line for node "
        f"{node_id}: no child is linked to branch {branch_key!r}."
    )


def _missing_backup_policy_error(node_id: object) -> RuntimeError:
    return RuntimeError(
        f"Cannot back up child values for node {node_id}: no backup_policy is configured."
    )


def _missing_best_branch_for_pv_rebuild_error(node_id: object) -> RuntimeError:
    return RuntimeError(
        "Cannot rebuild the principal variation for node "
        f"{node_id}: no current best branch is available."
    )


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
    """Minimal child-node protocol needed by the shared tree-evaluation engine.

    Family wrappers can extend this protocol when they want stronger recursive
    typing, but the generic engine only depends on this surface.
    """

    @property
    def tree_evaluation(self) -> ChildTreeEvaluation:
        """Return the child node's tree evaluation."""
        ...


class ChildTreeEvaluation(Protocol):
    """Minimal child-evaluation protocol consumed by family-neutral helpers."""

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
    StateT: AnyTurnState = AnyTurnState,
]:
    """Shared concrete state/helper base for tree-evaluation families.

    This is the generic engine underneath the thin named family wrappers such as
    ``NodeMinmaxEvaluation`` and ``NodeMaxEvaluation``. Decision ordering,
    backup delegation, and family-neutral best-branch equivalence live here,
    while family-specific defaults and vocabulary stay in the wrappers.
    """

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = None

    tree_node: TreeNode[NodeT, StateT]
    objective: Objective[StateT] | None = None

    backup_policy: BackupPolicy[Any] | None = None
    direct_value: Value | None = None
    _backed_up_value: Value | None = None
    decision_ordering: DecisionOrderingState = field(
        default_factory=DecisionOrderingState
    )
    pv_state: PrincipalVariationState = field(
        default_factory=make_principal_variation_state_factory
    )
    branch_frontier: BranchFrontierState = field(
        default_factory=make_branch_frontier_factory
    )

    def __post_init__(self) -> None:
        """Eagerly install the family's default backup policy when omitted."""
        if self.backup_policy is not None:
            return

        default_factory = cast(
            "BackupPolicyFactory | None",
            getattr(self.__class__, "_default_backup_policy_factory", None),
        )
        if default_factory is not None:
            self.backup_policy = default_factory()

    @property
    def required_objective(self) -> Objective[StateT]:
        """Return the configured objective or raise a clear configuration error."""
        if self.objective is None:
            raise _missing_objective_error(self.tree_node.id)
        return self.objective

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
        return value_access.get_value_candidate(self)

    def get_value(self) -> Value:
        """Return the canonical Value for this node."""
        return value_access.get_value(self)

    def get_score(self) -> float:
        """Return the canonical scalar score for this node."""
        return value_access.get_score(self)

    def has_exact_value(self) -> bool:
        """Return True when the candidate Value is exact."""
        return value_access.has_exact_value(self)

    def is_terminal(self) -> bool:
        """Return True when the candidate Value says this node's own state is terminal."""
        return value_access.is_terminal(self)

    def has_over_event(self) -> bool:
        """Return True when the candidate Value carries exact outcome metadata."""
        return value_access.has_over_event(self)

    def get_over_event_candidate(self) -> AnyOverEvent | None:
        """Return exact outcome metadata from the candidate Value when present."""
        return value_access.get_over_event_candidate(self)

    @property
    def over_event(self) -> AnyOverEvent | None:
        """Return exact outcome metadata when present on the canonical candidate."""
        return self.get_over_event_candidate()

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the best available Value candidate for a child branch."""
        return value_access.child_value_candidate(self, branch_key)

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
        if child_tree_evaluation is None:
            raise _missing_pv_child_for_branch_error(
                node_id=self.tree_node.id,
                branch_key=branch_key,
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
        return principal_variation_runtime.child_pv_version(child)

    def _child_best_branch_sequence(self, child: Any) -> list[BranchKey]:
        """Return one child's current PV sequence."""
        return principal_variation_runtime.child_best_branch_sequence(child)

    def _pv_child_version_for_sequence(self, sequence: list[BranchKey]) -> int | None:
        """Return the cached best-child PV version for a given PV sequence head."""
        return principal_variation_runtime.pv_child_version_for_sequence(
            sequence=sequence,
            child_tree_evaluation_getter=self,
        )

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

    def _ordered_candidate_branches_with_child_fallback(
        self,
        preferred_ordered_branches: Iterable[BranchKey],
    ) -> tuple[BranchKey, ...]:
        """Return preferred branches first, then any remaining child branches."""
        return branch_ordering_runtime.ordered_candidate_branches_with_child_fallback(
            preferred_ordered_branches=preferred_ordered_branches,
            available_child_branches=self.tree_node.branches_children,
        )

    def _ordered_candidate_branches_for_frontier(self) -> tuple[BranchKey, ...]:
        """Return candidate branches from cached decision ordering plus fallback."""
        return self._ordered_candidate_branches_with_child_fallback(
            self.decision_ordering.decision_ordered_branches(
                child_value_candidate_getter=self.child_value_candidate,
                semantic_compare=self._decision_semantic_compare,
            )
        )

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using the node's current objective semantics."""
        return self.required_objective.semantic_compare(
            left, right, self.tree_node.state
        )

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by node-local semantics plus cached tie-breaks."""
        return self.decision_ordering.decision_ordered_branches(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch."""
        return self.decision_ordering.best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def second_best_branch(self) -> BranchKey:
        """Return the current second-best branch."""
        return self.decision_ordering.second_best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Delegate backup work to the configured tree-evaluation backup policy."""
        if self.backup_policy is None:
            raise _missing_backup_policy_error(self.tree_node.id)
        return self.backup_policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
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
        """Return the cached branch-ordering key for one branch."""
        return self.decision_ordering.branch_ordering_keys[branch]

    def branch_ordering_key(self, branch: BranchKey) -> BranchOrderingKey:
        """Return the cached ordering key for one child branch."""
        return self._branch_ordering_key(branch)

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
            return self._branch_values_are_considered_equal(
                branch=branch,
                best_branch=best_branch,
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
        """Return whether two branches share Value and tactical quality."""
        return best_branch_equivalence.branch_values_are_equal(
            branch=branch,
            best_branch=best_branch,
            child_value_candidate_getter=self.child_value_candidate,
            tactical_quality_key_getter=self._branch_tactical_quality_key,
        )

    def _branch_tactical_quality_key(self, branch: BranchKey) -> int:
        """Return the shared tactical-quality key used for exact branch equality."""
        return self._branch_exact_line_tactical_quality(branch)

    def _branch_exact_line_tactical_quality(self, branch: BranchKey) -> int:
        """Return outcome-aware line quality for one child branch.

        The shared default prefers shorter lines. Exact non-draw outcomes may
        flip the sign when the family says the outcome is unfavorable for this
        node, which makes longer lines compare as tactically better.
        """
        child_value = self.child_value_candidate(branch)
        assert child_value is not None, (
            f"Exact-line tactical quality requires a child Value for branch {branch!r}."
        )

        child_tree_evaluation = self.child_tree_evaluation(branch)
        assert child_tree_evaluation is not None, (
            "Exact-line tactical quality requires an existing child evaluation for "
            f"branch {branch!r}."
        )
        pv_length = len(child_tree_evaluation.best_branch_sequence)

        over_event = child_value.over_event
        if over_event is None or over_event.is_draw():
            return pv_length

        polarity = self._exact_outcome_polarity(
            over_event=over_event,
            child_value=child_value,
        )
        if polarity < 0:
            return -pv_length
        return pv_length

    def _exact_outcome_polarity(
        self,
        *,
        over_event: AnyOverEvent,
        child_value: Value,
    ) -> int:
        """Return whether one exact outcome is favorable, unfavorable, or neutral."""
        del child_value
        role = self.tree_node.state.turn

        if over_event.is_win_for(role):
            return 1
        if over_event.is_loss_for(role):
            return -1
        return 0

    def _branch_equivalence_score(self, branch: BranchKey) -> float:
        """Return the cached primary score used for branch-equivalence heuristics."""
        return self._branch_ordering_key(branch).primary_score

    def _branch_logistic_equivalence_score(self, branch: BranchKey) -> float:
        """Return the score used by logistic-style best-branch equivalence."""
        return best_branch_equivalence.logistic_equivalence_score(
            self._branch_equivalence_score(branch)
        )

    def _are_almost_equal_scores(self, left: float, right: float) -> bool:
        """Return whether two scalar branch-equivalence scores are close enough."""
        return best_branch_equivalence.are_almost_equal_scores(left, right)

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Refresh cached branch-ordering keys for branches that currently have a value."""
        branch_ordering_runtime.update_branches_values(
            decision_ordering=self.decision_ordering,
            branches_to_consider=branches_to_consider,
            child_value_candidate_getter=self.child_value_candidate,
            branch_ordering_key_getter=self._compute_branch_ordering_key,
        )

    def _branch_values_are_considered_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two branches tie under the node's decision semantics."""
        return best_branch_equivalence.branch_values_are_considered_equal(
            branch=branch,
            best_branch=best_branch,
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=lambda left, right: (
                self.required_objective.semantic_compare(
                    left,
                    right,
                    self.tree_node.state,
                )
            ),
        )

    def _compute_branch_ordering_key(self, branch_key: BranchKey) -> BranchOrderingKey:
        """Build the cached branch-ordering key for one child branch."""
        return branch_ordering_runtime.compute_branch_ordering_key(
            branch_key=branch_key,
            child_node_getter=self.tree_node.branches_children.get,
            child_value_candidate_getter=self.child_value_candidate,
            primary_score_getter=lambda key: self.required_objective.evaluate_value(
                cast("Value", self.child_value_candidate(key)),
                self.tree_node.state,
            ),
            tactical_quality_getter=self._branch_exact_line_tactical_quality,
        )

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
        return principal_variation_runtime.update_best_branch_sequence(
            pv_state=self.pv_state,
            best_branch_key=best_branch_key,
            best_child=best_child,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

    def set_best_branch_sequence(self, new_seq: list[BranchKey]) -> bool:
        """Replace PV content and update child-version tracking when it changes."""
        return principal_variation_runtime.set_best_branch_sequence(
            pv_state=self.pv_state,
            new_sequence=new_seq,
            current_best_child_version=self._pv_child_version_for_sequence(new_seq),
        )

    def clear_best_branch_sequence(self) -> bool:
        """Clear the stored principal variation content."""
        return principal_variation_runtime.clear_best_branch_sequence(
            pv_state=self.pv_state
        )

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Rebuild the PV head from the current best child."""
        best_branch_key = self.best_branch()
        if best_branch_key is None:
            raise _missing_best_branch_for_pv_rebuild_error(self.tree_node.id)
        has_best_branch_seq_changed = self.set_best_branch_sequence(
            self.best_branch_line_from_child(best_branch_key)
        )
        assert self.best_branch_sequence, (
            "Rebuilding the principal variation from a best child must produce "
            "a non-empty PV."
        )
        return has_best_branch_seq_changed

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
        """Backward-compatible debug helper for logging the current best line."""
        debug_printing.print_best_line(self)

    def print_branch_ordering(
        self,
        dynamics: SearchDynamics[Any, Any],
    ) -> None:
        """Backward-compatible debug helper for logging branch order."""
        debug_printing.print_branch_ordering(self, dynamics=dynamics)


class NodeTreeEvaluation[StateT: State = State](
    NodeValueEvaluation,
    DecisionOrderedEvaluation,
    Protocol,
):
    """Shared tree-search evaluation surface exposed by all family wrappers."""

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the current principal-variation branch sequence."""
        ...

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the principal variation from changed child branches."""
        ...

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Rebuild the principal variation from the current best child."""
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

    def branch_ordering_key(self, branch: BranchKey) -> BranchOrderingKey:
        """Return the cached ordering key for one child branch."""
        ...
