"""Runnable tree-search runtime centered on ``TreeExploration``."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any, cast

from valanga import BranchKey, State
from valanga.evaluations import Certainty, Value
from valanga.game import BranchName
from valanga.policy import NotifyProgressCallable, Recommendation

from anemone._valanga_types import AnyTurnState
from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.common.branch_frontier import (
    require_branch_frontier_aware,
)
from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.nodes.utils import best_node_sequence_from_node
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
    ProgressMonitor,
    create_stopping_criterion,
)
from anemone.search_factory import NodeSelectorFactory
from anemone.value_updates import NodeValueUpdate, NodeValueUpdateResult

from . import node_selector as node_sel
from . import recommender_rule, trees
from . import tree_manager as tree_man
from .trees.factory import ValueTreeFactory

if TYPE_CHECKING:
    from valanga.policy import BranchPolicy


type IterationProgressReporter = Callable[[Any, Random], None]
type SearchResultReporter = Callable[[Any], None]


_SUPPORTED_REFRESH_SCOPES = (
    "root_children",
    "pv",
    "frontier",
    "leaves",
    "all",
    "pv_and_root_children",
)


def _invalid_frontier_limit_error() -> ValueError:
    """Return the canonical public error for invalid frontier limits."""
    return ValueError("k must be non-negative or None")


def _unsupported_refresh_scope_error(scope: str) -> ValueError:
    """Return the canonical public error for invalid refresh scopes."""
    supported_scopes = ", ".join(_SUPPORTED_REFRESH_SCOPES)
    return ValueError(
        f"Unsupported refresh scope {scope!r}. Supported scopes are: "
        f"{supported_scopes}."
    )


def _missing_node_value_update_error(node_ids: tuple[str, ...]) -> ValueError:
    """Return the public error for disallowed missing node ids."""
    formatted_node_ids = ", ".join(repr(node_id) for node_id in node_ids)
    return ValueError(
        f"Node value updates reference missing node ids: {formatted_node_ids}."
    )


def _duplicate_public_node_id_error(node_id: str) -> RuntimeError:
    """Return the internal error for impossible duplicate public ids."""
    return RuntimeError(f"Live tree contains duplicate public node id {node_id!r}.")


@dataclass
class TreeExplorationResult[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Result of exploring one search tree."""

    branch_recommendation: Recommendation
    tree: trees.Tree[NodeT]


@dataclass(slots=True)
class ReevaluationReport:
    """Summary of one explicit node reevaluation request."""

    evaluator_version: int
    requested_count: int
    reevaluated_count: int
    changed_count: int
    skipped_terminal_count: int = 0


def compute_child_evals[StateT: State](
    root: AlgorithmNode[StateT],
    dynamics: SearchDynamics[StateT, Any],
) -> dict[BranchName, Value]:
    """Compute evaluations for each existing child branch."""
    evals: dict[BranchName, Value] = {}
    for bk, child in root.branches_children.items():
        if child is None:
            continue

        # Use whatever your canonical per-node evaluation is:
        bk_name = dynamics.action_name(root.state, bk)
        evals[bk_name] = child.tree_evaluation.get_value()
    return evals


@dataclass
class TreeExploration[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Runnable runtime object for one tree search.

    ``TreeExploration`` is the single explicit sequencing owner for one search
    iteration. It owns the search state for one run and exposes the two main
    runtime operations:

    * ``step()`` for one iteration
    * ``explore(...)`` for a full run to recommendation

    ``SearchRuntime`` is the preferred public alias for callers who want a
    shorter top-level name for this runtime concept.
    """

    tree: trees.Tree[NodeT]
    tree_manager: tree_man.AlgorithmNodeTreeManager[NodeT]
    node_selector: node_sel.NodeSelector[NodeT]
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    stopping_criterion: ProgressMonitor[NodeT]
    notify_percent_function: NotifyProgressCallable
    iteration_progress_reporter: IterationProgressReporter | None = None
    search_result_reporter: SearchResultReporter | None = None
    evaluator_version: int = 0
    _latest_tree_expansions: tree_man.TreeExpansions[NodeT] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Seed selector-visible iteration state for the first step."""
        self._latest_tree_expansions = self._make_initial_tree_expansions()
        node_evaluator = getattr(self.tree_manager, "node_evaluator", None)
        if node_evaluator is not None:
            node_evaluator.current_evaluator_version = self.evaluator_version

    def _make_initial_tree_expansions(self) -> tree_man.TreeExpansions[NodeT]:
        """Return the synthetic root creation log used before the first step."""
        tree_expansions: tree_man.TreeExpansions[NodeT] = tree_man.TreeExpansions()
        tree_expansions.record_creation(
            tree_expansion=tree_man.TreeExpansion(
                child_node=self.tree.root_node,
                parent_node=None,
                state_modifications=None,
                creation_child_node=True,
                branch_key=None,
            )
        )
        return tree_expansions

    @property
    def latest_tree_expansions(self) -> tree_man.TreeExpansions[NodeT]:
        """Return selector-visible expansion records from the latest iteration."""
        return self._latest_tree_expansions

    @latest_tree_expansions.setter
    def latest_tree_expansions(
        self,
        value: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Restore selector-visible expansion records from a checkpoint payload."""
        self._latest_tree_expansions = value

    def _report_iteration_progress(self, random_generator: Random) -> None:
        """Invoke the optional iteration-progress reporter for this runtime."""
        if self.iteration_progress_reporter is None:
            return
        self.iteration_progress_reporter(self, random_generator)

    def print_info_during_branch_computation(self, random_generator: Random) -> None:
        """Backward-compatible alias for the iteration-progress reporter."""
        self._report_iteration_progress(random_generator)

    def _report_search_result(self) -> None:
        """Invoke the optional search-result reporter for this runtime."""
        if self.search_result_reporter is None:
            return
        self.search_result_reporter(self)

    def set_evaluator(self, new_evaluator: MasterStateValueEvaluator) -> None:
        """Replace the active evaluator for future direct evaluations only.

        Existing node values, backups, and tree structure are left untouched.
        Only direct evaluations that happen after this call use the new
        evaluator and the incremented evaluator version.
        """
        node_evaluator = self.tree_manager.node_evaluator
        assert node_evaluator is not None
        node_evaluator.master_state_value_evaluator = new_evaluator
        self.evaluator_version += 1
        node_evaluator.current_evaluator_version = self.evaluator_version

    def reevaluate_nodes(self, nodes: Sequence[NodeT]) -> ReevaluationReport:
        """Reevaluate existing nodes without changing the current tree structure."""
        outcome = self.tree_manager.reevaluate_nodes(tree=self.tree, nodes=nodes)
        return ReevaluationReport(
            evaluator_version=self.evaluator_version,
            requested_count=len(nodes),
            reevaluated_count=outcome.reevaluated_count,
            changed_count=outcome.changed_count,
            skipped_terminal_count=outcome.skipped_terminal_count,
        )

    def apply_node_value_updates(
        self,
        updates: Iterable[NodeValueUpdate],
        *,
        recompute_backups: bool = True,
        allow_missing: bool = True,
    ) -> NodeValueUpdateResult:
        """Apply direct-value updates to live tree nodes addressed by public node id.

        Missing node ids are reported by default so external callers can apply
        bounded patch artifacts against a tree that may have moved on. When
        requested, existing value-propagation logic recomputes affected
        ancestors and refreshes derived exploration indices.
        """
        materialized_updates = list(updates)
        nodes_by_id = self._nodes_by_public_id()
        missing_node_ids = tuple(
            update.node_id
            for update in materialized_updates
            if update.node_id not in nodes_by_id
        )
        if missing_node_ids and not allow_missing:
            raise _missing_node_value_update_error(missing_node_ids)

        applied_nodes: list[NodeT] = []
        changed_nodes: list[NodeT] = []
        for update in materialized_updates:
            node = nodes_by_id.get(update.node_id)
            if node is None:
                continue

            changed = self._apply_node_value_update(node=node, update=update)
            applied_nodes.append(node)
            if changed:
                changed_nodes.append(node)

        recomputed_count: int | None = None
        if recompute_backups:
            if changed_nodes:
                recomputed_nodes = self.tree_manager.value_propagator.propagate_after_local_value_changes(
                    changed_nodes
                )
                recomputed_count = len(recomputed_nodes)
                self.tree_manager.refresh_exploration_indices(tree=self.tree)
            else:
                recomputed_count = 0

        return NodeValueUpdateResult(
            requested_count=len(materialized_updates),
            applied_count=len(applied_nodes),
            missing_node_ids=missing_node_ids,
            recomputed_count=recomputed_count,
        )

    def _nodes_by_public_id(self) -> dict[str, NodeT]:
        """Return live nodes keyed by Anemone's public string node id."""
        nodes_by_id: dict[str, NodeT] = {}
        for node in self._all_nodes_in_tree_order():
            public_node_id = str(node.id)
            if public_node_id in nodes_by_id:
                raise _duplicate_public_node_id_error(public_node_id)
            nodes_by_id[public_node_id] = node
        return nodes_by_id

    def _apply_node_value_update(
        self,
        *,
        node: NodeT,
        update: NodeValueUpdate,
    ) -> bool:
        """Apply one already-validated update and report whether value state changed."""
        node_eval = node.tree_evaluation
        direct_value_before = node_eval.direct_value
        backed_up_value_before = node_eval.backed_up_value

        node_eval.direct_value = self._value_from_update_score(
            score=update.direct_value,
            update=update,
            existing_value=direct_value_before,
        )
        if update.backed_up_value is not None:
            node_eval.backed_up_value = self._value_from_update_score(
                score=update.backed_up_value,
                update=update,
                existing_value=backed_up_value_before,
            )

        return (
            direct_value_before != node_eval.direct_value
            or backed_up_value_before != node_eval.backed_up_value
        )

    def _value_from_update_score(
        self,
        *,
        score: float,
        update: NodeValueUpdate,
        existing_value: Value | None,
    ) -> Value:
        """Create a safe ``Value`` for a scalar update and optional exactness hints."""
        existing_over_event = (
            existing_value.over_event if existing_value is not None else None
        )
        if update.is_terminal is True and existing_over_event is not None:
            return canonical_value.make_terminal_value(
                score=score,
                over_event=existing_over_event,
            )
        if update.is_exact is True:
            return canonical_value.make_forced_value(
                score=score,
                over_event=existing_over_event,
            )
        if update.is_exact is False:
            return canonical_value.make_estimate_value(score=score)
        if existing_value is not None and existing_value.certainty == Certainty.FORCED:
            return canonical_value.make_forced_value(
                score=score,
                over_event=existing_over_event,
            )
        if (
            update.is_terminal is not False
            and existing_value is not None
            and existing_value.certainty == Certainty.TERMINAL
            and existing_over_event is not None
        ):
            return canonical_value.make_terminal_value(
                score=score,
                over_event=existing_over_event,
            )
        return canonical_value.make_estimate_value(score=score)

    def _deduplicate_nodes_in_order(self, nodes: Sequence[NodeT]) -> list[NodeT]:
        """Return one stable deduplicated node list keyed by object identity."""
        deduplicated_nodes: list[NodeT] = []
        seen_node_ids: set[int] = set()
        for node in nodes:
            node_id = id(node)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            deduplicated_nodes.append(node)
        return deduplicated_nodes

    def _all_nodes_in_tree_order(self) -> list[NodeT]:
        """Return all currently known nodes in stable depth/insertion order."""
        descendants = self.tree.descendants
        return [
            node
            for tree_depth in descendants.range()
            for node in descendants[tree_depth].values()
        ]

    def _linked_child_nodes_in_tree_order(self, node: NodeT) -> list[NodeT]:
        """Return linked children in stable structural child-link order.

        The shared runtime does not currently expose one stronger generic child
        ordering accessor here, so this helper intentionally follows the
        insertion-ordered structural child mapping.
        """
        return [child for child in node.branches_children.values() if child is not None]

    def _collect_root_children(self) -> list[NodeT]:
        """Collect the current root's linked children."""
        return self._linked_child_nodes_in_tree_order(self.tree.root_node)

    def _collect_pv_nodes(self) -> list[NodeT]:
        """Collect the current principal variation below the root."""
        pv_nodes = cast(
            "list[NodeT]", best_node_sequence_from_node(self.tree.root_node)
        )
        if len(pv_nodes) <= 1:
            return []
        return pv_nodes[1:]

    def _collect_leaf_nodes(self) -> list[NodeT]:
        """Collect structural leaves in stable tree order."""
        return [
            node
            for node in self._all_nodes_in_tree_order()
            if not self._linked_child_nodes_in_tree_order(node)
        ]

    def _collect_frontier_nodes(self, k: int | None = None) -> list[NodeT]:
        """Collect frontier nodes via branch-frontier semantics, not all leaves.

        This follows the currently stored branch-frontier ordering to find the
        unresolved nodes that sit at the tips of search-relevant frontier
        branches.
        """
        if k is not None and k < 0:
            raise _invalid_frontier_limit_error()
        if k == 0:
            return []

        frontier_nodes: list[NodeT] = []
        visited_nodes: set[int] = set()

        def visit(node: NodeT) -> None:
            if k is not None and len(frontier_nodes) >= k:
                return
            node_identity = id(node)
            if node_identity in visited_nodes:
                return
            visited_nodes.add(node_identity)

            if node.tree_evaluation.has_exact_value():
                return

            frontier_aware = require_branch_frontier_aware(node.tree_evaluation)
            if not frontier_aware.has_frontier_branches():
                frontier_nodes.append(node)
                return

            for branch in frontier_aware.frontier_branches_in_order():
                child = node.branches_children.get(branch)
                if child is None:
                    continue
                visit(child)
                if k is not None and len(frontier_nodes) >= k:
                    return

        visit(self.tree.root_node)
        return frontier_nodes

    def _collect_all_nodes(self) -> list[NodeT]:
        """Collect all nodes currently present in the tree."""
        return self._all_nodes_in_tree_order()

    def _collect_pv_and_root_children(self) -> list[NodeT]:
        """Collect the stable union of PV nodes and root children."""
        return self._deduplicate_nodes_in_order(
            [*self._collect_root_children(), *self._collect_pv_nodes()]
        )

    def _validate_refresh_scope_inputs(
        self,
        *,
        scope: str,
        k: int | None = None,
    ) -> None:
        """Validate public refresh-scope inputs before mutating runtime state."""
        if scope not in _SUPPORTED_REFRESH_SCOPES:
            raise _unsupported_refresh_scope_error(scope)
        if scope == "frontier" and k is not None and k < 0:
            raise _invalid_frontier_limit_error()

    def _collect_nodes_for_refresh_scope(
        self,
        *,
        scope: str,
        k: int | None = None,
    ) -> list[NodeT]:
        """Collect nodes for one supported refresh scope."""
        if scope == "root_children":
            return self._collect_root_children()
        if scope == "pv":
            return self._collect_pv_nodes()
        if scope == "frontier":
            return self._collect_frontier_nodes(k=k)
        if scope == "leaves":
            return self._collect_leaf_nodes()
        if scope == "all":
            return self._collect_all_nodes()
        if scope == "pv_and_root_children":
            return self._collect_pv_and_root_children()
        raise _unsupported_refresh_scope_error(scope)

    def reevaluate_root_children(self) -> ReevaluationReport:
        """Reevaluate the current root's immediate children."""
        return self.reevaluate_nodes(self._collect_root_children())

    def reevaluate_pv(self) -> ReevaluationReport:
        """Reevaluate the current principal variation below the root."""
        return self.reevaluate_nodes(self._collect_pv_nodes())

    def reevaluate_frontier(self, k: int | None = None) -> ReevaluationReport:
        """Reevaluate current branch-frontier nodes, optionally limited to ``k``."""
        return self.reevaluate_nodes(self._collect_frontier_nodes(k=k))

    def reevaluate_leaves(self) -> ReevaluationReport:
        """Reevaluate the tree's current structural leaves."""
        return self.reevaluate_nodes(self._collect_leaf_nodes())

    def refresh_with_evaluator(
        self,
        new_evaluator: MasterStateValueEvaluator,
        *,
        scope: str = "frontier",
        k: int | None = None,
    ) -> ReevaluationReport:
        """Swap evaluator, then reevaluate one selected tree region."""
        self._validate_refresh_scope_inputs(scope=scope, k=k)
        self.set_evaluator(new_evaluator)
        nodes = self._collect_nodes_for_refresh_scope(scope=scope, k=k)
        return self.reevaluate_nodes(nodes)

    def _select_node_for_expansion(self) -> node_sel.OpeningInstructions[NodeT]:
        """Ask the selector for the next branches to open."""
        return self.node_selector.choose_node_and_branch_to_open(
            tree=self.tree,
            latest_tree_expansions=self._latest_tree_expansions,
        )

    def _limit_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
    ) -> node_sel.OpeningInstructions[NodeT]:
        """Apply the stopping criterion's opening-budget filter."""
        return self.stopping_criterion.respectful_opening_instructions(
            opening_instructions=opening_instructions,
            tree=self.tree,
        )

    def _expand_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
    ) -> tree_man.TreeExpansions[NodeT]:
        """Run the structural expansion phase for one iteration."""
        return self.tree_manager.expand_instructions(
            tree=self.tree,
            opening_instructions=opening_instructions,
        )

    def _evaluate_expansions(
        self,
        tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Run the direct-evaluation phase for newly created nodes."""
        self.tree_manager.evaluate_expansions(tree_expansions=tree_expansions)

    def _propagate_iteration_updates(
        self,
        tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Run the post-evaluation propagation and refresh phases."""
        self.tree_manager.update_backward(tree_expansions=tree_expansions)
        self.tree_manager.propagate_depth_index(tree_expansions=tree_expansions)
        self.tree_manager.refresh_exploration_indices(tree=self.tree)

    def step(self) -> None:
        """Run one search iteration in the canonical runtime order."""
        # Search stops once the root value is exact, even if the root state is
        # still non-terminal and some siblings remain unopened.
        assert not self.tree.root_node.tree_evaluation.has_exact_value()

        opening_instructions = self._select_node_for_expansion()
        opening_instructions_subset = self._limit_opening_instructions(
            opening_instructions
        )
        tree_expansions = self._expand_opening_instructions(opening_instructions_subset)
        self._evaluate_expansions(tree_expansions)
        self._propagate_iteration_updates(tree_expansions)
        self._latest_tree_expansions = tree_expansions

    def explore(self, random_generator: Random) -> TreeExplorationResult[NodeT]:
        """Explore the tree to find the best branch.

        Args:
            random_generator (Random): The random number generator.

        Returns:
            TreeExplorationResult[NodeT]: The recommended branch and its evaluation.

        """
        self._latest_tree_expansions = self._make_initial_tree_expansions()

        loop: int = 0
        while self.stopping_criterion.should_we_continue(tree=self.tree):
            loop = loop + 1
            self._report_iteration_progress(random_generator=random_generator)
            self.step()

            if loop % 10 == 0:
                self.stopping_criterion.notify_percent_progress(
                    tree=self.tree, notify_percent_function=self.notify_percent_function
                )

        policy: BranchPolicy = self.recommend_branch_after_exploration.policy(
            self.tree.root_node
        )

        best_branch: BranchKey = self.recommend_branch_after_exploration.sample(
            policy, random_generator
        )

        best_branch_name = self.tree_manager.dynamics.action_name(
            self.tree.root_node.state, best_branch
        )
        self._report_search_result()

        branch_recommendation = Recommendation(
            recommended_name=best_branch_name,
            evaluation=self.tree.root_node.tree_evaluation.get_value(),
            policy=policy,
            branch_evals=compute_child_evals(
                self.tree.root_node,
                dynamics=self.tree_manager.dynamics,
            ),
        )

        tree_exploration_result: TreeExplorationResult[NodeT] = TreeExplorationResult(
            branch_recommendation=branch_recommendation, tree=self.tree
        )

        return tree_exploration_result


SearchRuntime = TreeExploration


def create_tree_exploration[StateT: AnyTurnState](
    node_selector_create: NodeSelectorFactory,
    starting_state: StateT,
    tree_manager: tree_man.AlgorithmNodeTreeManager[AlgorithmNode[StateT]],
    tree_factory: ValueTreeFactory[StateT],
    stopping_criterion_args: AllStoppingCriterionArgs,
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs,
    notify_percent_function: NotifyProgressCallable | None = None,
    iteration_progress_reporter: IterationProgressReporter | None = None,
    search_result_reporter: SearchResultReporter | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Assemble ``TreeExploration`` from already-created collaborators.

    Top-level callers usually prefer the higher-level
    ``anemone.create_tree_and_value_exploration(...)`` helpers, which build
    these collaborators and return this same runtime object directly.
    """
    # creates the tree
    tree: trees.Tree[AlgorithmNode[StateT]] = tree_factory.create(
        starting_state=starting_state
    )
    # creates the node selector
    node_selector: node_sel.NodeSelector[AlgorithmNode[StateT]] = node_selector_create()
    stopping_criterion: ProgressMonitor[AlgorithmNode[StateT]] = (
        create_stopping_criterion(
            args=stopping_criterion_args, node_selector=node_selector
        )
    )

    tree_exploration: TreeExploration[AlgorithmNode[StateT]] = TreeExploration(
        tree=tree,
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
        node_selector=node_selector,
        recommend_branch_after_exploration=recommend_branch_after_exploration,
        notify_percent_function=notify_percent_function,
        iteration_progress_reporter=iteration_progress_reporter,
        search_result_reporter=search_result_reporter,
    )

    return tree_exploration
