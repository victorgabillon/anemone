"""Single-player depth-aware selector based on direct node values."""

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from valanga.evaluations import Value

from anemone import trees
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


@dataclass
class LinooArgs:
    """Arguments for the Linoo node selector."""

    type: Literal[NodeSelectorType.LINOO]


@dataclass(frozen=True, slots=True)
class LinooDepthSelectionRow:
    """Selector-observable summary for one depth in the live tree."""

    depth: int
    total_nodes: int
    opened_count: int
    frontier_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int
    selection_index: int | None
    active: bool
    selected: bool

    def __post_init__(self) -> None:
        """Keep the diagnostic accounting honest in debug runs."""
        assert self.total_nodes == (
            self.opened_count
            + self.frontier_count
            + self.terminal_count
            + self.exact_count
            + self.uncached_terminal_candidates
            + self.non_openable_count
        )


@dataclass(frozen=True, slots=True)
class LinooSelectionReport:
    """Structured observability payload for the latest Linoo selection."""

    selected_depth: int
    selected_node_id: int
    selected_node_direct_value: float | None
    selected_depth_selection_index: int
    depth_rows: tuple[LinooDepthSelectionRow, ...]
    collect_frontier_state_s: float | None = None
    choose_depth_s: float | None = None
    heap_update_s: float | None = None
    choose_node_s: float | None = None
    make_report_s: float | None = None
    total_s: float | None = None
    depth_row_count: int | None = None
    total_nodes_scanned: int | None = None
    frontier_nodes_scanned: int | None = None
    uncached_terminal_candidates: int | None = None
    selected_depth_frontier_count: int | None = None
    stale_candidates_skipped: int | None = None
    heap_candidates_registered: int | None = None

    def format_depth_table(self) -> str:
        """Return an aligned text table for Linoo depth diagnostics."""
        headers = (
            "depth",
            "total",
            "opened",
            "frontier",
            "terminal",
            "exact",
            "uncached_terminal",
            "non_openable",
            "index",
            "selected",
        )
        rows = [
            (
                str(row.depth),
                str(row.total_nodes),
                str(row.opened_count),
                str(row.frontier_count),
                str(row.terminal_count),
                str(row.exact_count),
                str(row.uncached_terminal_candidates),
                str(row.non_openable_count),
                _format_optional_int(row.selection_index),
                "yes" if row.selected else "no",
            )
            for row in self.depth_rows
        ]
        return _format_table(headers, rows)


@dataclass(init=False, slots=True)
class _LinooDepthAggregation[NodeT]:
    """Mutable accounting bucket for one depth during report construction."""

    total_nodes: int
    opened_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int
    frontier_nodes: list[NodeT]

    def __init__(self) -> None:
        """Initialize empty accounting for one depth."""
        self.total_nodes = 0
        self.opened_count = 0
        self.terminal_count = 0
        self.exact_count = 0
        self.uncached_terminal_candidates = 0
        self.non_openable_count = 0
        self.frontier_nodes = []

    @property
    def frontier_count(self) -> int:
        """Return how many nodes at this depth can currently be opened."""
        return len(self.frontier_nodes)


type _LinooHeapEntry[NodeT] = tuple[float, int, int, NodeT]


class LinooSelectionError(ValueError):
    """Base error for invalid Linoo selection contexts."""


class LinooDirectValueUnavailableError(LinooSelectionError):
    """Raised when a frontier node does not expose a direct single-player value."""


class LinooIncompatibleObjectiveError(LinooSelectionError):
    """Raised when Linoo is used outside the single-player max setting."""


def _no_frontier_nodes_error() -> LinooSelectionError:
    return LinooSelectionError(
        "Linoo could not find any unopened unresolved nodes in the current tree."
    )


def _missing_root_objective_error() -> LinooIncompatibleObjectiveError:
    return LinooIncompatibleObjectiveError(
        "Linoo requires a configured SingleAgentMaxObjective on the tree root."
    )


def _incompatible_objective_error(objective: object) -> LinooIncompatibleObjectiveError:
    return LinooIncompatibleObjectiveError(
        f"Linoo requires a SingleAgentMaxObjective; got {type(objective).__name__}."
    )


def _missing_direct_value_attribute_error(
    node_id: int,
) -> LinooDirectValueUnavailableError:
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} does not expose tree_evaluation.direct_value."
    )


def _missing_direct_value_error(node_id: int) -> LinooDirectValueUnavailableError:
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} has no direct_value."
    )


def _format_optional_int(value: int | None) -> str:
    """Format an optional integer for compact table display."""
    return "-" if value is None else str(value)


def _format_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    """Format string rows as a simple aligned whitespace table."""
    widths = [
        max([len(headers[column]), *(len(row[column]) for row in rows)])
        for column in range(len(headers))
    ]
    formatted_rows = [
        " ".join(
            value.rjust(widths[column])
            for column, value in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join(
        [
            " ".join(
                header.rjust(widths[column])
                for column, header in enumerate(headers)
            ),
            *formatted_rows,
        ]
    )


class Linoo[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Depth-aware single-player selector using frontier direct values.

    Linoo first selects the frontier depth minimizing
    ``opened_count(depth) * (depth + 1)``, then expands the unopened node at
    that depth with the best direct value.
    """

    opening_instructor: OpeningInstructor
    latest_selection_report: LinooSelectionReport | None
    _frontier_heaps_by_depth: dict[int, list[_LinooHeapEntry[NodeT]]]
    _candidate_versions_by_node_id: dict[int, int]
    _candidate_signature_by_node_id: dict[int, object]
    _candidate_heap_present_by_node_id: dict[int, bool]
    _cache_tree_identity: int | None
    _cache_objective_identity: int | None

    def __init__(self, opening_instructor: OpeningInstructor) -> None:
        """Store the opening instructor used to materialize branch openings."""
        self.opening_instructor = opening_instructor
        self.latest_selection_report = None
        self._frontier_heaps_by_depth = {}
        self._candidate_versions_by_node_id = {}
        self._candidate_signature_by_node_id = {}
        self._candidate_heap_present_by_node_id = {}
        self._cache_tree_identity = None
        self._cache_objective_identity = None

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Choose one unopened node according to the Linoo policy."""
        _ = latest_tree_expansions  # Linoo recomputes its view from the current tree.
        self.latest_selection_report = None
        total_started_at = perf_counter()

        objective = self._require_single_agent_objective(tree)
        self._reset_cache_if_needed(tree=tree, objective=objective)
        collect_started_at = perf_counter()
        depth_state_by_depth = self._collect_frontier_state(tree=tree)
        collect_frontier_state_s = perf_counter() - collect_started_at
        frontier_by_depth = {
            depth: depth_state.frontier_nodes
            for depth, depth_state in depth_state_by_depth.items()
            if depth_state.frontier_nodes
        }
        total_nodes_scanned = sum(
            depth_state.total_nodes for depth_state in depth_state_by_depth.values()
        )
        frontier_nodes_scanned = sum(
            depth_state.frontier_count for depth_state in depth_state_by_depth.values()
        )
        uncached_terminal_candidates = sum(
            depth_state.uncached_terminal_candidates
            for depth_state in depth_state_by_depth.values()
        )

        if not frontier_by_depth:
            raise _no_frontier_nodes_error()

        choose_depth_started_at = perf_counter()
        selected_depth = min(
            frontier_by_depth,
            key=lambda depth: (
                depth_state_by_depth[depth].opened_count * (depth + 1),
                depth,
            ),
        )
        choose_depth_s = perf_counter() - choose_depth_started_at
        heap_update_started_at = perf_counter()
        heap_candidates_registered = self._register_frontier_candidates(
            depth=selected_depth,
            nodes=frontier_by_depth[selected_depth],
            objective=objective,
        )
        heap_update_s = perf_counter() - heap_update_started_at
        choose_node_started_at = perf_counter()
        selected_node, stale_candidates_skipped = self._choose_best_node_at_depth(
            tree=tree,
            depth=selected_depth,
        )
        choose_node_s = perf_counter() - choose_node_started_at
        selected_depth_frontier_count = depth_state_by_depth[
            selected_depth
        ].frontier_count
        make_report_started_at = perf_counter()
        self.latest_selection_report = self._make_selection_report(
            depth_state_by_depth=depth_state_by_depth,
            selected_depth=selected_depth,
            selected_node=selected_node,
            collect_frontier_state_s=collect_frontier_state_s,
            choose_depth_s=choose_depth_s,
            heap_update_s=heap_update_s,
            choose_node_s=choose_node_s,
            total_s=perf_counter() - total_started_at,
            total_nodes_scanned=total_nodes_scanned,
            frontier_nodes_scanned=frontier_nodes_scanned,
            uncached_terminal_candidates=uncached_terminal_candidates,
            selected_depth_frontier_count=selected_depth_frontier_count,
            stale_candidates_skipped=stale_candidates_skipped,
            heap_candidates_registered=heap_candidates_registered,
        )
        if self.latest_selection_report is not None:
            self.latest_selection_report = replace(
                self.latest_selection_report,
                make_report_s=perf_counter() - make_report_started_at,
                total_s=perf_counter() - total_started_at,
            )

        all_branches_to_open = self.opening_instructor.all_branches_to_open(
            node_to_open=selected_node
        )
        return create_instructions_to_open_all_branches(
            branches_to_play=all_branches_to_open,
            node_to_open=selected_node,
        )

    def _require_single_agent_objective(
        self,
        tree: trees.Tree[NodeT],
    ) -> SingleAgentMaxObjective[Any]:
        """Return the configured single-agent objective or raise a clear error."""
        try:
            objective = tree.root_node.tree_evaluation.required_objective
        except Exception as exc:
            raise _missing_root_objective_error() from exc

        if not isinstance(objective, SingleAgentMaxObjective):
            raise _incompatible_objective_error(objective)
        return objective

    def _collect_frontier_state(
        self,
        *,
        tree: trees.Tree[NodeT],
    ) -> dict[int, _LinooDepthAggregation[NodeT]]:
        """Group node-state diagnostics by relative depth."""
        depth_state_by_depth: dict[int, _LinooDepthAggregation[NodeT]] = {}

        absolute_tree_depth: int
        for absolute_tree_depth in tree.descendants:
            node: NodeT
            for node in tree.descendants[absolute_tree_depth].values():
                depth = tree.node_depth(node)
                depth_state = depth_state_by_depth.setdefault(
                    depth,
                    _LinooDepthAggregation[NodeT](),
                )
                depth_state.total_nodes += 1

                if self._is_terminal_node(node):
                    depth_state.terminal_count += 1
                    continue

                if node.all_branches_generated:
                    depth_state.opened_count += 1
                    continue

                if node.tree_evaluation.has_exact_value():
                    depth_state.exact_count += 1
                    continue

                if self._direct_value_or_none(node) is None:
                    depth_state.uncached_terminal_candidates += 1
                    continue

                depth_state.frontier_nodes.append(node)

        for depth_state in depth_state_by_depth.values():
            depth_state.non_openable_count = depth_state.total_nodes - (
                depth_state.opened_count
                + depth_state.frontier_count
                + depth_state.terminal_count
                + depth_state.exact_count
                + depth_state.uncached_terminal_candidates
            )

        return depth_state_by_depth

    def _reset_cache_if_needed(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
    ) -> None:
        """Clear heap state when the selector is reused for a new runtime context."""
        tree_identity = id(tree.root_node)
        objective_identity = id(objective)
        if (
            self._cache_tree_identity == tree_identity
            and self._cache_objective_identity == objective_identity
        ):
            return

        self._frontier_heaps_by_depth.clear()
        self._candidate_versions_by_node_id.clear()
        self._candidate_signature_by_node_id.clear()
        self._candidate_heap_present_by_node_id.clear()
        self._cache_tree_identity = tree_identity
        self._cache_objective_identity = objective_identity

    def _register_frontier_candidates(
        self,
        *,
        depth: int,
        nodes: list[NodeT],
        objective: SingleAgentMaxObjective[Any],
    ) -> int:
        """Update cached heap entries for the selected depth only."""
        registered_count = 0
        for node in nodes:
            if self._register_frontier_candidate(
                depth=depth,
                node=node,
                objective=objective,
            ):
                registered_count += 1
        return registered_count

    def _register_frontier_candidate(
        self,
        *,
        depth: int,
        node: NodeT,
        objective: SingleAgentMaxObjective[Any],
    ) -> bool:
        """Push a heap entry when one node is new, changed, or reactivated."""
        direct_value = self._require_direct_value(node)
        node_id = node.id
        signature = self._candidate_signature(node=node, direct_value=direct_value)
        previous_signature = self._candidate_signature_by_node_id.get(node_id)
        heap_entry_present = self._candidate_heap_present_by_node_id.get(
            node_id,
            False,
        )
        if previous_signature == signature and heap_entry_present:
            return False

        version = self._candidate_versions_by_node_id.get(node_id, 0) + 1
        self._candidate_versions_by_node_id[node_id] = version
        self._candidate_signature_by_node_id[node_id] = signature
        self._candidate_heap_present_by_node_id[node_id] = True
        priority = objective.evaluate_value(direct_value, node.state)
        heappush(
            self._frontier_heaps_by_depth.setdefault(depth, []),
            (-priority, node_id, version, node),
        )
        return True

    def _choose_best_node_at_depth(
        self,
        *,
        tree: trees.Tree[NodeT],
        depth: int,
    ) -> tuple[NodeT, int]:
        """Return the best cached frontier node, lazily discarding stale entries."""
        heap = self._frontier_heaps_by_depth.setdefault(depth, [])
        stale_candidates_skipped = 0
        while heap:
            _priority, node_id, version, node = heap[0]
            current_version = self._candidate_versions_by_node_id.get(node_id)
            if version != current_version:
                heappop(heap)
                stale_candidates_skipped += 1
                continue

            if not self._is_current_frontier_candidate(
                tree=tree,
                node=node,
                expected_depth=depth,
            ):
                heappop(heap)
                self._candidate_heap_present_by_node_id[node_id] = False
                stale_candidates_skipped += 1
                continue

            return node, stale_candidates_skipped

        raise _no_frontier_nodes_error()

    def _is_current_frontier_candidate(
        self,
        *,
        tree: trees.Tree[NodeT],
        node: NodeT,
        expected_depth: int,
    ) -> bool:
        """Return whether one cached node is still selectable frontier."""
        return (
            tree.node_depth(node) == expected_depth
            and not self._is_terminal_node(node)
            and not node.all_branches_generated
            and not node.tree_evaluation.has_exact_value()
            and self._direct_value_or_none(node) is not None
        )

    def _make_selection_report(
        self,
        *,
        depth_state_by_depth: dict[int, _LinooDepthAggregation[NodeT]],
        selected_depth: int,
        selected_node: NodeT,
        collect_frontier_state_s: float,
        choose_depth_s: float,
        heap_update_s: float,
        choose_node_s: float,
        total_s: float,
        total_nodes_scanned: int,
        frontier_nodes_scanned: int,
        uncached_terminal_candidates: int,
        selected_depth_frontier_count: int,
        stale_candidates_skipped: int,
        heap_candidates_registered: int,
    ) -> LinooSelectionReport:
        """Build the structured latest-selection table without affecting policy."""
        rows: list[LinooDepthSelectionRow] = []
        selected_depth_selection_index: int | None = None

        for depth, depth_state in depth_state_by_depth.items():
            frontier_nodes = depth_state.frontier_nodes
            opened_count = depth_state.opened_count
            active = bool(frontier_nodes)
            selected = depth == selected_depth
            selection_index = opened_count * (depth + 1) if active else None
            row = LinooDepthSelectionRow(
                depth=depth,
                total_nodes=depth_state.total_nodes,
                opened_count=opened_count,
                frontier_count=depth_state.frontier_count,
                terminal_count=depth_state.terminal_count,
                exact_count=depth_state.exact_count,
                uncached_terminal_candidates=(
                    depth_state.uncached_terminal_candidates
                ),
                non_openable_count=depth_state.non_openable_count,
                selection_index=selection_index,
                active=active,
                selected=selected,
            )
            rows.append(row)
            if selected:
                selected_depth_selection_index = selection_index

        sorted_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.selection_index is None,
                    row.selection_index if row.selection_index is not None else 0,
                    row.depth,
                ),
            )
        )
        if selected_depth_selection_index is None:
            selected_depth_selection_index = (
                depth_state_by_depth[selected_depth].opened_count
                * (selected_depth + 1)
            )

        return LinooSelectionReport(
            selected_depth=selected_depth,
            selected_node_id=selected_node.id,
            selected_node_direct_value=self._require_direct_value(selected_node).score,
            selected_depth_selection_index=selected_depth_selection_index,
            depth_rows=sorted_rows,
            collect_frontier_state_s=collect_frontier_state_s,
            choose_depth_s=choose_depth_s,
            heap_update_s=heap_update_s,
            choose_node_s=choose_node_s,
            make_report_s=None,
            total_s=total_s,
            depth_row_count=len(sorted_rows),
            total_nodes_scanned=total_nodes_scanned,
            frontier_nodes_scanned=frontier_nodes_scanned,
            uncached_terminal_candidates=uncached_terminal_candidates,
            selected_depth_frontier_count=selected_depth_frontier_count,
            stale_candidates_skipped=stale_candidates_skipped,
            heap_candidates_registered=heap_candidates_registered,
        )

    def _is_terminal_node(self, node: NodeT) -> bool:
        """Return cached terminality only; Linoo must not recompute game-over."""
        is_terminal = getattr(node.tree_evaluation, "is_terminal", None)
        return bool(is_terminal()) if callable(is_terminal) else False

    def _direct_value_or_none(self, node: NodeT) -> Value | None:
        """Return direct value access for openable-candidate classification."""
        try:
            return node.tree_evaluation.direct_value
        except AttributeError as exc:
            raise _missing_direct_value_attribute_error(node.id) from exc

    def _require_direct_value(self, node: NodeT) -> Value:
        """Return one node's direct value or raise a clear Linoo error."""
        direct_value = self._direct_value_or_none(node)

        if direct_value is None:
            raise _missing_direct_value_error(node.id)
        return direct_value

    def _candidate_signature(self, *, node: NodeT, direct_value: Value) -> object:
        """Return the cheap invalidation signature for one heap candidate."""
        return (
            id(node),
            direct_value,
            node.all_branches_generated,
            node.tree_evaluation.has_exact_value(),
            self._is_terminal_node(node),
        )
