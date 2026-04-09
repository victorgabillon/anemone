"""Single-player depth-aware selector based on direct node values."""

from dataclasses import dataclass
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


class Linoo[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Depth-aware single-player selector using frontier direct values.

    Linoo first selects the frontier depth minimizing
    ``opened_count(depth) * (depth + 1)``, then expands the unopened node at
    that depth with the best direct value.
    """

    opening_instructor: OpeningInstructor

    def __init__(self, opening_instructor: OpeningInstructor) -> None:
        """Store the opening instructor used to materialize branch openings."""
        self.opening_instructor = opening_instructor

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Choose one unopened node according to the Linoo policy."""
        _ = latest_tree_expansions  # Linoo recomputes its view from the current tree.

        objective = self._require_single_agent_objective(tree)
        frontier_by_depth, opened_count_by_depth = self._collect_frontier_state(tree)

        if not frontier_by_depth:
            raise _no_frontier_nodes_error()

        selected_depth = min(
            frontier_by_depth,
            key=lambda depth: (
                opened_count_by_depth.get(depth, 0) * (depth + 1),
                depth,
            ),
        )
        selected_node = max(
            frontier_by_depth[selected_depth],
            key=lambda node: (
                objective.evaluate_value(
                    self._require_direct_value(node),
                    node.state,
                ),
                -node.id,
            ),
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
        tree: trees.Tree[NodeT],
    ) -> tuple[dict[int, list[NodeT]], dict[int, int]]:
        """Group unopened frontier nodes and opened counts by relative depth."""
        frontier_by_depth: dict[int, list[NodeT]] = {}
        opened_count_by_depth: dict[int, int] = {}

        absolute_tree_depth: int
        for absolute_tree_depth in tree.descendants:
            node: NodeT
            for node in tree.descendants[absolute_tree_depth].values():
                depth = tree.node_depth(node)
                if node.all_branches_generated:
                    opened_count_by_depth[depth] = (
                        opened_count_by_depth.get(depth, 0) + 1
                    )
                    continue

                if node.tree_evaluation.has_exact_value():
                    continue

                self._require_direct_value(node)
                frontier_by_depth.setdefault(depth, []).append(node)

        return frontier_by_depth, opened_count_by_depth

    def _require_direct_value(self, node: NodeT) -> Value:
        """Return one node's direct value or raise a clear Linoo error."""
        try:
            direct_value = node.tree_evaluation.direct_value
        except AttributeError as exc:
            raise _missing_direct_value_attribute_error(node.id) from exc

        if direct_value is None:
            raise _missing_direct_value_error(node.id)
        return direct_value
