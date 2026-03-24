"""Debug/logging helpers for tree-evaluation state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.dynamics import SearchDynamics
    from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey


class BestLinePrintableNodeEval(Protocol):
    """Minimal node-evaluation surface needed for best-line logging."""

    tree_node: Any

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the current principal-variation branch sequence."""
        ...


class BranchOrderingPrintableNodeEval(Protocol):
    """Minimal node-evaluation surface needed for branch-order logging."""

    tree_node: Any

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches in decision order."""
        ...

    def branch_ordering_key(self, branch: BranchKey) -> BranchOrderingKey:
        """Return the cached ordering key for one branch."""
        ...


def _missing_pv_child_error(*, node_id: object, branch: BranchKey) -> RuntimeError:
    return RuntimeError(
        "Cannot print best line for node "
        f"{node_id!s}: PV branch {branch!r} has no linked child."
    )


def print_best_line(node_eval: BestLinePrintableNodeEval) -> None:
    """Log the current best line by following the stored PV through children."""
    info_string = f"Best line from node {node_eval.tree_node.id!s}: "
    running_node_eval: Any = node_eval
    for branch in node_eval.best_branch_sequence:
        child = running_node_eval.tree_node.branches_children[branch]
        if child is None:
            raise _missing_pv_child_error(
                node_id=running_node_eval.tree_node.id,
                branch=branch,
            )
        info_string += f"{branch} ({child.id!s}) "
        running_node_eval = child.tree_evaluation
    anemone_logger.info(info_string)


def print_branch_ordering(
    node_eval: BranchOrderingPrintableNodeEval,
    *,
    dynamics: SearchDynamics[Any, Any],
) -> None:
    """Log child branches in the current decision order."""
    ordered_branches = node_eval.decision_ordered_branches()
    anemone_logger.info(
        "here are the %s branches in decision order:",
        len(ordered_branches),
    )

    string_info = ""
    for branch_key in ordered_branches:
        ordering_key = node_eval.branch_ordering_key(branch_key)
        branch_name = dynamics.action_name(node_eval.tree_node.state, branch_key)
        string_info += f" {branch_name} {ordering_key.primary_score} $$ "
    anemone_logger.info(string_info)
