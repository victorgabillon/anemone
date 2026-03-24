"""Presentation helpers for logging ``TreeExploration`` progress."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from anemone.node_evaluation.tree import debug_printing as tree_eval_debug_printing
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from random import Random

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration


def maybe_log_iteration_progress[NodeT: AlgorithmNode[Any]](
    exploration: TreeExploration[NodeT],
    random_generator: Random,
) -> None:
    """Log a lightweight snapshot of one iteration when sampling allows it."""
    if random_generator.random() >= 0.11:
        return

    root_node = exploration.tree.root_node
    current_best_branch = (
        str(root_node.tree_evaluation.best_branch_sequence[0])
        if root_node.tree_evaluation.best_branch_sequence
        else "?"
    )

    anemone_logger.info("state: %s", root_node.state)

    str_progress = exploration.stopping_criterion.get_string_of_progress(
        exploration.tree
    )
    anemone_logger.info(
        "%s | current best branch: %s | current white value: %s",
        str_progress,
        current_best_branch,
        root_node.tree_evaluation.get_score(),
    )
    tree_eval_debug_printing.print_branch_ordering(
        cast(
            "tree_eval_debug_printing.BranchOrderingPrintableNodeEval",
            root_node.tree_evaluation,
        ),
        dynamics=exploration.tree_manager.dynamics,
    )
    tree_eval_debug_printing.print_best_line(
        cast(
            "tree_eval_debug_printing.BestLinePrintableNodeEval",
            root_node.tree_evaluation,
        )
    )


def log_final_best_line[NodeT: AlgorithmNode[Any]](
    exploration: TreeExploration[NodeT],
) -> None:
    """Log the final best line after exploration finishes."""
    tree_eval_debug_printing.print_best_line(
        cast(
            "tree_eval_debug_printing.BestLinePrintableNodeEval",
            exploration.tree.root_node.tree_evaluation,
        )
    )
