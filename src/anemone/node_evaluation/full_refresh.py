"""Canonical full-refresh phases shared by live and detached evaluation.

No Growth executor, runtime, publication or user-supplied finalizer belongs here.
Callers preflight certificates before replacing the evaluator/version.
"""

# Runtime annotation introspection is supported, like native checkpoint helpers.
# ruff: noqa: TC001

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import batched
from typing import Any, cast

from anemone import node_selector as node_sel
from anemone.indices.index_manager import (
    NodeExplorationIndexManager,
    update_all_indices,
)
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.direct import EvaluationQueries, NodeDirectEvaluator
from anemone.node_evaluation.evaluator_refresh import validate_refresh_node
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective
from anemone.tree_manager.tree_expander import TreeExpansions
from anemone.trees.tree import Tree


@dataclass(slots=True)
class ReevaluationReport:
    """Summary of one explicit node reevaluation request."""

    evaluator_version: int
    requested_count: int
    reevaluated_count: int
    changed_count: int
    skipped_terminal_count: int = 0
    skipped_certified_count: int = 0


def iter_nodes[NodeT: AlgorithmNode[Any]](
    tree: Tree[NodeT], *, reverse: bool = False
) -> Iterator[NodeT]:
    """Walk native depth/insertion order without a second registry."""
    depths = tree.descendants.range()
    for depth in reversed(depths) if reverse else depths:
        yield from tree.descendants[depth].values()


def invalidate_selector(selector: node_sel.NodeSelector[Any]) -> bool:
    """Preserve the optional native selector invalidation contract."""
    if not isinstance(selector, node_sel.InvalidatableNodeSelector):
        return False
    invalidate = cast(
        "Callable[[], None] | None", getattr(selector, "invalidate", None)
    )
    if invalidate is None:
        return False
    invalidate()
    return True


def _changed_tree_certificate_error(node_id: int) -> ValueError:
    return ValueError(f"Tree certificate changed at node {node_id}")


def refresh_bound_context[NodeT: AlgorithmNode[Any]](
    *,
    tree: Tree[NodeT],
    evaluator: NodeDirectEvaluator[Any],
    evaluation_queries: EvaluationQueries,
    evaluator_version: int,
    index_manager: NodeExplorationIndexManager,
    node_selector: node_sel.NodeSelector[NodeT],
    latest_tree_expansions: TreeExpansions[NodeT],
    batch_size: int,
    validate_direct_certificate: Callable[[NodeT], bool] | None,
) -> ReevaluationReport:
    """Evaluate, back up, index and refresh selection in canonical order."""
    report = ReevaluationReport(evaluator_version, 0, 0, 0)
    for batch in batched(iter_nodes(tree), batch_size, strict=False):
        outcome = evaluator.evaluate_nodes(
            batch,
            evaluation_queries=evaluation_queries,
            clear_existing_direct_values=True,
            skip_terminal_nodes=True,
            preserve_certified_values=True,
        )
        report.requested_count += len(batch)
        report.reevaluated_count += outcome.reevaluated_count
        report.changed_count += outcome.changed_count
        report.skipped_terminal_count += outcome.skipped_terminal_count
        report.skipped_certified_count += outcome.skipped_certified_count

    for node in iter_nodes(tree, reverse=True):
        evaluation = node.tree_evaluation
        branches = {b for b, child in node.iter_child_links() if child is not None}
        if branches:
            certificate = evaluation.tree_value
            evaluation.backup_from_children(branches, branches)
            if canonical_value.is_exact_value(certificate):
                if evaluation.tree_value != certificate:
                    evaluation.tree_value = certificate
                    raise _changed_tree_certificate_error(node.id)
                evaluation.tree_value = certificate
        validate_refresh_node(node, validate_direct_certificate)

    update_all_indices(tree=tree, index_manager=index_manager)
    invalidate_selector(node_selector)
    objective = tree.root_node.tree_evaluation.required_objective
    if isinstance(node_selector, node_sel.StatefulNodeSelector) and isinstance(
        objective, SingleAgentMaxObjective
    ):
        stateful_selector: node_sel.StatefulNodeSelector[NodeT] = cast(
            "node_sel.StatefulNodeSelector[NodeT]", node_selector
        )
        stateful_selector.refresh_state_for_checkpoint(  # pylint: disable=no-member
            tree=tree,
            objective=objective,
            latest_tree_expansions=latest_tree_expansions,
        )
    return report
