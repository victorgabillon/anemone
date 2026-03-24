"""Pure helper formulas shared by exploration-index strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode


def node_score(node: AlgorithmNode[Any]) -> float:
    """Return the scalar score used by exploration-index strategies."""
    return node.tree_evaluation.get_score()


def global_min_change_metrics(
    *,
    child_score: float,
    parent_min_path_value: float,
    parent_max_path_value: float,
) -> tuple[float, float, float]:
    """Return child path bounds plus the global-min-change exploration index."""
    child_min_path_value = min(child_score, parent_min_path_value)
    child_max_path_value = max(child_score, parent_max_path_value)
    child_index = abs(child_max_path_value - child_min_path_value) / 2
    return child_min_path_value, child_max_path_value, child_index


def zipf_factored_probability(
    *,
    child_rank: int,
    parent_zipf_factored_proba: float,
) -> float:
    """Return the recursively factored Zipf probability for one child."""
    child_zipf_proba = 1 / (child_rank + 1)
    return child_zipf_proba * parent_zipf_factored_proba


def zipf_exploration_index(
    *,
    zipf_factored_proba_value: float,
    node_depth: int,
) -> float:
    """Return the Zipf-based exploration index for one node depth."""
    inverse_depth = 1 / (node_depth + 1)
    return -(zipf_factored_proba_value * inverse_depth)
