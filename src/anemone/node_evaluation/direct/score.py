"""Exact shared numeric transformation for nonterminal direct predictions."""

DISCOUNT = 0.99999999


def scale_nonterminal_score(raw_score: float, tree_depth: int) -> float:
    """Apply Growth's existing arithmetic at checkpoint-relative tree depth.

    Depth counts search edges from the checkpoint root, not absolute played
    moves. Callers retain the Value's certainty and must leave certified terminal
    scores outside this transformation. No rounding or validity policy is added.
    """
    return (1 / DISCOUNT) ** tree_depth * raw_score
