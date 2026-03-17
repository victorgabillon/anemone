"""Provide the adversarial node-evaluation protocol."""

from typing import Protocol

from valanga import (
    BranchKey,
    State,
)
from valanga.evaluations import Value

from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluation


class NodeAdversarialEvaluation[StateT: State = State](
    NodeTreeEvaluation[StateT], Protocol
):
    """Adversarial decision and backup semantics layered on canonical values."""

    @property
    def minmax_value(self) -> Value | None:
        """Return the adversarial backed-up minimax value for this node."""
        ...

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set the adversarial backed-up minimax value for this node."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...
