"""Small shared helpers for explicit backup policy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value


@dataclass(frozen=True, slots=True)
class SelectedValue:
    """One value candidate plus whether it was chosen from a child branch."""

    value: Value | None
    from_child: bool


class ChildValueReader(Protocol):
    """Protocol for node-evaluation objects that can expose child values."""

    tree_node: Any

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the candidate value for one child branch."""


def all_child_values_exact(node_eval: ChildValueReader) -> bool:
    """Return True when every existing child branch has an exact Value."""
    if not node_eval.tree_node.branches_children:
        return False
    for branch_key in node_eval.tree_node.branches_children:
        child_value = node_eval.child_value_candidate(branch_key)
        if not canonical_value.is_exact_value(child_value):
            return False
    return True
