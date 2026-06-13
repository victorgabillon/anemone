"""Reusable primitive for opening and recording one materialized branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone import nodes as node
from anemone import trees

from .tree_expander import TreeExpansion, TreeExpansions, record_tree_expansion

if TYPE_CHECKING:
    from collections.abc import Callable

    from valanga import BranchKey


class BranchOpeningTreeManager[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
](Protocol):
    """Structural tree-manager surface needed to open one branch."""

    def assert_branch_not_opened(
        self,
        *,
        parent_node: FamilyT,
        branch: BranchKey,
    ) -> None:
        """Raise if ``branch`` already has a concrete child from ``parent_node``."""
        ...

    def open_tree_expansion_from_branch(
        self,
        tree: trees.Tree[FamilyT],
        parent_node: FamilyT,
        branch: BranchKey,
    ) -> TreeExpansion[FamilyT]:
        """Open one structural branch and return its expansion record."""
        ...


@dataclass(slots=True)
class BranchOpeningService[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Open one branch through ``TreeManager`` and record the expansion."""

    tree_manager: BranchOpeningTreeManager[FamilyT]
    on_branch_opened: Callable[[FamilyT, BranchKey], None] | None = None

    def open_branch(
        self,
        *,
        tree: trees.Tree[FamilyT],
        parent_node: FamilyT,
        branch: BranchKey,
        tree_expansions: TreeExpansions[FamilyT],
    ) -> TreeExpansion[FamilyT]:
        """Open one branch, record its expansion, and return that expansion."""
        self.tree_manager.assert_branch_not_opened(
            parent_node=parent_node,
            branch=branch,
        )
        if self.on_branch_opened is not None:
            self.on_branch_opened(parent_node, branch)

        tree_expansion = self.tree_manager.open_tree_expansion_from_branch(  # pylint: disable=assignment-from-no-return
            tree=tree,
            parent_node=parent_node,
            branch=branch,
        )
        record_tree_expansion(
            tree=tree,
            tree_expansions=tree_expansions,
            tree_expansion=tree_expansion,
        )
        return tree_expansion


__all__ = [
    "BranchOpeningService",
    "BranchOpeningTreeManager",
]
