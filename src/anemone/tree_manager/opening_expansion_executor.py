"""Executors for materializing opening instructions into tree expansions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone import nodes as node
from anemone import trees
from anemone.nodes.opening_status import sync_opening_status

from .opening_expansion_budget import reserve_branch_opening
from .tree_expander import TreeExpansions

if TYPE_CHECKING:
    from anemone.dynamics import SearchDynamics
    from anemone.node_selector.opening_instructions import (
        OpeningInstruction,
        OpeningInstructions,
    )

    from .branch_opening_service import BranchOpeningService
    from .opening_expansion_budget import OpeningExpansionBudget


class OpeningExpansionExecutor[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
](Protocol):
    """Batch executor for opening instructions."""

    def expand(
        self,
        *,
        tree: trees.Tree[FamilyT],
        opening_instructions: OpeningInstructions[FamilyT],
        budget: OpeningExpansionBudget | None = None,
    ) -> TreeExpansions[FamilyT]:
        """Expand a batch of opening instructions."""
        ...


@dataclass(slots=True)
class OnePlyOpeningExpansionExecutor[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Execute each opening instruction as one materialized tree edge."""

    branch_opening_service: BranchOpeningService[FamilyT]
    dynamics: SearchDynamics[Any, Any]

    def expand(
        self,
        *,
        tree: trees.Tree[FamilyT],
        opening_instructions: OpeningInstructions[FamilyT],
        budget: OpeningExpansionBudget | None = None,
    ) -> TreeExpansions[FamilyT]:
        """Expand instructions and sync each touched parent exactly once."""
        tree_expansions: TreeExpansions[FamilyT] = TreeExpansions()
        touched_parent_nodes_by_id: dict[int, FamilyT] = {}

        opening_instruction: OpeningInstruction[FamilyT]
        for opening_instruction in opening_instructions.values():
            if not reserve_branch_opening(budget):
                break
            parent_node = opening_instruction.node_to_open
            self.branch_opening_service.open_branch(
                tree=tree,
                parent_node=parent_node,
                branch=opening_instruction.branch,
                tree_expansions=tree_expansions,
            )
            touched_parent_nodes_by_id[parent_node.id] = parent_node

        for parent_node in touched_parent_nodes_by_id.values():
            sync_opening_status(node=parent_node, dynamics=self.dynamics)

        return tree_expansions


__all__ = [
    "OnePlyOpeningExpansionExecutor",
    "OpeningExpansionExecutor",
]
