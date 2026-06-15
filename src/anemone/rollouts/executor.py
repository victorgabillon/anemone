"""Opening executor that materializes deterministic rollout continuations."""
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anemone import nodes as node
from anemone import trees
from anemone.nodes.opening_status import (
    legal_branch_keys,
    opened_branch_keys,
    sync_opening_status,
)
from anemone.tree_manager.opening_expansion_budget import reserve_branch_opening
from anemone.tree_manager.tree_expander import TreeExpansion, TreeExpansions

from .action_selector import RolloutDecisionContext
from .report import (
    RolloutExpansionReport,
    RolloutExpansionReportBuilder,
    RolloutStopReason,
)

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.dynamics import SearchDynamics
    from anemone.node_selector.opening_instructions import (
        OpeningInstruction,
        OpeningInstructions,
    )
    from anemone.tree_manager.branch_opening_service import BranchOpeningService
    from anemone.tree_manager.opening_expansion_budget import OpeningExpansionBudget

    from .action_selector import RolloutActionSelector


class InvalidRolloutActionError(ValueError):
    """Raised when a rollout action selector returns an invalid action."""


def _invalid_rollout_action_error(
    *,
    action: BranchKey,
    current_node: node.ITreeNode[Any],
    context: RolloutDecisionContext[Any],
) -> InvalidRolloutActionError:
    return InvalidRolloutActionError(
        f"Invalid rollout action {action!r} from node {current_node.id}: "
        f"legal_actions={context.legal_actions!r}, "
        f"opened_actions={context.opened_actions!r}, "
        f"openable_actions={context.openable_actions!r}"
    )


@dataclass(slots=True)
class RolloutOpeningExpansionExecutor[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Execute opening instructions and deterministic materialized continuations."""

    branch_opening_service: BranchOpeningService[NodeT]
    dynamics: SearchDynamics[Any, Any]
    rollout_action_selector: RolloutActionSelector[NodeT]
    max_extra_steps: int | None
    stop_on_existing_node: bool = True
    last_report: RolloutExpansionReport | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Reject invalid rollout configuration."""
        if self.max_extra_steps is not None and self.max_extra_steps < 0:
            msg = "max_extra_steps must be non-negative"
            raise ValueError(msg)

    def expand(
        self,
        *,
        tree: trees.Tree[NodeT],
        opening_instructions: OpeningInstructions[NodeT],
        budget: OpeningExpansionBudget | None = None,
    ) -> TreeExpansions[NodeT]:
        """Expand instructions, then materialize deterministic rollout edges."""
        tree_expansions: TreeExpansions[NodeT] = TreeExpansions()
        touched_parent_nodes_by_id: dict[int, NodeT] = {}
        report_builder = RolloutExpansionReportBuilder()

        opening_instruction: OpeningInstruction[NodeT]
        for opening_instruction in opening_instructions.values():
            if not reserve_branch_opening(budget):
                report_builder.record_stop(RolloutStopReason.BRANCH_BUDGET_EXHAUSTED)
                break
            parent_node = opening_instruction.node_to_open
            initial_expansion = self.branch_opening_service.open_branch(
                tree=tree,
                parent_node=parent_node,
                branch=opening_instruction.branch,
                tree_expansions=tree_expansions,
            )
            touched_parent_nodes_by_id[parent_node.id] = parent_node
            report_builder.record_initial_edge(
                created_node=initial_expansion.creation_child_node
            )

            stop_reason = self._rollout_from_initial_child(
                tree=tree,
                initial_expansion=initial_expansion,
                tree_expansions=tree_expansions,
                touched_parent_nodes_by_id=touched_parent_nodes_by_id,
                report_builder=report_builder,
                budget=budget,
            )
            report_builder.record_stop(stop_reason)

        for parent_node in touched_parent_nodes_by_id.values():
            sync_opening_status(node=parent_node, dynamics=self.dynamics)

        self.last_report = report_builder.build()
        return tree_expansions

    def _rollout_from_initial_child(
        self,
        *,
        tree: trees.Tree[NodeT],
        initial_expansion: TreeExpansion[NodeT],
        tree_expansions: TreeExpansions[NodeT],
        touched_parent_nodes_by_id: dict[int, NodeT],
        report_builder: RolloutExpansionReportBuilder,
        budget: OpeningExpansionBudget | None,
    ) -> RolloutStopReason:
        """Materialize rollout continuation edges after one initial expansion."""
        if self.max_extra_steps == 0:
            return RolloutStopReason.MAX_EXTRA_STEPS

        if self.stop_on_existing_node and not initial_expansion.creation_child_node:
            return RolloutStopReason.EXISTING_NODE

        current_node = initial_expansion.child_node
        rollout_step_index = 0

        while self._has_remaining_rollout_decision_budget(rollout_step_index):
            if _is_terminal_node(current_node):
                return RolloutStopReason.TERMINAL

            context = self._decision_context(
                current_node=current_node,
                rollout_step_index=rollout_step_index,
            )
            if not context.legal_actions:
                return RolloutStopReason.NO_LEGAL_ACTIONS

            action = self._choose_rollout_action(context)
            if action is None:
                return RolloutStopReason.ACTION_SELECTOR_STOP

            if action in context.openable_actions:
                if not reserve_branch_opening(budget):
                    return RolloutStopReason.BRANCH_BUDGET_EXHAUSTED

                rollout_expansion = self.branch_opening_service.open_branch(
                    tree=tree,
                    parent_node=current_node,
                    branch=action,
                    tree_expansions=tree_expansions,
                )
                touched_parent_nodes_by_id[current_node.id] = current_node
                report_builder.record_extra_edge(
                    rollout_depth=rollout_step_index + 1,
                    created_node=rollout_expansion.creation_child_node,
                )

                if (
                    self.stop_on_existing_node
                    and not rollout_expansion.creation_child_node
                ):
                    return RolloutStopReason.EXISTING_NODE

                current_node = rollout_expansion.child_node
                rollout_step_index += 1
                continue

            if action in context.opened_actions:
                child_node = current_node.branches_children.get(action)
                if child_node is None:
                    raise _invalid_rollout_action_error(
                        action=action,
                        current_node=current_node,
                        context=context,
                    )
                report_builder.record_traversal()
                current_node = child_node
                rollout_step_index += 1
                continue

            raise _invalid_rollout_action_error(
                action=action,
                current_node=current_node,
                context=context,
            )

        return RolloutStopReason.MAX_EXTRA_STEPS

    def _has_remaining_rollout_decision_budget(
        self,
        rollout_step_index: int,
    ) -> bool:
        """Return whether another rollout continuation decision may run."""
        return self.max_extra_steps is None or rollout_step_index < self.max_extra_steps

    def _decision_context(
        self,
        *,
        current_node: NodeT,
        rollout_step_index: int,
    ) -> RolloutDecisionContext[NodeT]:
        """Build a traversal-aware action-selection context."""
        legal_actions = tuple(
            legal_branch_keys(node=current_node, dynamics=self.dynamics)
        )
        opened_action_set = opened_branch_keys(current_node)
        opened_actions = tuple(
            action for action in legal_actions if action in opened_action_set
        )
        openable_actions = tuple(
            action for action in legal_actions if action not in opened_action_set
        )
        return RolloutDecisionContext(
            current_node=current_node,
            legal_actions=legal_actions,
            opened_actions=opened_actions,
            openable_actions=openable_actions,
            rollout_step_index=rollout_step_index,
        )

    def _choose_rollout_action(
        self,
        context: RolloutDecisionContext[NodeT],
    ) -> BranchKey | None:
        """Delegate action choice to the configured rollout action selector."""
        return self.rollout_action_selector.choose_action(context)


def _is_terminal_node(node_to_check: node.ITreeNode[Any]) -> bool:
    """Return whether a node exposes terminal state through existing APIs."""
    is_over = getattr(node_to_check, "is_over", None)
    if callable(is_over):
        return bool(is_over())

    tree_evaluation = getattr(node_to_check, "tree_evaluation", None)
    is_terminal = getattr(tree_evaluation, "is_terminal", None)
    if callable(is_terminal):
        return bool(is_terminal())

    return False


__all__ = [
    "InvalidRolloutActionError",
    "RolloutOpeningExpansionExecutor",
]
