"""Proof policies for classifying exactness after candidate selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from valanga.evaluations import Certainty

from anemone.backup_policies.common import (
    ProofClassification,
    SelectedValue,
    all_child_values_exact,
)
from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from valanga.evaluations import Value

    from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )
    from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
        NodeMaxEvaluation,
    )


class ProofPolicy[NodeEvalT](Protocol):
    """Protocol for classifying proof/exactness after candidate selection."""

    def classify_selected_value(
        self,
        *,
        node_eval: NodeEvalT,
        selection: SelectedValue,
    ) -> ProofClassification | None:
        """Return certainty/outcome metadata for one selected parent candidate."""
        ...


class MaxProofPolicy:
    """Proof policy for single-agent max backup semantics."""

    def classify_selected_value(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
        selection: SelectedValue,
    ) -> ProofClassification | None:
        """Classify certainty/outcome for one selected max candidate."""
        chosen_value = selection.value
        if chosen_value is None:
            return None

        if not selection.from_child:
            return ProofClassification.from_value(chosen_value)

        exact_from_children = (
            node_eval.tree_node.all_branches_generated
            and all_child_values_exact(node_eval)
        )
        return ProofClassification(
            certainty=Certainty.FORCED if exact_from_children else Certainty.ESTIMATE,
            over_event=chosen_value.over_event if exact_from_children else None,
        )


class MinimaxProofPolicy:
    """Proof policy for adversarial minimax backup semantics."""

    def classify_selected_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selection: SelectedValue,
    ) -> ProofClassification | None:
        """Classify certainty/outcome for one selected minimax candidate."""
        chosen_value = selection.value
        if chosen_value is None:
            return None

        if not selection.from_child:
            return ProofClassification.from_value(chosen_value)

        exact = self._selected_child_proves_exact_value(
            node_eval=node_eval,
            selected_child_value=chosen_value,
        )
        return ProofClassification(
            certainty=Certainty.FORCED if exact else Certainty.ESTIMATE,
            over_event=chosen_value.over_event if exact else None,
        )

    def _selected_child_proves_exact_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selected_child_value: Value,
    ) -> bool:
        if not canonical_value.is_exact_value(selected_child_value):
            return False

        # Important: child certainty must never be copied directly to the parent.
        # A non-terminal parent becomes FORCED only when the parent itself is now exact.
        if self._selected_child_proves_parent_exact_value_by_choice(
            node_eval=node_eval,
            selected_child_value=selected_child_value,
        ):
            return True

        if not node_eval.tree_node.all_branches_generated:
            return False

        return all_child_values_exact(node_eval)

    def _selected_child_proves_parent_exact_value_by_choice(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selected_child_value: Value,
    ) -> bool:
        """Return True when the chosen child alone proves the parent's exact value."""
        over_event = selected_child_value.over_event
        return over_event is not None and over_event.is_win_for(
            node_eval.tree_node.state.turn
        )
