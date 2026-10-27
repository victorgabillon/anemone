"""Forward real Morpion checkpoint operations through the test solo-state adapter."""

from typing import Any

from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from anemone.checkpoints import load_search_from_sharded_checkpoint
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests.morpion_refresh_helpers import Dynamics, Evaluator, State
from tests.test_checkpoint_load import _build_linoo_args


class Codec:
    """Delegate decoding to Atomheart; count operations without caching states."""

    def __init__(self) -> None:
        """Keep one domain codec and aggregate decode counters."""
        self.inner = MorpionStateCheckpointCodec()
        self.dynamics = Dynamics()
        self.anchor_decodes = 0
        self.delta_decodes = 0

    def wrap(self, board: Any) -> State:
        """Compute structural terminality with the actual game rules."""
        return State(board, self.dynamics.inner.is_terminal_state(board))

    def dump_anchor_ref(self, state: State) -> object:
        """Use the existing replayable domain anchor."""
        return self.inner.dump_anchor_ref(state.board)

    def dump_delta_from_parent(
        self, *, parent_state: State, child_state: State, branch_from_parent: Any = None
    ) -> object:
        """Use the existing checked parent-relative domain delta."""
        return self.inner.dump_delta_from_parent(
            parent_state=parent_state.board,
            child_state=child_state.board,
            branch_from_parent=branch_from_parent,
        )

    def load_anchor_ref(self, anchor_ref: object) -> State:
        """Count and delegate anchor reconstruction."""
        self.anchor_decodes += 1
        return self.wrap(self.inner.load_anchor_ref(anchor_ref))

    def load_child_from_delta(
        self, *, parent_state: State, delta_ref: object, branch_from_parent: Any = None
    ) -> State:
        """Count and delegate delta reconstruction while retaining the parent."""
        self.delta_decodes += 1
        return self.wrap(
            self.inner.load_child_from_delta(
                parent_state=parent_state.board,
                delta_ref=delta_ref,
                branch_from_parent=branch_from_parent,
            )
        )

    def dump_state_summary(self, state: State) -> object:
        """Keep the domain's original canonical identity and terminality summary."""
        return self.inner.dump_state_summary(state.board)

    def dump_state_parent_branch_for_checkpoint(
        self, branch_from_parent: Any
    ) -> object:
        """Use the domain codec's existing compact parent-branch policy."""
        return self.inner.dump_state_parent_branch_for_checkpoint(branch_from_parent)


def load(checkpoint: Any, codec: Codec, capacity: int | None) -> Any:
    """Use the existing streaming loader; None selects the baseline's old API."""
    args = _build_linoo_args()
    args.stopping_criterion.tree_branch_limit = 100000
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    return load_search_from_sharded_checkpoint(
        checkpoint,
        state_codec=codec,
        dynamics=Dynamics(),
        args=args,
        state_type=State,
        master_state_value_evaluator=Evaluator(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        **({} if capacity is None else {"state_cache_capacity": capacity}),
    )
