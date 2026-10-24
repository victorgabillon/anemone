"""Legal Atomheart Morpion fixtures for the scientific refresh regression."""

from dataclasses import dataclass, replace
from random import Random
from typing import Any

from atomheart.games.morpion import MorpionDynamics, MorpionState, initial_state
from valanga import SOLO, Outcome, OverEvent, SoloRole, Transition
from valanga.evaluations import Certainty, Value

from anemone import create_search_with_tree_eval_factory
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests.test_checkpoint_load import _build_linoo_args


@dataclass(frozen=True)
class State:
    """Add exact structural terminality and the solo turn to an Atomheart state."""

    board: MorpionState
    is_terminal: bool = False
    turn: SoloRole = SOLO

    @property
    def tag(self) -> int:
        """Return the game's canonical identity."""
        return self.board.tag

    def is_game_over(self) -> bool:
        """Return terminality obtained from the legal game transition."""
        return self.is_terminal

    def pprint(self) -> str:
        """Render the underlying board."""
        return self.board.pprint()


class Dynamics:
    """Preserve the real rules, actions and transitions, adding a solo turn."""

    def __init__(self) -> None:
        """Create the existing Atomheart rules implementation."""
        self.inner = MorpionDynamics()

    def legal_actions(self, state: State) -> Any:
        """Use existing symmetry-reduced legal action generation."""
        return self.inner.legal_actions(state.board)

    def step(self, state: State, action: Any) -> Transition[State]:
        """Wrap the real transition without changing game semantics."""
        result = self.inner.step(state.board, action)
        return replace(result, next_state=State(result.next_state, result.is_over))

    def action_name(self, state: State, action: Any) -> str:
        """Format the game's branch key."""
        return self.inner.action_name(state.board, action)

    def action_from_name(self, state: State, name: str) -> Any:
        """Parse a game's branch key."""
        return self.inner.action_from_name(state.board, name)


class Evaluator:
    """Use the research heuristic, with optional controlled raw predictions."""

    def __init__(self, predictions: dict[int, float] | None = None) -> None:
        """Bind controlled node predictions and record inference window sizes."""
        self.dynamics = Dynamics()
        self.over = self
        self.predictions = predictions or {}
        self.batches: list[tuple[int, ...]] = []

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent[Any] | None, float | None]:
        """Use genuine node-local game completion, scoring played moves."""
        if state.is_terminal:
            return OverEvent(
                outcome=Outcome.WIN, winner=None, termination="no_legal_moves"
            ), float(state.board.moves)
        return None, None

    def evaluate_batch_items(self, items: Any) -> list[Value]:
        """Keep every original prediction except explicitly controlled nodes."""
        self.batches.append(tuple(n.id for n in items))
        return [
            Value(
                score=self.predictions.get(
                    n.id,
                    float(n.state.board.moves)
                    + 0.01 * len(self.dynamics.legal_actions(n.state).get_all()),
                ),
                certainty=Certainty.ESTIMATE,
            )
            for n in items
        ]


def runtime(
    *, branch_limit: int = 6, steps: int = 2, state: State | None = None
) -> Any:
    """Grow the actual seed-4 seven-node witness using the existing factory."""
    args = _build_linoo_args()
    args.stopping_criterion.tree_branch_limit = branch_limit
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    result = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=state or State(initial_state()),
        args=args,
        random_generator=Random(4),
        master_state_evaluator=Evaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(steps):
        result.step()
    return result
