"""Integration test for Atomheart Nim through Anemone adversarial search."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, cast

import pytest
from valanga import BranchKey, Color, OverEvent, State
from valanga.evaluations import Certainty, Value
from valanga.over_event import HowOver, Winner
from valanga.policy import BranchPolicy

from anemone import (
    TreeAndValuePlayerArgs,
    create_tree_and_value_branch_selector_with_tree_eval_factory,
)
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from anemone.node_evaluation.tree.factory import NodeTreeMinmaxEvaluationFactory
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.objectives import AdversarialZeroSumObjective
from anemone.progress_monitor.progress_monitor import (
    DepthLimitArgs,
    StoppingCriterionTypes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atomheart.games.nim import NimDynamics, NimState
    from valanga.evaluations import EvalItem
    from valanga.policy import Recommendation

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_and_value_branch_selector import TreeAndValueBranchSelector
else:
    _nim_module = pytest.importorskip(
        "atomheart.games.nim",
        reason="requires an Atomheart installation that exposes the Nim game",
    )
    NimDynamics = _nim_module.NimDynamics
    NimState = _nim_module.NimState

pytestmark = pytest.mark.integration

NEUTRAL_NIM_SCORE = 0.0


def legal_nim_actions(stones: int) -> tuple[int, ...]:
    """Return the legal subtraction actions for a single-pile Nim state."""
    if stones <= 0:
        return ()
    return tuple(range(1, min(3, stones) + 1))


def is_winning_nim_position(stones: int) -> bool:
    """Return whether the side to move can force a win from ``stones``."""
    return stones % 4 != 0


def exact_optimal_nim_actions(stones: int) -> set[int]:
    """Return the exact optimal action set for the side to move."""
    actions = legal_nim_actions(stones)
    if not actions:
        return set()
    if not is_winning_nim_position(stones):
        return set(actions)
    return {action for action in actions if (stones - action) % 4 == 0}


def test_nim_oracle_anchor_cases() -> None:
    """Keep a couple of human-readable oracle anchor cases nearby."""
    assert exact_optimal_nim_actions(4) == {1, 2, 3}
    assert exact_optimal_nim_actions(5) == {1}


class _NimOverDetector(OverEventDetector):
    """Recover Nim terminal metadata from the side to move at ``stones == 0``."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        if not state.is_game_over():
            return None, None

        turn = getattr(state, "turn", None)
        assert isinstance(turn, Color)
        winner = Winner.BLACK if turn is Color.WHITE else Winner.WHITE
        return (
            OverEvent(
                how_over=HowOver.WIN,
                who_is_winner=winner,
                termination="last_stone",
            ),
            None,
        )


class _NeutralNimEvaluator(MasterStateValueEvaluator):
    """Evaluate non-terminal Nim states as neutral and defer terminal handling."""

    over: OverEventDetector

    def __init__(self) -> None:
        self.over = _NimOverDetector()

    def evaluate(self, state: State) -> Value:
        del state
        return canonical_value.make_estimate_value(score=NEUTRAL_NIM_SCORE)

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Value]:
        return [self.evaluate(getattr(item, "state", item)) for item in items]


@dataclass(frozen=True, slots=True)
class _GreedyBestBranchRule:
    """Deterministically choose the currently best root branch."""

    type: str = "greedy_best_branch"

    def policy[StateT: State](self, root_node: AlgorithmNode[StateT]) -> BranchPolicy:
        """Return a one-hot policy on the best branch when available."""
        best_branch = root_node.tree_evaluation.best_branch()
        if best_branch is None:
            return BranchPolicy(probs={})
        return BranchPolicy(probs={best_branch: 1.0})

    def sample(self, policy: BranchPolicy, rng: Random) -> BranchKey:
        """Select the only policy branch without probabilistic sampling."""
        del rng
        assert policy.probs
        return next(iter(policy.probs))


def _build_search_args(*, search_depth: int) -> TreeAndValuePlayerArgs:
    """Return a minimal deterministic uniform-search configuration."""
    return TreeAndValuePlayerArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=DepthLimitArgs(
            type=StoppingCriterionTypes.DEPTH_LIMIT,
            depth_limit=search_depth,
        ),
        recommender_rule=cast("Any", _GreedyBestBranchRule()),
        index_computation=None,
    )


def make_nim_selector(
    *,
    stones: int,
    turn: Color = Color.WHITE,
    search_depth: int,
) -> tuple[TreeAndValueBranchSelector[NimState], NimState, NimDynamics]:
    """Build the real search stack for a tiny adversarial Nim position."""
    state = NimState(stones=stones, turn=turn)
    dynamics = NimDynamics()
    selector = create_tree_and_value_branch_selector_with_tree_eval_factory(
        state_type=NimState,
        dynamics=dynamics,
        args=_build_search_args(search_depth=search_depth),
        random_generator=Random(0),
        master_state_evaluator=_NeutralNimEvaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeTreeMinmaxEvaluationFactory(
            objective=AdversarialZeroSumObjective()
        ),
        hooks=None,
    )
    return selector, state, dynamics


@pytest.mark.parametrize("stones", [1, 2, 3, 4, 5, 6, 7, 8])
def test_uniform_search_finds_an_optimal_root_action_in_nim(stones: int) -> None:
    """Uniform adversarial search should recover an oracle-optimal first move."""
    expected_actions = exact_optimal_nim_actions(stones)

    # DepthLimit(k) expands depths 0..k-1, and Nim always terminates within
    # ``stones`` ply because each move removes at least one stone.
    selector, state, dynamics = make_nim_selector(
        stones=stones,
        search_depth=stones,
    )

    recommendation: Recommendation = selector.recommend(state=state, seed=0)
    chosen_action_name = recommendation.recommended_name
    chosen_action = dynamics.action_from_name(state, chosen_action_name)

    assert recommendation.evaluation is not None
    assert recommendation.evaluation.certainty in {
        Certainty.FORCED,
        Certainty.TERMINAL,
    }
    assert recommendation.branch_evals is not None
    assert chosen_action_name in recommendation.branch_evals
    assert chosen_action in expected_actions
    assert chosen_action_name == dynamics.action_name(state, chosen_action)


@pytest.mark.parametrize("starting_stones", [5, 6, 7])
def test_repeated_replanning_follows_an_optimal_nim_course(
    starting_stones: int,
) -> None:
    """Repeated search should keep choosing oracle-optimal moves for both sides."""
    dynamics = NimDynamics()
    state = NimState(stones=starting_stones, turn=Color.WHITE)
    final_over_event: OverEvent | None = None

    while not state.is_game_over():
        selector, _, _ = make_nim_selector(
            stones=state.stones,
            turn=state.turn,
            search_depth=state.stones,
        )
        recommendation: Recommendation = selector.recommend(state=state, seed=0)
        assert recommendation.evaluation is not None
        chosen_action = dynamics.action_from_name(
            state, recommendation.recommended_name
        )

        assert chosen_action in exact_optimal_nim_actions(state.stones)

        transition = dynamics.step(state, chosen_action)
        final_over_event = transition.over_event
        state = transition.next_state

    assert final_over_event is not None
    assert final_over_event.who_is_winner == Winner.WHITE
