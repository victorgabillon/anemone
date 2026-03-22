"""Integration test for Atomheart integer reduction through Anemone search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import TYPE_CHECKING, Any, cast

import pytest
import valanga
from valanga import BranchKey, Color, Outcome, OverEvent, State, Transition, TurnState
from valanga.evaluations import Certainty, Value
from valanga.policy import BranchPolicy

from anemone import (
    TreeAndValuePlayerArgs,
    create_tree_and_value_branch_selector_with_tree_eval_factory,
)
from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.direct.node_direct_evaluator import NodeDirectEvaluator
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
from anemone.objectives import SingleAgentMaxObjective
from anemone.progress_monitor.progress_monitor import (
    DepthLimitArgs,
    StoppingCriterionTypes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atomheart.games.integer_reduction import (
        IntegerReductionDynamics,
        IntegerReductionState,
    )
    from valanga.evaluations import EvalItem
    from valanga.policy import Recommendation

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_and_value_branch_selector import TreeAndValueBranchSelector
else:
    _integer_reduction_module = pytest.importorskip(
        "atomheart.games.integer_reduction",
        reason=(
            "requires an Atomheart installation that exposes the integer_reduction game"
        ),
    )
    IntegerReductionDynamics = _integer_reduction_module.IntegerReductionDynamics
    IntegerReductionState = _integer_reduction_module.IntegerReductionState

pytestmark = pytest.mark.integration

NONTERMINAL_FALLBACK_SCORE = -10_000.0


@lru_cache
def exact_steps_to_one(n: int) -> int:
    """Return the exact minimum number of actions needed to reach ``1``."""
    if n == 1:
        return 0

    child_costs = [exact_steps_to_one(n - 1)]
    if n % 2 == 0:
        child_costs.append(exact_steps_to_one(n // 2))
    return 1 + min(child_costs)


def exact_optimal_actions(n: int) -> set[str]:
    """Return the set of optimal first action names for ``n``."""
    child_costs = {"dec1": exact_steps_to_one(n - 1)}
    if n % 2 == 0:
        child_costs["half"] = exact_steps_to_one(n // 2)

    best_cost = min(child_costs.values())
    return {
        action_name
        for action_name, child_cost in child_costs.items()
        if child_cost == best_cost
    }


def test_integer_reduction_oracle_anchor_cases() -> None:
    """Keep a couple of human-readable oracle anchor cases near the regression."""
    assert exact_optimal_actions(4) == {"half"}
    assert exact_optimal_actions(5) == {"dec1"}


@dataclass(frozen=True, slots=True)
class _IntegerReductionTurnState(TurnState):
    """Single-agent wrapper around Atomheart's integer-reduction state.

    Anemone's current single-agent path still expects ``TurnState`` and also
    deduplicates nodes by ``state.tag`` at each depth. Integer reduction can
    expose two distinct legal actions from the same parent that land on the
    same integer, for example ``2 -> dec1`` and ``2 -> half``. The
    ``incoming_edge_token`` keeps those action-distinct children separate so the
    test can genuinely assert which first action search recommends.
    """

    base_state: IntegerReductionState
    turn: Color = Color.WHITE
    incoming_edge_token: tuple[valanga.StateTag, str] | None = None

    @property
    def value(self) -> int:
        """Expose the wrapped integer value for test-local helpers."""
        return self.base_state.value

    @property
    def tag(self) -> valanga.StateTag:
        """Return the cache key used by Anemone tree deduplication."""
        if self.incoming_edge_token is None:
            return self.base_state.tag
        return (self.base_state.tag, self.incoming_edge_token)

    def is_game_over(self) -> bool:
        """Return whether the wrapped Atomheart state is terminal."""
        return self.base_state.is_game_over()

    def pprint(self) -> str:
        """Return the wrapped state's human-readable form."""
        return self.base_state.pprint()

    def __str__(self) -> str:
        """Return the wrapped state's compact string form."""
        return str(self.base_state)


class _NeverOverDetector(OverEventDetector):
    """Defer terminal handling to the node-aware evaluator below."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        del state
        return None, None


class _FallbackIntegerReductionEvaluator(MasterStateValueEvaluator):
    """Return a fixed bad estimate for every non-terminal state."""

    over: OverEventDetector

    def __init__(self) -> None:
        self.over = _NeverOverDetector()

    def evaluate(self, state: State) -> Value:
        del state
        return canonical_value.make_estimate_value(score=NONTERMINAL_FALLBACK_SCORE)

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Value]:
        return [self.evaluate(getattr(item, "state", item)) for item in items]


class _DepthAwareIntegerReductionEvaluator(
    NodeDirectEvaluator[_IntegerReductionTurnState]
):
    """Score terminal nodes by actual search depth and others by fallback."""

    def check_obvious_over_events(
        self, node: AlgorithmNode[_IntegerReductionTurnState]
    ) -> None:
        if not node.state.is_game_over():
            return

        node.tree_evaluation.direct_value = canonical_value.make_terminal_value(
            score=-float(node.tree_depth),
            over_event=_reached_one_over_event(),
        )


@dataclass(frozen=True, slots=True)
class _IntegerReductionSearchDynamics(SearchDynamics[_IntegerReductionTurnState, str]):
    """Search-time adapter that delegates rule logic to Atomheart."""

    __anemone_search_dynamics__ = True

    base_dynamics: IntegerReductionDynamics

    def legal_actions(
        self, state: _IntegerReductionTurnState
    ) -> valanga.BranchKeyGeneratorP[str]:
        """Return the real legal actions for the wrapped Atomheart state."""
        return self.base_dynamics.legal_actions(state.base_state)

    def step(
        self,
        state: _IntegerReductionTurnState,
        action: str,
        *,
        depth: int,
    ) -> Transition[_IntegerReductionTurnState]:
        """Step the wrapped Atomheart state while preserving single-agent turn."""
        del depth
        transition = self.base_dynamics.step(state.base_state, action)
        return Transition(
            next_state=_IntegerReductionTurnState(
                base_state=transition.next_state,
                turn=state.turn,
                incoming_edge_token=(state.tag, action),
            ),
            modifications=transition.modifications,
            is_over=transition.is_over,
            over_event=transition.over_event,
            info=transition.info,
        )

    def action_name(self, state: _IntegerReductionTurnState, action: str) -> str:
        """Return the canonical Atomheart action name."""
        return self.base_dynamics.action_name(state.base_state, action)

    def action_from_name(self, state: _IntegerReductionTurnState, name: str) -> str:
        """Resolve an action name via Atomheart's real parser."""
        return self.base_dynamics.action_from_name(state.base_state, name)


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


def _reached_one_over_event() -> OverEvent:
    """Return the canonical exact metadata for reaching the goal state."""
    return OverEvent(
        outcome=Outcome.DRAW,
        termination="reached_one",  # type: ignore[arg-type]
    )


def _build_search_args(*, search_depth: int) -> TreeAndValuePlayerArgs:
    """Return the minimal deterministic uniform-search configuration."""
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


def make_integer_reduction_selector(
    *,
    start_value: int,
    search_depth: int,
) -> tuple[
    TreeAndValueBranchSelector[_IntegerReductionTurnState],
    _IntegerReductionTurnState,
    _IntegerReductionSearchDynamics,
]:
    """Assemble the real branch selector with only a tiny evaluator override.

    We deliberately build through
    ``create_tree_and_value_branch_selector_with_tree_eval_factory(...)`` so the
    full search stack stays on the normal production assembly path. The only
    post-construction tweak is swapping in a node-aware direct evaluator,
    because the current public factory accepts only state-only evaluator hooks.
    """
    state = _IntegerReductionTurnState(base_state=IntegerReductionState(start_value))
    dynamics = _IntegerReductionSearchDynamics(IntegerReductionDynamics())
    master_evaluator = _FallbackIntegerReductionEvaluator()
    selector = create_tree_and_value_branch_selector_with_tree_eval_factory(
        state_type=_IntegerReductionTurnState,
        dynamics=dynamics,
        args=_build_search_args(search_depth=search_depth),
        random_generator=Random(0),
        master_state_evaluator=master_evaluator,
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeTreeMinmaxEvaluationFactory(
            objective=SingleAgentMaxObjective()
        ),
        hooks=None,
    )

    depth_aware_evaluator = _DepthAwareIntegerReductionEvaluator(
        master_state_evaluator=master_evaluator
    )
    # The current public factory only exposes state-level evaluator injection,
    # so replace the direct evaluator after construction to make terminal
    # scoring depend on the actual node depth.
    selector.tree_factory.node_direct_evaluator = depth_aware_evaluator
    selector.tree_manager.node_evaluator = depth_aware_evaluator

    return selector, state, dynamics


@pytest.mark.parametrize("start_value", [4, 5, 6, 8, 10])
def test_uniform_search_finds_an_optimal_first_action_for_integer_reduction(
    start_value: int,
) -> None:
    """Uniform search should recover an oracle-optimal first move."""
    expected_actions = exact_optimal_actions(start_value)
    optimal_depth = exact_steps_to_one(start_value)

    # DepthLimit(k) expands nodes at depths 0..k-1, which is enough to create
    # and evaluate terminal children at exact depth ``k``.
    selector, state, dynamics = make_integer_reduction_selector(
        start_value=start_value,
        search_depth=optimal_depth,
    )

    recommendation: Recommendation = selector.recommend(state=state, seed=0)
    chosen_action_name = recommendation.recommended_name

    assert recommendation.evaluation is not None
    assert recommendation.evaluation.certainty in {
        Certainty.ESTIMATE,
        Certainty.FORCED,
        Certainty.TERMINAL,
    }
    assert recommendation.branch_evals is not None
    assert chosen_action_name in recommendation.branch_evals
    assert chosen_action_name in expected_actions
    assert chosen_action_name == dynamics.action_name(
        state,
        dynamics.action_from_name(state, chosen_action_name),
    )
