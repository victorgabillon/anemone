"""Thin toy-domain adapters that feed the real Anemone exploration stack.

This module intentionally does not implement any toy search runtime. The toy
scenario layer is limited to:

* declarative state/domain fixtures
* a tiny search-dynamics adapter
* a tiny direct-evaluation adapter
* one helper that assembles the existing production exploration pipeline via
  ``create_tree_and_value_branch_selector_with_tree_eval_factory(...)``

All tree growth, node selection, backup propagation, and stopping behavior come
from the real Anemone engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any, cast

from valanga import (
    BranchKey,
    Color,
    OverEvent,
    State,
    StateTag,
    Transition,
)

from anemone.dynamics import SearchDynamics
from anemone.factory import (
    TreeAndValuePlayerArgs,
    create_tree_and_value_branch_selector_with_tree_eval_factory,
)
from anemone.node_evaluation.node_direct_evaluation.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from anemone.node_evaluation.node_max_evaluation_factory import (
    NodeMaxEvaluationFactory,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import SoftmaxRule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from valanga.evaluations import EvalItem

    from anemone.tree_exploration import TreeExploration

    from .model import ToyPlayerRole, ToyScenarioSpec


class InvalidToyActionNameError(ValueError):
    """Raised when a branch label does not exist in the current toy state."""

    def __init__(self, *, node_id: str, action_name: str) -> None:
        super().__init__(f"Unknown action {action_name!r} for toy state {node_id!r}.")


class InvalidToyScenarioRolesError(ValueError):
    """Raised when a scenario mixes incompatible non-terminal player roles."""

    def __init__(self) -> None:
        super().__init__(
            "Toy scenarios must use either only 'single' non-terminal nodes or "
            "only 'max'/'min' non-terminal nodes."
        )


class InvalidToyBranchKeyError(TypeError):
    """Raised when a toy scenario receives a non-string branch key."""

    def __init__(self, *, branch_key: object) -> None:
        super().__init__(
            f"Toy scenarios require string branch labels, got {type(branch_key)!r}."
        )


class ToyBranchKeyGenerator:
    """Small deterministic branch-key generator over action labels."""

    sort_branch_keys: bool = False

    def __init__(self, keys: list[str] | tuple[str, ...]) -> None:
        self._keys = list(keys)
        self._index = 0

    @property
    def all_generated_keys(self) -> list[str] | None:
        """Return all pre-generated keys."""
        return self._keys

    def __iter__(self) -> ToyBranchKeyGenerator:
        """Return an iterator over the action labels."""
        self._index = 0
        return self

    def __next__(self) -> str:
        """Return the next action label."""
        if self._index >= len(self._keys):
            raise StopIteration
        key = self._keys[self._index]
        self._index += 1
        return key

    def more_than_one(self) -> bool:
        """Return whether more than one action is available."""
        return len(self._keys) > 1

    def get_all(self) -> list[str]:
        """Return all action labels in deterministic order."""
        return list(self._keys)

    def copy_with_reset(self) -> ToyBranchKeyGenerator:
        """Return a fresh generator over the same action labels."""
        return ToyBranchKeyGenerator(self._keys)


@dataclass(slots=True)
class ToyState:
    """Tiny deterministic state for a declarative toy scenario node."""

    scenario_spec: ToyScenarioSpec = field(repr=False)
    node_id: str
    turn: Color
    adversarial_turns: bool = False

    @property
    def tag(self) -> StateTag:
        """Return the human-readable state tag used by the debug views."""
        node_spec = self.scenario_spec.nodes[self.node_id]
        return node_spec.state_tag or self.node_id

    @property
    def branch_keys(self) -> ToyBranchKeyGenerator:
        """Return the legal branch labels from this state."""
        node_spec = self.scenario_spec.nodes[self.node_id]
        return ToyBranchKeyGenerator(tuple(node_spec.children))

    def branch_name_from_key(self, key: BranchKey) -> str:
        """Return the branch label for ``key``."""
        return str(key)

    def is_game_over(self) -> bool:
        """Return True when this scenario node is terminal."""
        return self.scenario_spec.nodes[self.node_id].terminal_value is not None

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> ToyState:
        """Return a shallow state copy compatible with ``valanga.State``."""
        del stack, deep_copy_legal_moves
        return ToyState(
            scenario_spec=self.scenario_spec,
            node_id=self.node_id,
            turn=self.turn,
            adversarial_turns=self.adversarial_turns,
        )

    def step(self, branch_key: BranchKey) -> None:
        """Mutate the state in place for compatibility with ``valanga.State``."""
        branch_name = _require_branch_name(branch_key)
        node_spec = self.scenario_spec.nodes[self.node_id]
        child_id = node_spec.children[branch_name]
        child_spec = self.scenario_spec.nodes[child_id]
        self.node_id = child_id
        self.turn = _turn_for_child(
            child_player=child_spec.player,
            current_turn=self.turn,
            adversarial_turns=self.adversarial_turns,
        )

    def __str__(self) -> str:
        """Return a concise state description for debug output."""
        return f"{self.scenario_spec.name}:{self.tag}"

    def pprint(self) -> str:
        """Return the pretty-print representation expected by ``valanga.State``."""
        return str(self)


class ToyTerminalOverDetector(OverEventDetector):
    """Detect toy terminal states and return explicit terminal scores."""

    def __init__(self, scenario_spec: ToyScenarioSpec) -> None:
        self._scenario_spec = scenario_spec

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return a terminal over-event + value when the toy node is terminal."""
        if not isinstance(state, ToyState):
            return None, None
        node_spec = self._scenario_spec.nodes[state.node_id]
        if node_spec.terminal_value is None:
            return None, None
        return _terminal_over_event(node_spec.terminal_value), node_spec.terminal_value


class ToyStateValueEvaluator(MasterStateValueEvaluator):
    """Evaluate toy states while leaving backup semantics to real Anemone code."""

    over: OverEventDetector

    def __init__(self, scenario_spec: ToyScenarioSpec) -> None:
        """Build a direct evaluator over one declarative toy scenario."""
        self._scenario_spec = scenario_spec
        self.over = ToyTerminalOverDetector(scenario_spec)

    def evaluate(self, state: State) -> Any:
        """Return a direct heuristic ``Value`` for one toy state."""
        from valanga.evaluations import (  # type: ignore[attr-defined]  # pylint: disable=import-outside-toplevel,no-name-in-module
            Certainty,
            Value,
        )

        return Value(
            score=self._score_for_state(cast("ToyState", state)),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Any]:
        """Evaluate a batch of items, accepting either EvalItems or nodes."""
        values: list[Any] = []
        for item in items:
            state = getattr(item, "state", item)
            values.append(self.evaluate(cast("State", state)))
        return values

    def _score_for_state(self, state: ToyState) -> float:
        node_spec = self._scenario_spec.nodes[state.node_id]
        if node_spec.terminal_value is not None:
            return node_spec.terminal_value
        if node_spec.heuristic_value is not None:
            return node_spec.heuristic_value
        return 0.0


class ToyDynamics(SearchDynamics[ToyState, str]):
    """Search-time dynamics for one declarative toy scenario."""

    __anemone_search_dynamics__ = True

    def __init__(self, scenario_spec: ToyScenarioSpec) -> None:
        """Initialize the thin search-dynamics adapter for one toy scenario."""
        self._scenario_spec = scenario_spec
        self._adversarial_turns = _uses_adversarial_turns(scenario_spec)
        self._over_detector = ToyTerminalOverDetector(scenario_spec)

    def legal_actions(self, state: ToyState) -> Any:
        """Return all legal action labels from ``state``."""
        node_spec = self._scenario_spec.nodes[state.node_id]
        return ToyBranchKeyGenerator(tuple(node_spec.children))

    def step(
        self, state: ToyState, action: str, *, depth: int
    ) -> Transition[ToyState]:
        """Apply one action and return the resulting transition."""
        del depth
        child_id = self._scenario_spec.nodes[state.node_id].children[action]
        child_spec = self._scenario_spec.nodes[child_id]
        next_state = ToyState(
            scenario_spec=self._scenario_spec,
            node_id=child_id,
            turn=_turn_for_child(
                child_player=child_spec.player,
                current_turn=state.turn,
                adversarial_turns=self._adversarial_turns,
            ),
            adversarial_turns=self._adversarial_turns,
        )
        over_event, _terminal_value = self._over_detector.check_obvious_over_events(
            next_state
        )
        return Transition(
            next_state=next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=over_event,
            info={},
        )

    def action_name(self, state: ToyState, action: str) -> str:
        """Return the human-readable action label."""
        del state
        return action

    def action_from_name(self, state: ToyState, name: str) -> str:
        """Resolve a branch label back into the canonical toy action."""
        branch_name = name.strip()
        if branch_name not in self._scenario_spec.nodes[state.node_id].children:
            raise InvalidToyActionNameError(
                node_id=state.node_id,
                action_name=name,
            )
        return branch_name


def create_real_tree_exploration_for_toy_scenario(
    scenario_spec: ToyScenarioSpec,
) -> TreeExploration[Any]:
    """Assemble a real ``TreeExploration`` for one toy scenario.

    The resulting exploration object is built from the production Anemone search
    components through the same high-level branch-selector factory used by the
    main package. The toy layer only contributes the tiny domain adapters.
    """
    dynamics = ToyDynamics(scenario_spec)
    master_evaluator = ToyStateValueEvaluator(scenario_spec)
    branch_selector = create_tree_and_value_branch_selector_with_tree_eval_factory(
        state_type=ToyState,
        dynamics=dynamics,
        args=_build_tree_and_value_args(scenario_spec),
        random_generator=Random(0),
        master_state_evaluator=master_evaluator,
        state_representation_factory=None,
        node_tree_evaluation_factory=cast(
            "Any", _build_node_tree_evaluation_factory(scenario_spec)
        ),
        hooks=None,
    )

    return branch_selector.create_tree_exploration(
        state=ToyState(
            scenario_spec=scenario_spec,
            node_id=scenario_spec.root_id,
            turn=_root_turn(scenario_spec),
            adversarial_turns=_uses_adversarial_turns(scenario_spec),
        ),
        notify_progress=None,
    )


def _build_tree_and_value_args(scenario_spec: ToyScenarioSpec) -> TreeAndValuePlayerArgs:
    """Return the production search configuration used for toy-domain runs.

    The toy scenarios intentionally use one clear, documented real-engine policy:

    * uniform node selection
    * all-children opening
    * tree-branch-limit stopping sized to the declarative toy graph
    * softmax recommender at the end of the run
    """
    return TreeAndValuePlayerArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=_build_stopping_criterion_args(scenario_spec),
        recommender_rule=SoftmaxRule(
            type="softmax",
            temperature=1.0,
        ),
        index_computation=None,
    )


def _build_stopping_criterion_args(scenario_spec: ToyScenarioSpec) -> TreeBranchLimitArgs:
    return TreeBranchLimitArgs(
        type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
        tree_branch_limit=_total_edge_count(scenario_spec),
    )


def _build_node_tree_evaluation_factory(
    scenario_spec: ToyScenarioSpec,
) -> NodeTreeMinmaxEvaluationFactory[Any] | NodeMaxEvaluationFactory[Any]:
    if _uses_adversarial_turns(scenario_spec):
        return NodeTreeMinmaxEvaluationFactory()
    return NodeMaxEvaluationFactory()


def _total_edge_count(scenario_spec: ToyScenarioSpec) -> int:
    return sum(len(node_spec.children) for node_spec in scenario_spec.nodes.values())


def _uses_adversarial_turns(scenario_spec: ToyScenarioSpec) -> bool:
    non_terminal_roles = {
        node_spec.player
        for node_spec in scenario_spec.nodes.values()
        if node_spec.children
    }
    if non_terminal_roles <= {"single"}:
        return False
    if non_terminal_roles <= {"max", "min"}:
        return True
    raise InvalidToyScenarioRolesError


def _root_turn(scenario_spec: ToyScenarioSpec) -> Color:
    root_player = scenario_spec.nodes[scenario_spec.root_id].player
    return _color_for_player(root_player)


def _turn_for_child(
    *,
    child_player: ToyPlayerRole,
    current_turn: Color,
    adversarial_turns: bool,
) -> Color:
    if child_player == "max":
        return Color.WHITE
    if child_player == "min":
        return Color.BLACK
    if adversarial_turns:
        return Color.BLACK if current_turn == Color.WHITE else Color.WHITE
    return Color.WHITE


def _color_for_player(player: ToyPlayerRole) -> Color:
    if player == "min":
        return Color.BLACK
    return Color.WHITE


def _terminal_over_event(score: float) -> OverEvent:
    from valanga.over_event import (  # pylint: disable=import-outside-toplevel
        HowOver,
        Winner,
    )

    if score > 0:
        return OverEvent(how_over=HowOver.WIN, who_is_winner=Winner.WHITE)
    if score < 0:
        return OverEvent(how_over=HowOver.WIN, who_is_winner=Winner.BLACK)
    return OverEvent(how_over=HowOver.DRAW, who_is_winner=Winner.NO_KNOWN_WINNER)


def _require_branch_name(branch_key: BranchKey) -> str:
    if not isinstance(branch_key, str):
        raise InvalidToyBranchKeyError(branch_key=branch_key)
    return branch_key


__all__ = [
    "ToyDynamics",
    "ToyState",
    "ToyStateValueEvaluator",
    "create_real_tree_exploration_for_toy_scenario",
]
