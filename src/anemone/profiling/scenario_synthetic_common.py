"""Shared deterministic synthetic profiling scenarios."""
# pylint: disable=duplicate-code,useless-return

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from random import Random
from typing import TYPE_CHECKING, Any, Literal

import valanga
from valanga import BranchKey, Color, State, StateModifications, StateTag

from anemone.dynamics import SearchDynamics
from anemone.factory import SearchArgs, create_search
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.profiling.collectors import wrap_profiled_components
from anemone.profiling.scenario_runtime import ScenarioRuntime, ScenarioRuntimeOptions
from anemone.profiling.scenarios import ProfilingScenario
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import SoftmaxRule
from anemone.utils.logger import anemone_logger, suppress_logging

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from anemone._valanga_types import AnyOverEvent

EvaluatorMode = Literal["cheap", "expensive"]
ReusePattern = Literal["tree", "shared_last_layer", "diamond", "chain"]

_VALANGA_EVALUATIONS = import_module("valanga.evaluations")
Certainty = _VALANGA_EVALUATIONS.Certainty
Value = _VALANGA_EVALUATIONS.Value


@dataclass(frozen=True, slots=True)
class SyntheticScenarioConfig:
    """Deterministic shape and cost parameters for one synthetic scenario."""

    name: str
    description: str
    branching_factor: int
    max_depth: int
    evaluator_mode: EvaluatorMode
    evaluator_work_units: int
    reuse_pattern: ReusePattern
    stopping_branch_limit: int
    random_seed: int


class _OrdinalBranchKeyGenerator:
    """Deterministic ordinal branch-key iterator."""

    sort_branch_keys: bool = False

    def __init__(self, keys: Sequence[int]) -> None:
        self._keys = list(keys)
        self._index = 0

    @property
    def all_generated_keys(self) -> Sequence[int] | None:
        """Return the generated keys as a sequence."""
        return self._keys

    def __iter__(self) -> Iterator[int]:
        """Reset iteration over generated branch keys."""
        self._index = 0
        return self

    def __next__(self) -> int:
        """Return the next generated branch key."""
        if self._index >= len(self._keys):
            raise StopIteration
        key = self._keys[self._index]
        self._index += 1
        return key

    def more_than_one(self) -> bool:
        """Return whether more than one branch is available."""
        return len(self._keys) > 1

    def get_all(self) -> Sequence[int]:
        """Return all generated keys."""
        return list(self._keys)

    def copy_with_reset(self) -> _OrdinalBranchKeyGenerator:
        """Return a fresh generator over the same branch keys."""
        return _OrdinalBranchKeyGenerator(self._keys)


@dataclass(slots=True)
class SyntheticState(State):
    """Minimal deterministic synthetic state for profiling scenarios."""

    config: SyntheticScenarioConfig
    depth: int
    node_key: int
    turn: Color = Color.WHITE

    @property
    def tag(self) -> StateTag:
        """Return a stable deduplication tag for the state."""
        return int(self.depth * 1_000_000_000 + self.node_key)

    @property
    def branch_keys(self) -> _OrdinalBranchKeyGenerator:
        """Return the legal branch keys from the current node."""
        return _OrdinalBranchKeyGenerator(range(self.branch_count()))

    def branch_count(self) -> int:
        """Return how many legal actions are available in this state."""
        if self.depth >= self.config.max_depth:
            return 0
        if self.config.reuse_pattern == "chain":
            return 1
        return self.config.branching_factor

    def branch_name_from_key(self, key: BranchKey) -> str:
        """Return a readable branch name for one ordinal move."""
        child_key = self.child_key_from_branch(key)
        return f"d{self.depth}:n{self.node_key}->n{child_key}"

    def is_game_over(self) -> bool:
        """Return whether this state is terminal."""
        return self.depth >= self.config.max_depth

    def copy(
        self,
        stack: bool,
        deep_copy_legal_moves: bool = True,
    ) -> SyntheticState:
        """Return a shallow copy compatible with the Valanga state contract."""
        del stack, deep_copy_legal_moves
        return SyntheticState(
            config=self.config,
            depth=self.depth,
            node_key=self.node_key,
            turn=self.turn,
        )

    def step(self, branch_key: BranchKey) -> StateModifications | None:
        """Mutate in place so the state matches the Valanga state contract."""
        self.depth += 1
        self.node_key = self.child_key_from_branch(branch_key)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return None

    def pprint(self) -> str:
        """Return a compact human-readable representation of the state."""
        return (
            "SyntheticState("
            f"name={self.config.name}, depth={self.depth}, node_key={self.node_key}"
            f", turn={self.turn.name})"
        )

    def child_key_from_branch(self, branch_key: BranchKey) -> int:
        """Resolve one branch key to the corresponding child node key."""
        branch_index = _validated_branch_index(branch_key, self.branch_count())
        next_depth = self.depth + 1

        if self.config.reuse_pattern == "chain":
            return self.node_key + 1
        if self.config.reuse_pattern == "diamond":
            return (self.node_key + branch_index) % max(2, self.config.branching_factor)
        if (
            self.config.reuse_pattern == "shared_last_layer"
            and next_depth == self.config.max_depth
        ):
            return branch_index % max(1, self.config.branching_factor // 2)
        return self.node_key * self.config.branching_factor + branch_index + 1


class _NeverOverDetector(OverEventDetector):
    """Synthetic scenarios rely on transition terminality, not explicit events."""

    def check_obvious_over_events(
        self,
        state: State,
    ) -> tuple[AnyOverEvent | None, float | None]:
        del state
        return None, None


class _SyntheticValueEvaluator(MasterStateValueEvaluator):
    """Deterministic synthetic evaluator with cheap and expensive modes."""

    over: OverEventDetector

    def __init__(self, config: SyntheticScenarioConfig) -> None:
        self._config = config
        self.over = _NeverOverDetector()

    def evaluate_batch_items(
        self,
        items: Sequence[Any],
    ) -> list[Any]:
        return [
            Value(
                score=self.value_white(getattr(item, "state", item)),
                certainty=Certainty.ESTIMATE,
                over_event=None,
            )
            for item in items
        ]

    def evaluate(self, state: State) -> Any:
        """Return the deterministic synthetic value for one state."""
        return Value(
            score=self.value_white(state),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )

    def value_white(self, state: State) -> float:
        """Return a deterministic scalar score for one synthetic state."""
        synthetic_state = _require_synthetic_state(state)
        if self._config.evaluator_mode == "cheap":
            return _cheap_score(synthetic_state)
        return _expensive_score(
            synthetic_state,
            work_units=self._config.evaluator_work_units,
        )

    def value_white_batch_items(self, items: Sequence[Any]) -> list[float]:
        """Return deterministic scalar values for a batch of synthetic states."""
        return [self.value_white(getattr(item, "state", item)) for item in items]


class _SyntheticDynamics(SearchDynamics[SyntheticState, int]):
    """Deterministic dynamics used by synthetic profiling scenarios."""

    __anemone_search_dynamics__ = True

    def legal_actions(
        self,
        state: SyntheticState,
    ) -> valanga.BranchKeyGeneratorP[int]:
        """Return legal ordinal branch keys."""
        return state.branch_keys

    def step(
        self,
        state: SyntheticState,
        action: int,
        *,
        depth: int,
    ) -> valanga.Transition[SyntheticState]:
        """Return the next deterministic synthetic search state."""
        del depth
        child_key = state.child_key_from_branch(action)
        next_turn = Color.BLACK if state.turn == Color.WHITE else Color.WHITE
        next_state = SyntheticState(
            config=state.config,
            depth=state.depth + 1,
            node_key=child_key,
            turn=next_turn,
        )
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=None,
            info={},
        )

    def action_name(self, state: SyntheticState, action: int) -> str:
        """Return a readable action name for one branch."""
        return state.branch_name_from_key(action)

    def action_from_name(self, state: SyntheticState, name: str) -> int:
        """Resolve a branch name back to its ordinal index."""
        text = name.strip()
        if "branch_" not in text:
            return int(text)
        return int(text.split("branch_", 1)[1])


def build_synthetic_runtime(
    config: SyntheticScenarioConfig,
    options: ScenarioRuntimeOptions,
) -> ScenarioRuntime:
    """Build one synthetic profiling runtime from a deterministic config."""
    starting_state = SyntheticState(
        config=config,
        depth=0,
        node_key=0,
        turn=Color.WHITE,
    )
    dynamics: SearchDynamics[SyntheticState, Any] = _SyntheticDynamics()
    evaluator: MasterStateValueEvaluator = _SyntheticValueEvaluator(config)
    component_collectors = None
    if options.component_summary:
        wrapped_evaluator, wrapped_dynamics, component_collectors = (
            wrap_profiled_components(
                evaluator=evaluator,
                dynamics=dynamics,
            )
        )
        assert wrapped_evaluator is not None
        assert wrapped_dynamics is not None
        evaluator = wrapped_evaluator
        dynamics = wrapped_dynamics

    exploration = create_search(
        state_type=SyntheticState,
        dynamics=dynamics,
        starting_state=starting_state,
        args=_build_search_args(config),
        random_generator=Random(config.random_seed),
        master_state_value_evaluator=evaluator,
        state_representation_factory=None,
    )

    def runner() -> None:
        with suppress_logging(anemone_logger):
            result = exploration.explore(random_generator=Random(config.random_seed))

        if exploration.tree.nodes_count <= 0:
            raise _empty_tree_error(config.name)
        if not result.branch_recommendation.recommended_name:
            raise _missing_recommendation_error(config.name)

    return ScenarioRuntime(
        runner=runner,
        component_collectors=component_collectors,
    )


def build_synthetic_scenario(config: SyntheticScenarioConfig) -> ProfilingScenario:
    """Build one profiling scenario descriptor from a synthetic config."""

    def runner() -> None:
        build_synthetic_runtime(
            config,
            ScenarioRuntimeOptions(),
        ).runner()

    def runtime_builder(options: ScenarioRuntimeOptions) -> ScenarioRuntime:
        return build_synthetic_runtime(config, options)

    return ProfilingScenario(
        name=config.name,
        description=config.description,
        runner=runner,
        runtime_builder=runtime_builder,
    )


def _build_search_args(config: SyntheticScenarioConfig) -> SearchArgs:
    """Build a stable search configuration for one synthetic scenario."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=config.stopping_branch_limit,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=None,
    )


def _validated_branch_index(branch_key: BranchKey, branch_count: int) -> int:
    if not isinstance(branch_key, int):
        raise _branch_key_type_error(branch_key)
    if branch_key < 0 or branch_key >= branch_count:
        raise _invalid_branch_error(branch_key, branch_count)
    return branch_key


def _require_synthetic_state(state: State) -> SyntheticState:
    if not isinstance(state, SyntheticState):
        raise _state_type_error(state)
    return state


def _cheap_score(state: SyntheticState) -> float:
    base = (state.node_key * 17) + (state.depth * 11)
    if state.turn == Color.BLACK:
        base += 5
    return (base % 97) / 32.0 - 1.5


def _expensive_score(state: SyntheticState, *, work_units: int) -> float:
    accumulator = (state.node_key * 31) + (state.depth * 17)
    if state.turn == Color.BLACK:
        accumulator += 7
    for index in range(work_units):
        accumulator = ((accumulator * 33) + index + 17) % 104_729
    return accumulator / 52_364.5 - 1.0


def _branch_key_type_error(branch_key: object) -> TypeError:
    return TypeError(
        f"Synthetic scenario branch_key must be an int, got {type(branch_key)}"
    )


def _invalid_branch_error(branch_key: int, branch_count: int) -> ValueError:
    return ValueError(
        f"Invalid synthetic branch {branch_key}; expected 0..{branch_count - 1}"
    )


def _state_type_error(state: object) -> TypeError:
    return TypeError(
        f"Synthetic scenario evaluator expected SyntheticState, got {type(state)}"
    )


def _empty_tree_error(scenario_name: str) -> RuntimeError:
    return RuntimeError(
        f"Synthetic profiling scenario {scenario_name!r} produced no tree"
    )


def _missing_recommendation_error(scenario_name: str) -> RuntimeError:
    return RuntimeError(
        f"Synthetic profiling scenario {scenario_name!r} did not produce a recommendation"
    )
