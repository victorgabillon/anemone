"""Real tiny end-to-end smoke scenario for the profiling foundation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Self

import valanga
from valanga import BranchKey, Color, OverEvent, State, StateModifications, StateTag
from valanga.evaluations import Certainty, Value

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


class _OrdinalBranchKeyGenerator:
    """Generate ordinal branch keys for the tiny smoke scenario tree."""

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
        """Return the next ordinal branch key."""
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

    def copy_with_reset(self) -> Self:
        """Return a fresh generator over the same branch keys."""
        return _OrdinalBranchKeyGenerator(self._keys)


@dataclass(slots=True)
class _SmokeState(State):
    """Minimal concrete state used by the profiling smoke scenario."""

    node_id: int
    children_by_id: dict[int, list[int]]
    turn: Color = Color.WHITE

    @property
    def tag(self) -> StateTag:
        """Return a stable deduplication tag for the state."""
        return int(self.node_id)

    @property
    def branch_keys(self) -> _OrdinalBranchKeyGenerator:
        """Return the legal branch keys from the current node."""
        children = self.children_by_id.get(self.node_id, [])
        return _OrdinalBranchKeyGenerator(range(len(children)))

    def branch_name_from_key(self, key: BranchKey) -> str:
        """Return a readable branch name for one ordinal move."""
        child_id = self._child_id_from_branch(key)
        return f"{self.node_id}->{child_id}"

    def is_game_over(self) -> bool:
        """Return whether the current node has no legal children."""
        return len(self.children_by_id.get(self.node_id, [])) == 0

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> Self:
        """Return a shallow copy compatible with the Valanga state contract."""
        del stack, deep_copy_legal_moves
        return _SmokeState(
            node_id=self.node_id,
            children_by_id=self.children_by_id,
            turn=self.turn,
        )

    def step(self, branch_key: BranchKey) -> StateModifications | None:
        """Mutate in place so the state is compatible with Valanga expectations."""
        self.node_id = self._child_id_from_branch(branch_key)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return None

    def pprint(self) -> str:
        """Return a compact human-readable representation of the state."""
        return f"SmokeState(node_id={self.node_id}, turn={self.turn.name})"

    def _child_id_from_branch(self, branch_key: BranchKey) -> int:
        children = self.children_by_id.get(self.node_id, [])
        if not isinstance(branch_key, int):
            raise TypeError(
                f"Smoke scenario branch_key must be an int, got {type(branch_key)}"
            )
        if branch_key < 0 or branch_key >= len(children):
            raise ValueError(
                f"Invalid smoke branch {branch_key} for node {self.node_id}; "
                f"expected 0..{len(children) - 1}"
            )
        return children[branch_key]


class _NeverOverDetector(OverEventDetector):
    """Report that the smoke scenario never ends via an explicit over event."""

    def check_obvious_over_events(
        self,
        state: State,
    ) -> tuple[OverEvent | None, float | None]:
        del state
        return None, None


class _SmokeValueEvaluator(MasterStateValueEvaluator):
    """Return pre-baked node values for the smoke scenario tree."""

    over: OverEventDetector

    def __init__(self, value_by_id: dict[int, float]) -> None:
        self._value_by_id = value_by_id
        self.over = _NeverOverDetector()

    def evaluate_batch_items[ItemStateT: State](
        self,
        items: Sequence[Any],
    ) -> list[Value]:
        return [
            Value(
                score=self.value_white(getattr(item, "state", item)),
                certainty=Certainty.ESTIMATE,
                over_event=None,
            )
            for item in items
        ]

    def evaluate(self, state: State) -> Value:
        """Return the cached value for one state."""
        return Value(
            score=self.value_white(state),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )

    def value_white(self, state: State) -> float:
        """Return the white-perspective scalar value for one state."""
        return float(self._value_by_id[int(state.tag)])

    def value_white_batch_items(self, items: Sequence[Any]) -> list[float]:
        """Return scalar values for a batch of states or eval items."""
        return [self.value_white(getattr(item, "state", item)) for item in items]


class _SmokeDynamics(SearchDynamics[_SmokeState, Any]):
    """Tiny concrete dynamics used for the profiling smoke scenario."""

    __anemone_search_dynamics__ = True

    def legal_actions(
        self,
        state: _SmokeState,
    ) -> valanga.BranchKeyGeneratorP[BranchKey]:
        """Return legal ordinal branch keys."""
        return state.branch_keys

    def step(
        self,
        state: _SmokeState,
        action: BranchKey,
        *,
        depth: int,
    ) -> valanga.Transition[_SmokeState]:
        """Return the next search state without mutating the input state."""
        del depth
        child_id = state._child_id_from_branch(action)
        next_turn = Color.BLACK if state.turn == Color.WHITE else Color.WHITE
        next_state = _SmokeState(
            node_id=child_id,
            children_by_id=state.children_by_id,
            turn=next_turn,
        )
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=None,
            info={},
        )

    def action_name(self, state: _SmokeState, action: BranchKey) -> str:
        """Return a readable action name for one branch."""
        return state.branch_name_from_key(action)

    def action_from_name(self, state: _SmokeState, name: str) -> BranchKey:
        """Resolve a branch name back to its ordinal index."""
        text = name.strip()
        if "->" in text:
            _, child_text = text.split("->", 1)
            child_id = int(child_text)
        else:
            child_id = int(text)

        children = state.children_by_id.get(state.node_id, [])
        try:
            return children.index(child_id)
        except ValueError as exc:
            raise ValueError(
                f"State node_id={state.node_id} has no child_id={child_id}"
            ) from exc


def _build_smoke_args() -> SearchArgs:
    """Return a tiny but real end-to-end search configuration."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=2,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=None,
    )


def _build_smoke_runtime(options: ScenarioRuntimeOptions) -> ScenarioRuntime:
    """Build the smoke runtime, optionally wrapping injectable components."""
    children_by_id = {
        0: [1, 2],
        1: [],
        2: [],
    }
    value_by_id = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
    }
    starting_state = _SmokeState(
        node_id=0,
        children_by_id=children_by_id,
        turn=Color.WHITE,
    )

    dynamics: SearchDynamics[_SmokeState, Any] = _SmokeDynamics()
    evaluator: MasterStateValueEvaluator = _SmokeValueEvaluator(value_by_id)
    component_collectors = None
    if options.component_summary:
        evaluator, dynamics, component_collectors = wrap_profiled_components(
            evaluator=evaluator,
            dynamics=dynamics,
        )

    exploration = create_search(
        state_type=_SmokeState,
        dynamics=dynamics,
        starting_state=starting_state,
        args=_build_smoke_args(),
        random_generator=Random(0),
        master_state_value_evaluator=evaluator,
        state_representation_factory=None,
    )

    def runner() -> None:
        with suppress_logging(anemone_logger):
            result = exploration.explore(random_generator=Random(0))

        if exploration.tree.nodes_count != 3:
            raise RuntimeError(
                "Smoke profiling scenario expected a 3-node tree, "
                f"got {exploration.tree.nodes_count}"
            )
        if result.branch_recommendation.recommended_name is None:
            raise RuntimeError(
                "Smoke profiling scenario did not produce a recommendation"
            )

    return ScenarioRuntime(
        runner=runner,
        component_collectors=component_collectors,
    )


def _run_smoke_search() -> None:
    """Run a tiny real search via the public API to validate the profiling shell."""
    _build_smoke_runtime(ScenarioRuntimeOptions()).runner()


def build_smoke_scenario() -> ProfilingScenario:
    """Build the lazily-loaded smoke scenario descriptor."""
    return ProfilingScenario(
        name="smoke",
        description="Tiny real end-to-end search using public Anemone APIs.",
        runner=_run_smoke_search,
        runtime_builder=_build_smoke_runtime,
    )
