from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import valanga
from valanga import BranchKey, Color, OverEvent, State, StateModifications, StateTag

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.node_direct_evaluation.protocols import (
    MasterStateValueEvaluator,
)
from anemone.node_evaluation.node_direct_evaluation.protocols import OverEventDetector
from anemone.values import Certainty, Value

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from valanga.evaluations import EvalItem


def build_yaml_maps(
    yaml_nodes: list[dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, float]]:
    children_by_id: dict[int, list[int]] = {}
    value_by_id: dict[int, float] = {}

    for n in yaml_nodes:
        node_id = int(n["id"])
        children_by_id.setdefault(node_id, [])
        value_by_id[node_id] = float(n["value"])

    for n in yaml_nodes:
        parent = n.get("parents", None)

        # accept multiple "no parent" conventions
        if parent is None or parent == "None" or parent == "null":
            continue

        parent_id = int(parent)
        node_id = int(n["id"])
        children_by_id.setdefault(parent_id, []).append(node_id)

    return children_by_id, value_by_id


class FakeBranchKeyGenerator:
    """Generates ordinal branch keys 0..n-1.
    This matches your OpeningInstruction(branch=0) usage.
    """

    sort_branch_keys: bool = False

    def __init__(self, keys: Sequence[int]) -> None:
        self._keys = list(keys)
        self._i = 0

    @property
    def all_generated_keys(self) -> Sequence[int] | None:
        return self._keys

    def __iter__(self) -> Iterator[int]:
        self._i = 0
        return self

    def __next__(self) -> int:
        if self._i >= len(self._keys):
            raise StopIteration
        k = self._keys[self._i]
        self._i += 1
        return k

    def more_than_one(self) -> bool:
        return len(self._keys) > 1

    def get_all(self) -> Sequence[int]:
        return list(self._keys)

    def copy_with_reset(self) -> Self:
        return FakeBranchKeyGenerator(self._keys)


@dataclass(slots=True)
class FakeYamlState(State):
    """A minimal State that represents being at YAML node_id.

    IMPORTANT: step() mutates self, to be compatible with ValangaStateTransition.
    """

    node_id: int
    children_by_id: dict[int, list[int]]
    turn: Color = Color.WHITE

    @property
    def tag(self) -> StateTag:
        # TreeManager uses state.tag for dedup at depth
        return int(self.node_id)

    @property
    def branch_keys(self) -> FakeBranchKeyGenerator:
        children = self.children_by_id.get(self.node_id, [])
        return FakeBranchKeyGenerator(range(len(children)))

    def branch_name_from_key(self, key: BranchKey) -> str:
        try:
            child_id = self._child_id_from_branch(key)
            return f"{self.node_id}->{child_id}"
        except Exception:
            return f"{self.node_id}->{key}"

    def is_game_over(self) -> bool:
        return len(self.children_by_id.get(self.node_id, [])) == 0

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> Self:
        return FakeYamlState(
            node_id=self.node_id,
            children_by_id=self.children_by_id,
            turn=self.turn,
        )

    def step(self, branch_key: BranchKey) -> StateModifications | None:
        """Mutate in place to satisfy ValangaStateTransition.
        Branch key is ordinal 0..n-1 -> picks the corresponding child in YAML order.
        """
        child_id = self._child_id_from_branch(branch_key)
        self.node_id = child_id
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return None

    def _child_id_from_branch(self, branch_key: BranchKey) -> int:
        children = self.children_by_id.get(self.node_id, [])
        if not children:
            raise ValueError(
                f"Node {self.node_id} has no children; cannot step({branch_key})."
            )

        if not isinstance(branch_key, int):
            raise TypeError(
                f"branch_key must be int ordinal, got {type(branch_key)}: {branch_key!r}"
            )

        if branch_key < 0 or branch_key >= len(children):
            raise ValueError(
                f"Invalid branch {branch_key} for node {self.node_id}. "
                f"Expected 0..{len(children) - 1} (children ids={children})."
            )

        return children[branch_key]


class NeverOverDetector(OverEventDetector):
    """Always says 'not over'."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        return None, None


class MasterStateValueEvaluatorFromYaml(MasterStateValueEvaluator):
    """Returns the YAML value for the node id stored in state.tag."""

    over: OverEventDetector

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Value]:
        scores = self.value_white_batch_items(items)
        return [
            Value(score=float(score), certainty=Certainty.ESTIMATE, over_event=None)
            for score in scores
        ]

    def evaluate(self, state: State) -> Value:
        return Value(
            score=self.value_white(state),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )

    def __init__(self, value_by_id: dict[int, float]) -> None:
        self._value_by_id = value_by_id
        self.over = NeverOverDetector()

    def value_white(self, state: State) -> float:
        node_id = int(state.tag)
        try:
            return float(self._value_by_id[node_id])
        except KeyError as e:
            raise KeyError(f"No YAML value for node id={node_id}") from e

    def value_white_batch_items(self, items: Sequence[Any]) -> list[float]:
        # NodeDirectEvaluator in your code passes nodes, despite the EvalItem type hint.
        out: list[float] = []
        for it in items:
            st = getattr(it, "state", it)
            out.append(self.value_white(st))
        return out


class FakeYamlDynamics(SearchDynamics[FakeYamlState, Any]):
    """Search-time dynamics for FakeYamlState.

    BranchKey is an ordinal int 0..n-1 selecting the corresponding child in YAML order.
    """

    __anemone_search_dynamics__ = True

    def legal_actions(
        self, state: FakeYamlState
    ) -> valanga.BranchKeyGeneratorP[BranchKey]:
        # FakeYamlState.branch_keys already returns a BranchKey generator over ordinals.
        return state.branch_keys

    def step(
        self, state: FakeYamlState, action: BranchKey, *, depth: int
    ) -> valanga.Transition[FakeYamlState]:
        # depth is intentionally ignored for this fake game
        del depth

        # Compute next state without mutating input state
        child_id = state._child_id_from_branch(action)
        next_turn = Color.BLACK if state.turn == Color.WHITE else Color.WHITE

        next_state = FakeYamlState(
            node_id=child_id,
            children_by_id=state.children_by_id,
            turn=next_turn,
        )

        # For tests, we don't model over_event; keep it None.
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=None,
            info={},
        )

    def action_name(self, state: FakeYamlState, action: BranchKey) -> str:
        # Reuse your existing naming logic (node_id->child_id)
        return state.branch_name_from_key(action)

    def action_from_name(self, state: FakeYamlState, name: str) -> BranchKey:
        """Parse a name back into an ordinal branch key.

        Accepts:
        - "PARENT->CHILD" (canonical)
        - "CHILD" (fallback)
        """
        text = name.strip()

        # Accept "parent->child" or just "child"
        if "->" in text:
            left, right = text.split("->", 1)
            left = left.strip()
            right = right.strip()

            # If left is present and doesn't match current node, be strict:
            # This prevents confusing cross-state parsing.
            if left and int(left) != int(state.node_id):
                raise ValueError(
                    f"Move name {name!r} does not apply to state node_id={state.node_id}"
                )

            child_id = int(right)
        else:
            child_id = int(text)

        children = state.children_by_id.get(state.node_id, [])
        try:
            ordinal = children.index(child_id)
        except ValueError as e:
            raise ValueError(
                f"State node_id={state.node_id} has no child_id={child_id} "
                f"(children={children}); cannot parse action_from_name({name!r})."
            ) from e

        return ordinal
