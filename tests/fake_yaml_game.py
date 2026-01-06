from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence, Self

from valanga import BranchKey, Color, OverEvent, State, StateModifications, StateTag
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
    OverEventDetector,
)


def build_yaml_maps(
    yaml_nodes: list[dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """
    Build:
      - children_by_id: parent_id -> ordered list of child ids (YAML order)
      - value_by_id: node_id -> float(value)
    """
    children_by_id: dict[int, list[int]] = {}
    value_by_id: dict[int, float] = {}

    for n in yaml_nodes:
        node_id = int(n["id"])
        children_by_id.setdefault(node_id, [])
        value_by_id[node_id] = float(n["value"])

    for n in yaml_nodes:
        parent = n.get("parents", None)
        if parent is None:
            continue
        parent_id = int(parent)
        node_id = int(n["id"])
        children_by_id.setdefault(parent_id, []).append(node_id)

    return children_by_id, value_by_id


class FakeBranchKeyGenerator:
    """
    Generates ordinal branch keys 0..n-1.
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
    """
    A minimal State that represents being at YAML node_id.

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
        # stack/deep_copy_legal_moves irrelevant for this toy game
        return FakeYamlState(
            node_id=self.node_id,
            children_by_id=self.children_by_id,
            turn=self.turn,
        )

    def step(self, branch_key: BranchKey) -> StateModifications | None:
        """
        Mutate in place to satisfy ValangaStateTransition.
        Branch key is ordinal 0..n-1 -> picks the corresponding child in YAML order.
        """
        child_id = self._child_id_from_branch(branch_key)
        self.node_id = child_id
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return None

    def _child_id_from_branch(self, branch_key: BranchKey) -> int:
        children = self.children_by_id.get(self.node_id, [])
        if not children:
            raise ValueError(f"Node {self.node_id} has no children; cannot step({branch_key}).")

        if not isinstance(branch_key, int):
            raise TypeError(f"branch_key must be int ordinal, got {type(branch_key)}: {branch_key!r}")

        if branch_key < 0 or branch_key >= len(children):
            raise ValueError(
                f"Invalid branch {branch_key} for node {self.node_id}. "
                f"Expected 0..{len(children)-1} (children ids={children})."
            )

        return children[branch_key]


class NeverOverDetector(OverEventDetector):
    """Always says 'not over'."""
    def check_obvious_over_events(self, state: State) -> tuple[OverEvent | None, float | None]:
        return None, None


class MasterStateEvaluatorFromYaml(MasterStateEvaluator):
    """
    Returns the YAML value for the node id stored in state.tag.
    """
    over: OverEventDetector

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
