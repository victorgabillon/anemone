from dataclasses import dataclass
from types import SimpleNamespace

from valanga import Color

from anemone.node_selector.uniform.uniform import Uniform
from anemone.values import Certainty, Value


@dataclass
class _FakeEval:
    value: Value | None
    terminal: bool = False

    def is_terminal_candidate(self) -> bool:
        return self.terminal

    def get_value(self) -> Value:
        assert self.value is not None
        return self.value


@dataclass
class _FakeNode:
    id: int
    tree_evaluation: _FakeEval


class _FakeOpeningInstructor:
    def all_branches_to_open(self, node_to_open: _FakeNode) -> list[int]:
        _ = node_to_open
        return [0]


def _instruction_node_ids(opening_instructions) -> list[int]:
    return [instr.node_to_open.id for instr in opening_instructions.values()]


def test_uniform_selector_uses_projection_search_ordering() -> None:
    selector = Uniform(opening_instructor=_FakeOpeningInstructor())

    # For WHITE search_sort_key, large positive estimate gets a lower key
    # than terminal win projection (+1), so estimate should be expanded first.
    estimate = _FakeNode(
        id=1,
        tree_evaluation=_FakeEval(
            value=Value(score=50.0, certainty=Certainty.ESTIMATE, over_event=None)
        ),
    )
    forced_win = _FakeNode(
        id=2,
        tree_evaluation=_FakeEval(
            value=Value(score=0.0, certainty=Certainty.FORCED, over_event=None)
        ),
    )

    tree = SimpleNamespace(
        tree_root_tree_depth=0,
        descendants={0: {estimate.id: estimate, forced_win.id: forced_win}},
        root_node=SimpleNamespace(state=SimpleNamespace(turn=Color.WHITE)),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _instruction_node_ids(instructions) == [1, 2]


def test_uniform_selector_skips_terminal_candidates() -> None:
    selector = Uniform(opening_instructor=_FakeOpeningInstructor())

    terminal = _FakeNode(
        id=10,
        tree_evaluation=_FakeEval(
            value=Value(score=0.0, certainty=Certainty.FORCED, over_event=None),
            terminal=True,
        ),
    )
    live = _FakeNode(
        id=11,
        tree_evaluation=_FakeEval(
            value=Value(score=0.1, certainty=Certainty.ESTIMATE, over_event=None)
        ),
    )

    tree = SimpleNamespace(
        tree_root_tree_depth=0,
        descendants={0: {terminal.id: terminal, live.id: live}},
        root_node=SimpleNamespace(state=SimpleNamespace(turn=Color.WHITE)),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _instruction_node_ids(instructions) == [11]
