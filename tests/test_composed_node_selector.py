"""Tests for composed node selector behavior."""

from dataclasses import dataclass

from anemone.node_selector.composed.composed_node_selector import ComposedNodeSelector
from anemone.node_selector.opening_instructions import OpeningInstructions


@dataclass
class DummyPriorityCheck:
    result: OpeningInstructions | None

    def maybe_choose_opening(self, tree, latest_tree_expansions):
        _ = tree
        _ = latest_tree_expansions
        return self.result


@dataclass
class DummyBaseSelector:
    result: OpeningInstructions
    calls: int = 0

    def choose_node_and_branch_to_open(self, tree, latest_tree_expansions):
        _ = tree
        _ = latest_tree_expansions
        self.calls += 1
        return self.result


def test_composed_selector_falls_back_to_base() -> None:
    base_result = OpeningInstructions()
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelector(result=base_result),
    )

    result = selector.choose_node_and_branch_to_open(
        tree=None, latest_tree_expansions=None
    )

    assert result is base_result
    assert selector.base.calls == 1


def test_composed_selector_priority_wins() -> None:
    priority_result = OpeningInstructions()
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=priority_result),
        base=DummyBaseSelector(result=OpeningInstructions()),
    )

    result = selector.choose_node_and_branch_to_open(
        tree=None, latest_tree_expansions=None
    )

    assert result is priority_result
    assert selector.base.calls == 0


def test_composed_selector_str_mentions_both_parts() -> None:
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelector(result=OpeningInstructions()),
    )

    text = str(selector)

    assert "ComposedNodeSelector" in text
    assert "base=" in text
    assert "priority=" in text
