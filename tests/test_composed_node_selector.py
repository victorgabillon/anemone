"""Tests for composed node selector behavior."""

# ruff: noqa: ANN001, ANN201, D101, D102, D103

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
    latest_selection_report: object | None = None
    calls: int = 0
    invalidations: int = 0
    refresh_calls: int = 0
    build_calls: int = 0
    restore_calls: int = 0
    payload_to_return: object | None = None
    restore_result: bool = True

    def choose_node_and_branch_to_open(self, tree, latest_tree_expansions):
        _ = tree
        _ = latest_tree_expansions
        self.calls += 1
        return self.result

    def invalidate(self) -> None:
        self.invalidations += 1

    def refresh_state_for_checkpoint(
        self,
        *,
        tree,
        objective,
        latest_tree_expansions,
    ) -> None:
        _ = tree
        _ = objective
        _ = latest_tree_expansions
        self.refresh_calls += 1

    def build_checkpoint_payload(self, objective):
        _ = objective
        self.build_calls += 1
        return self.payload_to_return

    def restore_from_checkpoint_payload(self, *, tree, objective, payload) -> bool:
        _ = tree
        _ = objective
        _ = payload
        self.restore_calls += 1
        return self.restore_result


@dataclass
class DummyBaseSelectorWithoutInvalidate:
    result: OpeningInstructions

    def choose_node_and_branch_to_open(self, tree, latest_tree_expansions):
        _ = tree
        _ = latest_tree_expansions
        return self.result


def test_composed_selector_falls_back_to_base() -> None:
    base_result = OpeningInstructions()
    selection_report = object()
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelector(
            result=base_result,
            latest_selection_report=selection_report,
        ),
    )

    result = selector.choose_node_and_branch_to_open(
        tree=None, latest_tree_expansions=None
    )

    assert result is base_result
    assert selector.base.calls == 1
    assert selector.latest_selection_report is selection_report


def test_composed_selector_priority_wins() -> None:
    priority_result = OpeningInstructions()
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=priority_result),
        base=DummyBaseSelector(result=OpeningInstructions()),
    )
    object.__setattr__(selector, "latest_selection_report", object())

    result = selector.choose_node_and_branch_to_open(
        tree=None, latest_tree_expansions=None
    )

    assert result is priority_result
    assert selector.base.calls == 0
    assert selector.latest_selection_report is None


def test_composed_selector_str_mentions_both_parts() -> None:
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelector(result=OpeningInstructions()),
    )

    text = str(selector)

    assert "ComposedNodeSelector" in text
    assert "base=" in text
    assert "priority=" in text


def test_composed_selector_forwards_invalidate_to_base() -> None:
    base = DummyBaseSelector(result=OpeningInstructions())
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=base,
    )

    selector.invalidate()

    assert base.invalidations == 1


def test_composed_selector_invalidate_is_noop_when_base_has_no_invalidate() -> None:
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelectorWithoutInvalidate(result=OpeningInstructions()),
    )

    selector.invalidate()


def test_composed_selector_forwards_refresh_state_for_checkpoint_to_base() -> None:
    base = DummyBaseSelector(result=OpeningInstructions())
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=base,
    )

    selector.refresh_state_for_checkpoint(
        tree=object(),
        objective=object(),
        latest_tree_expansions=object(),
    )

    assert base.refresh_calls == 1


def test_composed_selector_forwards_build_checkpoint_payload_to_base() -> None:
    payload = object()
    base = DummyBaseSelector(
        result=OpeningInstructions(),
        payload_to_return=payload,
    )
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=base,
    )

    assert selector.build_checkpoint_payload(objective=object()) is payload
    assert base.build_calls == 1


def test_composed_selector_forwards_restore_checkpoint_payload_to_base() -> None:
    base = DummyBaseSelector(result=OpeningInstructions())
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=base,
    )

    restored = selector.restore_from_checkpoint_payload(
        tree=object(),
        objective=object(),
        payload=object(),
    )

    assert restored is True
    assert base.restore_calls == 1


def test_composed_selector_checkpoint_methods_are_noop_without_base_support() -> None:
    selector = ComposedNodeSelector(
        priority_check=DummyPriorityCheck(result=None),
        base=DummyBaseSelectorWithoutInvalidate(result=OpeningInstructions()),
    )

    selector.refresh_state_for_checkpoint(
        tree=object(),
        objective=object(),
        latest_tree_expansions=object(),
    )

    assert selector.build_checkpoint_payload(objective=object()) is None
    assert (
        selector.restore_from_checkpoint_payload(
            tree=object(),
            objective=object(),
            payload=object(),
        )
        is False
    )
