"""Checkpointed heap priorities must never acquire false validity signatures."""

from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

import pytest

from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.checkpoints.sharded_decode import _selector_payload_from_mapping
from anemone.node_selector.linoo import Linoo
from anemone.tree_manager.tree_expander import TreeExpansions
from tests.test_linoo_alternating import _signature
from tests.test_linoo_selector import _FakeOpeningInstructor, _node, _tree, _value


def _select(selector: Any, tree: Any) -> Any:
    """Use the real expansion API so selection retains the incremental heap."""
    selector.choose_node_and_branch_to_open(tree, TreeExpansions())
    assert selector.latest_selection_report is not None
    return selector.latest_selection_report


@pytest.mark.parametrize(
    "policy", ["inverse_depth", "opened_count_depth_index", "alternating_by_step"]
)
@pytest.mark.parametrize("value_kind", ["direct_value", "effective_value"])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("mapping_roundtrip", [False, True])
def test_checkpoint_after_cached_frontier_value_change(
    policy: str, value_kind: str, changed: bool, mapping_roundtrip: bool
) -> None:
    """Compare untouched, saved-live and restored selections after a rank reversal."""

    def fixture() -> tuple[Any, Any]:
        selector = Linoo(
            opening_instructor=_FakeOpeningInstructor(),
            random_generator=Random(4),
            depth_selection_policy=policy,
        )
        return selector, _tree(
            _node(0, 0, opened=True),
            _node(10, 1, score=3.0),
            _node(11, 1, score=2.0),
        )

    live, tree = fixture()
    untouched, control_tree = fixture()
    assert _signature(_select(live, tree)) == _signature(
        _select(untouched, control_tree)
    )
    # Both frontier priorities are cached. A backup/evaluation can change a
    # frontier value before the next selection refreshes its heap entry.
    if changed:
        for current_tree in (tree, control_tree):
            setattr(
                current_tree.descendants[1][11].tree_evaluation,
                value_kind,
                _value(4.0),
            )
    before_rng = live.random_generator.getstate()
    payload = live.build_checkpoint_payload(
        tree.root_node.tree_evaluation.required_objective
    )
    assert payload is not None
    cached = {
        candidate.node_id: candidate.priority
        for depth in payload.candidates_by_depth
        for candidate in depth.candidates
    }
    assert cached == {10: 3.0, 11: 2.0}
    assert live.random_generator.getstate() == before_rng
    if mapping_roundtrip:
        payload = _selector_payload_from_mapping(
            checkpoint_payload_to_jsonable(payload)
        )

    restored, _ = fixture()
    restored_tree = deepcopy(tree)
    restored.random_generator.setstate(before_rng)
    assert restored.restore_from_checkpoint_payload(
        tree=restored_tree,
        objective=restored_tree.root_node.tree_evaluation.required_objective,
        payload=payload,
    )
    assert restored.random_generator.getstate() == before_rng
    for step in range(4):
        expected = _select(untouched, control_tree)
        assert not expected.state_rebuilt
        if step == 0:
            assert expected.selected_node_id == (11 if changed else 10)
        assert _signature(_select(live, tree)) == _signature(expected)
        actual = _select(restored, restored_tree)
        assert not actual.state_rebuilt
        assert _signature(actual) == _signature(expected)
        assert (
            live.random_generator.getstate()
            == restored.random_generator.getstate()
            == untouched.random_generator.getstate()
        )
