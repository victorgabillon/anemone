"""Regressions for effective-value changes when Growth completes an opening."""

from typing import Any

import pytest
from valanga.evaluations import Certainty

from anemone import NodeValueUpdate
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from tests.scientific_refresh_helpers import (
    Evaluator,
    nodes,
    reference,
    runtime,
    snapshot,
)


def _open(search: Any, tag: int, branch: int) -> None:
    """Run the ordinary structural/evaluation/backup phases for one edge."""
    node = next(n for n in nodes(search) if n.tag == tag)
    instructions = OpeningInstructions(
        {(node.id, branch): OpeningInstruction(node, branch)}
    )
    manager = search.tree_manager
    expansions = manager.expand_instructions(search.tree, instructions)
    manager.evaluate_expansions(expansions)
    manager.update_backward(expansions)


def _partial_dag(parent_score: float) -> tuple[Any, dict[int, float]]:
    """Make two incoming parents and an unchanged best subtree below a mask."""
    children = {
        0: [1, 2],
        1: [3, 7],
        2: [3, 8],
        3: [4, 5],
        4: [6],
        5: [6],
        6: [],
        7: [],
        8: [],
    }
    values = {
        0: -1.0,
        1: parent_score,
        2: parent_score,
        3: 10.0,
        4: 4.0,
        5: 2.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
    }
    search = runtime(children, values)
    for tag, branch in ((0, 0), (0, 1), (1, 0), (2, 0), (3, 0)):
        _open(search, tag, branch)
    shared = next(n for n in nodes(search) if n.tag == 3)
    assert len(shared.parent_nodes) == 2
    assert snapshot(search) == reference(search)
    return search, values


@pytest.mark.parametrize("parent_score", [0.0, 20.0])
@pytest.mark.parametrize("refresh_first", [False, True])
def test_growth_opening_completion_propagates_without_tree_or_pv_change(
    parent_score: float, refresh_first: bool
) -> None:
    """Both visible and masked incoming caches must match native full backup."""
    search, values = _partial_dag(parent_score)
    if refresh_first:
        search.refresh_with_evaluator(Evaluator(values), scope="all")
        assert snapshot(search) == reference(search)
    shared = next(n for n in nodes(search) if n.tag == 3)
    before = snapshot(search)[shared.id]
    _open(search, 3, 1)
    after = snapshot(search)[shared.id]
    assert shared.all_branches_generated
    assert before["tree"] == after["tree"]
    assert before["pv"] == after["pv"]
    assert before["effective"] != after["effective"]
    assert after["tree"].certainty == Certainty.ESTIMATE
    assert snapshot(search) == reference(search)


@pytest.mark.parametrize("parent_score", [0.0, 20.0])
def test_external_direct_update_propagates_without_tree_or_pv_change(
    parent_score: float,
) -> None:
    """An external direct update must settle both incoming DAG paths."""
    search, _ = _partial_dag(parent_score)
    shared = next(n for n in nodes(search) if n.tag == 3)
    before = snapshot(search)[shared.id]
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(shared.id), direct_value=3.0)]
    )
    after = snapshot(search)[shared.id]
    assert before["tree"] == after["tree"]
    assert before["pv"] == after["pv"]
    assert before["effective"] != after["effective"]
    assert snapshot(search) == reference(search)
