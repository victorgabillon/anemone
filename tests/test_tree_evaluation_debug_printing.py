"""Tests for generic tree-evaluation branch-order debug printing."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluation
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.utils.logger import anemone_logger


class _SoloRole(Enum):
    SOLO = "solo"


class _FakeDynamics:
    def action_name(self, state: Any, action: int) -> str:
        del state
        return f"move:{action}"


def _call_generic_debug_print(node_eval: NodeTreeEvaluation[Any]) -> None:
    node_eval.print_branch_ordering(dynamics=_FakeDynamics())


def _max_child(node_id: int, score: float) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    value = Value(score=score, certainty=Certainty.ESTIMATE)
    evaluation.direct_value = value
    evaluation.backed_up_value = value
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=evaluation)


def _minimax_child(node_id: int, score: float) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    value = Value(score=score, certainty=Certainty.ESTIMATE)
    evaluation.direct_value = value
    evaluation.backed_up_value = value
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=evaluation)


def _make_single_agent_parent() -> NodeMaxEvaluation[Any]:
    children = {
        0: _max_child(20, 0.5),
        1: _max_child(10, 0.5),
    }
    tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.update_branches_values({0, 1})
    return evaluation


def _make_minimax_parent() -> NodeMinmaxEvaluation[Any, Any]:
    children = {
        0: _minimax_child(10, 0.2),
        1: _minimax_child(11, 0.7),
    }
    tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    evaluation.update_branches_values({0, 1})
    return evaluation


@pytest.mark.parametrize(
    ("parent", "expected_order", "expected_primary_scores"),
    [
        pytest.param(_make_single_agent_parent(), ["move:1", "move:0"], ["0.5"]),
        pytest.param(_make_minimax_parent(), ["move:1", "move:0"], ["0.7", "0.2"]),
    ],
)
def test_print_branch_ordering_is_available_via_shared_protocol(
    monkeypatch: pytest.MonkeyPatch,
    parent: NodeTreeEvaluation[Any],
    expected_order: list[str],
    expected_primary_scores: list[str],
) -> None:
    messages: list[str] = []

    def _record_info(msg: object, *args: object) -> None:
        rendered = str(msg) % args if args else str(msg)
        messages.append(rendered)

    monkeypatch.setattr(anemone_logger, "info", _record_info)

    _call_generic_debug_print(parent)

    assert messages[0] == "here are the 2 branches in decision order:"
    assert expected_order[0] in messages[1]
    assert expected_order[1] in messages[1]
    assert messages[1].index(expected_order[0]) < messages[1].index(expected_order[1])
    for score in expected_primary_scores:
        assert score in messages[1]
