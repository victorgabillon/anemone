"""Bit-exact parity with the original Growth expression and terminal separation."""
# ruff: noqa: D103

import math
import struct
import sys

import pytest
from atomheart.games.morpion import MorpionDynamics, initial_state
from valanga.evaluations import Certainty, Value

from anemone.node_evaluation.direct import scale_nonterminal_score
from anemone.node_evaluation.direct.node_direct_evaluator import DISCOUNT
from tests.morpion_refresh_helpers import Evaluator, State, runtime


@pytest.mark.parametrize("depth", [0, 1, 62, 1_000_000, 1_000_000_000])
@pytest.mark.parametrize(
    "raw",
    [
        0.0,
        -0.0,
        42.0,
        -42.0,
        35.0,
        1.0,
        sys.float_info.max,
        -sys.float_info.max,
        float.fromhex("0x0.0000000000001p-1022"),
        math.inf,
        -math.inf,
        math.nan,
        struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0],
    ],
)
def test_exact_bits_match_original_expression(raw: float, depth: int) -> None:
    # Independent frozen policy, including literal constants and operation order.
    expected = (1 / 0.99999999) ** depth * raw
    assert struct.pack(">d", scale_nonterminal_score(raw, depth)) == struct.pack(
        ">d", expected
    )


def test_discount_constant_remains_available_at_the_existing_import_path() -> None:
    assert struct.pack(">d", DISCOUNT) == struct.pack(">d", 0.99999999)


def test_score_certainty_and_ties_are_not_reclassified() -> None:
    estimate = Value(score=42.0, certainty=Certainty.ESTIMATE)
    forced = Value(score=42.0, certainty=Certainty.FORCED)
    assert scale_nonterminal_score(estimate.score, 1) == scale_nonterminal_score(
        forced.score, 1
    )
    assert (
        estimate.certainty == Certainty.ESTIMATE
        and forced.certainty == Certainty.FORCED
    )
    assert math.copysign(1.0, scale_nonterminal_score(-0.0, 1)) == -1.0
    assert scale_nonterminal_score(100.0, 1) == 100.00000100000001


def test_original_power_overflow_behavior_is_preserved() -> None:
    with pytest.raises(OverflowError):
        (1 / 0.99999999) ** 100_000_000_000 * 42.0
    with pytest.raises(OverflowError):
        scale_nonterminal_score(42.0, 100_000_000_000)


def test_real_move_34_root_uses_depth_zero_and_terminal_35_is_not_scaled() -> None:
    dynamics = MorpionDynamics()
    state = initial_state()
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        assert actions
        transition = dynamics.step(state, actions[0])
        if transition.is_over:
            break
        state = transition.next_state
    else:
        pytest.fail("deterministic endgame did not terminate")
    assert state.moves == 34 and transition.next_state.moves == 35 and len(actions) == 1
    search = runtime(state=State(state), branch_limit=128, steps=1)
    root = search.tree.root_node
    terminal = next(root.iter_child_nodes())
    original_terminal = terminal.tree_evaluation.direct_value
    assert root.state.board.moves == 34 and root.tree_depth == 0
    assert terminal.tree_depth == 1
    search.refresh_with_evaluator(
        Evaluator({0: 42.0, terminal.id: 1000.0}), scope="all", batch_size=1
    )
    assert (
        root.tree_evaluation.direct_value.score
        == scale_nonterminal_score(42.0, 0)
        == 42.0
    )
    assert root.tree_evaluation.direct_value.score != scale_nonterminal_score(42.0, 34)
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.tree_value.score == 35.0
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert terminal.tree_evaluation.direct_value is original_terminal
    assert original_terminal.certainty == Certainty.TERMINAL
    assert (
        original_terminal.score == 35.0
        and original_terminal.score != scale_nonterminal_score(35.0, 1)
    )
