"""Regression checks for shared tree-evaluation test doubles."""

from valanga import Color
from valanga.evaluations import Certainty, Value

from tests.fakes_tree_evaluation import (
    FakeChildEvaluation,
    FakeChildNode,
    FakeOverEvent,
)


def test_get_value_candidate_and_getters_return_value() -> None:
    fake = FakeChildEvaluation(value_white=0.42)

    candidate = fake.get_value_candidate()

    assert candidate is not None
    assert fake.get_value() is candidate
    assert fake.get_score() == 0.42


def test_fake_child_evaluation_exposes_exactness_and_terminality_separately() -> None:
    terminal = FakeChildEvaluation(
        value_white=0.0,
        minmax_value=Value(
            score=0.0,
            certainty=Certainty.TERMINAL,
            over_event=FakeOverEvent(_is_over=True, winner=Color.WHITE),
        ),
    )
    forced_without_over = FakeChildEvaluation(
        value_white=0.0,
        minmax_value=Value(score=0.0, certainty=Certainty.FORCED, over_event=None),
    )
    estimate_without_over = FakeChildEvaluation(
        value_white=0.0,
        minmax_value=Value(score=0.0, certainty=Certainty.ESTIMATE, over_event=None),
    )

    assert terminal.has_exact_value()
    assert terminal.is_terminal()
    assert forced_without_over.has_exact_value()
    assert not forced_without_over.is_terminal()
    assert not estimate_without_over.has_over_event()
    assert not estimate_without_over.has_exact_value()


def test_fake_child_node_is_over_uses_true_terminality() -> None:
    eval_with_raw_over_only = FakeChildEvaluation(
        value_white=0.0,
        over_event=FakeOverEvent(_is_over=True, winner=Color.WHITE),
        minmax_value=Value(score=0.0, certainty=Certainty.ESTIMATE, over_event=None),
    )
    terminal_eval = FakeChildEvaluation(
        value_white=0.0,
        minmax_value=Value(
            score=0.0,
            certainty=Certainty.TERMINAL,
            over_event=FakeOverEvent(_is_over=True, winner=Color.WHITE),
        ),
    )

    assert not FakeChildNode(
        node_id=1, tree_evaluation=eval_with_raw_over_only
    ).is_over()
    assert FakeChildNode(node_id=2, tree_evaluation=terminal_eval).is_over()
