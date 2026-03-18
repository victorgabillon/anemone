"""Focused tests for shared backup-policy helpers."""

# ruff: noqa: D103

from valanga.evaluations import Certainty, Value

from anemone.backup_policies.common import (
    SelectedValue,
    select_value_from_best_child_and_direct,
)


def test_select_value_uses_direct_when_no_best_child_exists() -> None:
    direct = Value(score=0.3, certainty=Certainty.ESTIMATE)

    selection = select_value_from_best_child_and_direct(
        best_child_value=None,
        direct_value=direct,
        all_branches_generated=False,
        child_beats_direct=lambda _child, _direct: (_ for _ in ()).throw(
            AssertionError("comparison should not run without a best child")
        ),
    )

    assert selection == SelectedValue(value=direct, from_child=False)


def test_select_value_uses_child_when_no_direct_value_exists() -> None:
    child = Value(score=0.8, certainty=Certainty.ESTIMATE)

    selection = select_value_from_best_child_and_direct(
        best_child_value=child,
        direct_value=None,
        all_branches_generated=False,
        child_beats_direct=lambda _child, _direct: (_ for _ in ()).throw(
            AssertionError("comparison should not run without a direct value")
        ),
    )

    assert selection == SelectedValue(value=child, from_child=True)


def test_select_value_full_expansion_prefers_child_without_compare() -> None:
    child = Value(score=0.1, certainty=Certainty.ESTIMATE)
    direct = Value(score=0.9, certainty=Certainty.ESTIMATE)

    selection = select_value_from_best_child_and_direct(
        best_child_value=child,
        direct_value=direct,
        all_branches_generated=True,
        child_beats_direct=lambda _child, _direct: (_ for _ in ()).throw(
            AssertionError("comparison should not run for fully expanded nodes")
        ),
    )

    assert selection == SelectedValue(value=child, from_child=True)


def test_select_value_partial_expansion_uses_family_compare() -> None:
    child = Value(score=0.7, certainty=Certainty.ESTIMATE)
    direct = Value(score=0.5, certainty=Certainty.ESTIMATE)

    winning_selection = select_value_from_best_child_and_direct(
        best_child_value=child,
        direct_value=direct,
        all_branches_generated=False,
        child_beats_direct=lambda _child, _direct: True,
    )
    losing_selection = select_value_from_best_child_and_direct(
        best_child_value=child,
        direct_value=direct,
        all_branches_generated=False,
        child_beats_direct=lambda _child, _direct: False,
    )

    assert winning_selection == SelectedValue(value=child, from_child=True)
    assert losing_selection == SelectedValue(value=direct, from_child=False)
