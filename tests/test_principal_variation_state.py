"""Focused tests for the shared principal-variation state helper."""

# ruff: noqa: D103

from dataclasses import dataclass

from anemone.node_evaluation.common.principal_variation import (
    PrincipalVariationState,
)


@dataclass(frozen=True)
class _FakePvChild:
    pv_version: int
    best_branch_sequence: list[int]


def _child_pv_version(child: _FakePvChild) -> int:
    return child.pv_version


def _child_best_branch_sequence(child: _FakePvChild) -> list[int]:
    return child.best_branch_sequence.copy()


def test_setting_same_sequence_twice_bumps_version_once() -> None:
    pv_state = PrincipalVariationState()

    assert pv_state.set_sequence([1, 2], current_best_child_version=7)
    assert not pv_state.set_sequence([1, 2], current_best_child_version=7)
    assert pv_state.best_branch_sequence == [1, 2]
    assert pv_state.pv_version == 1
    assert pv_state.cached_best_child_version == 7


def test_clearing_an_already_empty_pv_is_a_noop() -> None:
    pv_state = PrincipalVariationState()

    assert not pv_state.clear()
    assert pv_state.best_branch_sequence == []
    assert pv_state.pv_version == 0
    assert pv_state.cached_best_child_version is None


def test_rebuilding_from_best_child_appends_child_sequence() -> None:
    pv_state = PrincipalVariationState()
    child = _FakePvChild(pv_version=3, best_branch_sequence=[4, 5])

    changed = pv_state.try_update_from_best_child(
        best_branch_key=1,
        best_child=child,
        branches_with_updated_best_branch_seq={1},
        child_pv_version_getter=_child_pv_version,
        child_best_branch_sequence_getter=_child_best_branch_sequence,
    )

    assert changed
    assert pv_state.best_branch_sequence == [1, 4, 5]
    assert pv_state.pv_version == 1
    assert pv_state.cached_best_child_version == 3


def test_try_update_from_best_child_clears_when_no_best_branch_remains() -> None:
    pv_state = PrincipalVariationState(
        best_branch_sequence=[1, 4, 5],
        pv_version=2,
        cached_best_child_version=3,
    )

    changed = pv_state.try_update_from_best_child(
        best_branch_key=None,
        best_child=None,
        branches_with_updated_best_branch_seq=set(),
        child_pv_version_getter=_child_pv_version,
        child_best_branch_sequence_getter=_child_best_branch_sequence,
    )

    assert changed
    assert pv_state.best_branch_sequence == []
    assert pv_state.pv_version == 3
    assert pv_state.cached_best_child_version is None


def test_cached_child_version_prevents_redundant_rebuild() -> None:
    pv_state = PrincipalVariationState()
    child = _FakePvChild(pv_version=9, best_branch_sequence=[4, 5])

    pv_state.set_sequence([1, 4, 5], current_best_child_version=9)

    changed = pv_state.try_update_from_best_child(
        best_branch_key=1,
        best_child=_FakePvChild(pv_version=9, best_branch_sequence=[7, 8]),
        branches_with_updated_best_branch_seq={1},
        child_pv_version_getter=_child_pv_version,
        child_best_branch_sequence_getter=_child_best_branch_sequence,
    )

    assert not changed
    assert pv_state.best_branch_sequence == [1, 4, 5]
    assert pv_state.pv_version == 1
    assert pv_state.cached_best_child_version == child.pv_version


def test_changed_child_version_rebuilds_existing_sequence() -> None:
    pv_state = PrincipalVariationState()
    pv_state.set_sequence([1, 4, 5], current_best_child_version=3)

    changed = pv_state.try_update_from_best_child(
        best_branch_key=1,
        best_child=_FakePvChild(pv_version=4, best_branch_sequence=[7, 8]),
        branches_with_updated_best_branch_seq={1},
        child_pv_version_getter=_child_pv_version,
        child_best_branch_sequence_getter=_child_best_branch_sequence,
    )

    assert changed
    assert pv_state.best_branch_sequence == [1, 7, 8]
    assert pv_state.pv_version == 2
    assert pv_state.cached_best_child_version == 4
