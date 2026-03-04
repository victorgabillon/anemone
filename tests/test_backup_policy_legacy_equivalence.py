# ruff: noqa: D103
"""Value-equivalence tests and explicit PV invariants for minimax backup policies."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from valanga import Color

from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.backup_policies.legacy_minimax import LegacyMinimaxBackupPolicy
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)


@dataclass
class _FakeChildNode:
    node_id: int
    tree_evaluation: NodeMinmaxEvaluation[Any, Any]

    @property
    def tree_node(self) -> Any:
        return self.tree_evaluation.tree_node


def _make_leaf_eval(
    *,
    turn: Color,
    value_white: float,
    pv_tail: list[int],
    node_id: int,
) -> NodeMinmaxEvaluation[Any, Any]:
    leaf_tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=turn),
        branches_children={},
        all_branches_generated=True,
    )
    ev = NodeMinmaxEvaluation(tree_node=leaf_tree_node, backup_policy=None)
    ev.set_evaluation(value_white)
    ev.set_best_branch_sequence(pv_tail[:])
    return ev


def _build_parent_eval(
    *,
    turn: Color,
    children: dict[int, _FakeChildNode],
    all_generated: bool,
    parent_eval_value: float,
    policy: Any,
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=all_generated,
    )
    ev = NodeMinmaxEvaluation(tree_node=parent_tree_node, backup_policy=policy)
    ev.set_evaluation(parent_eval_value)

    # Minimal init: populate branch ordering so `best_branch()` is defined.
    all_branches = set(children.keys())
    if all_branches:
        ev.update_branches_values(branches_to_consider=all_branches)

    return ev


def _run_backup(
    ev: NodeMinmaxEvaluation[Any, Any],
    *,
    updated_values: set[int],
    updated_best_seq: set[int],
) -> tuple[float | None, list[int], Any]:
    result = ev.backup_from_children(
        branches_with_updated_value=updated_values,
        branches_with_updated_best_branch_seq=updated_best_seq,
    )
    return ev.value_white_minmax, ev.best_branch_sequence.copy(), result


def _run_backup_or_exc(
    ev: NodeMinmaxEvaluation[Any, Any],
    *,
    updated_values: set[int],
    updated_best_seq: set[int],
) -> tuple[bool, type[Exception] | tuple[float | None, list[int], Any]]:
    try:
        return False, _run_backup(
            ev,
            updated_values=updated_values,
            updated_best_seq=updated_best_seq,
        )
    except Exception as exc:
        return True, type(exc)


def _assert_explicit_pv_invariants(explicit: NodeMinmaxEvaluation[Any, Any]) -> None:
    best_branch = explicit.best_branch()
    if best_branch is None:
        assert explicit.best_branch_sequence == []
        return

    if explicit.tree_node.all_branches_generated:
        assert explicit.best_branch_sequence
        assert explicit.best_branch_sequence[0] == best_branch
        return

    direct_value = explicit.direct_value
    if direct_value is None:
        assert explicit.best_branch_sequence == []
        return

    child_value = explicit._child_value_candidate(best_branch)
    if child_value is None:
        assert explicit.best_branch_sequence == []
        return
    child_dominates = (
        explicit.evaluation_ordering.semantic_compare(
            child_value,
            direct_value,
            side_to_move=explicit.tree_node.state.turn,
        )
        >= 0
    )

    if child_dominates:
        assert explicit.best_branch_sequence
        assert explicit.best_branch_sequence[0] == best_branch
    else:
        assert explicit.best_branch_sequence == []


def _assert_value_equivalent(
    legacy: NodeMinmaxEvaluation[Any, Any],
    explicit: NodeMinmaxEvaluation[Any, Any],
    *,
    updated_values: set[int],
    updated_best_seq: set[int],
) -> None:
    legacy_is_exc, legacy_outcome = _run_backup_or_exc(
        legacy,
        updated_values=updated_values,
        updated_best_seq=updated_best_seq,
    )
    explicit_is_exc, explicit_outcome = _run_backup_or_exc(
        explicit,
        updated_values=updated_values,
        updated_best_seq=updated_best_seq,
    )

    assert legacy_is_exc == explicit_is_exc

    if legacy_is_exc:
        assert legacy_outcome == explicit_outcome
        return

    legacy_value, _, legacy_result = legacy_outcome
    explicit_value, _, explicit_result = explicit_outcome
    assert legacy_value == explicit_value
    assert legacy_result.over_changed == explicit_result.over_changed
    _assert_explicit_pv_invariants(explicit)


@pytest.mark.parametrize(
    "turn, all_generated, parent_eval, child0, child1, expected_value, expected_best_branch",
    [
        (Color.WHITE, True, 0.0, 0.1, 0.9, 0.9, 1),
        (Color.BLACK, True, 0.0, 0.1, 0.9, 0.1, 0),
    ],
)
def test_equivalence_full_generation_value_and_pv_head(
    turn: Color,
    all_generated: bool,
    parent_eval: float,
    child0: float,
    child1: float,
    expected_value: float,
    expected_best_branch: int,
) -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(
                turn=Color.WHITE, value_white=child0, pv_tail=[7], node_id=10
            ),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(
                turn=Color.WHITE, value_white=child1, pv_tail=[8], node_id=11
            ),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(
                turn=Color.WHITE, value_white=child0, pv_tail=[7], node_id=10
            ),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(
                turn=Color.WHITE, value_white=child1, pv_tail=[8], node_id=11
            ),
        ),
    }

    legacy = _build_parent_eval(
        turn=turn,
        children=children_legacy,
        all_generated=all_generated,
        parent_eval_value=parent_eval,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=turn,
        children=children_explicit,
        all_generated=all_generated,
        parent_eval_value=parent_eval,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.value_white_minmax == expected_value
    assert explicit.value_white_minmax == expected_value
    assert explicit.best_branch_sequence[:1] == [expected_best_branch]


def test_equivalence_partial_expansion_direct_dominates_clears_pv() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.4, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.3, pv_tail=[8], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.4, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.3, pv_tail=[8], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=False,
        parent_eval_value=0.5,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=False,
        parent_eval_value=0.5,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.value_white_minmax == 0.5
    assert explicit.best_branch_sequence == []


def test_equivalence_partial_expansion_direct_dominates_clears_pv_black() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.6, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.7, pv_tail=[8], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.6, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.7, pv_tail=[8], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.BLACK,
        children=children_legacy,
        all_generated=False,
        parent_eval_value=0.5,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.BLACK,
        children=children_explicit,
        all_generated=False,
        parent_eval_value=0.5,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.value_white_minmax == 0.5
    assert explicit.best_branch_sequence == []


def test_equivalence_partial_expansion_child_dominates_sets_pv() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.8, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.8, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=False,
        parent_eval_value=0.1,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=False,
        parent_eval_value=0.1,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.value_white_minmax == 0.8
    assert explicit.best_branch_sequence[:1] == [0]


def test_equivalence_partial_expansion_child_dominates_sets_pv_black() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.1, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.1, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.BLACK,
        children=children_legacy,
        all_generated=False,
        parent_eval_value=0.5,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.BLACK,
        children=children_explicit,
        all_generated=False,
        parent_eval_value=0.5,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.value_white_minmax == 0.1
    assert explicit.best_branch_sequence[:1] == [0]


def test_equivalence_pv_only_propagation_when_child_best_line_changes() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )
    assert explicit.best_branch_sequence[:1] == [0]

    children_legacy[0].tree_evaluation.set_best_branch_sequence([99])
    children_explicit[0].tree_evaluation.set_best_branch_sequence([99])

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values=set(),
        updated_best_seq={0},
    )

    assert explicit.best_branch_sequence == [0, 99]


def test_equivalence_pv_only_propagation_when_child_best_line_changes_black() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.1, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.1, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.BLACK,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.BLACK,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )
    assert explicit.best_branch_sequence[:1] == [0]

    children_legacy[0].tree_evaluation.set_best_branch_sequence([99])
    children_explicit[0].tree_evaluation.set_best_branch_sequence([99])

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values=set(),
        updated_best_seq={0},
    )

    assert explicit.best_branch_sequence == [0, 99]


def test_equivalence_pv_update_on_non_best_branch_is_noop() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    legacy_before = legacy.best_branch_sequence.copy()
    explicit_before = explicit.best_branch_sequence.copy()

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values=set(),
        updated_best_seq={1},
    )

    assert legacy.best_branch_sequence == legacy_before
    assert explicit.best_branch_sequence == explicit_before


def test_equivalence_equal_child_values_is_stable() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.5, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.5, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.5, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.5, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    legacy_first = legacy.best_branch_sequence.copy()
    explicit_first = explicit.best_branch_sequence.copy()

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    assert legacy.best_branch_sequence == legacy_first
    assert explicit.best_branch_sequence == explicit_first


def test_equivalence_value_and_best_pv_change_in_same_backup_call() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )

    children_legacy[1].tree_evaluation.set_evaluation(0.95)
    children_explicit[1].tree_evaluation.set_evaluation(0.95)
    children_legacy[1].tree_evaluation.set_best_branch_sequence([99])
    children_explicit[1].tree_evaluation.set_best_branch_sequence([99])

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq={1},
    )

    assert legacy.value_white_minmax == 0.95
    assert explicit.best_branch_sequence == [1, 99]


def test_equivalence_empty_updates_after_baseline_is_noop() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.9, pv_tail=[4], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[5], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy, explicit, updated_values={0, 1}, updated_best_seq=set()
    )

    legacy_before = (legacy.value_white_minmax, legacy.best_branch_sequence.copy())
    explicit_before = (
        explicit.value_white_minmax,
        explicit.best_branch_sequence.copy(),
    )

    _assert_value_equivalent(
        legacy, explicit, updated_values=set(), updated_best_seq=set()
    )

    assert (legacy.value_white_minmax, legacy.best_branch_sequence) == legacy_before
    assert (
        explicit.value_white_minmax,
        explicit.best_branch_sequence,
    ) == explicit_before


def test_equivalence_partial_expansion_without_float_direct_eval() -> None:
    children_legacy = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.8, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }
    children_explicit = {
        0: _FakeChildNode(
            10,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.8, pv_tail=[7], node_id=10),
        ),
        1: _FakeChildNode(
            11,
            _make_leaf_eval(turn=Color.WHITE, value_white=0.2, pv_tail=[8], node_id=11),
        ),
    }

    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children=children_legacy,
        all_generated=False,
        parent_eval_value=0.1,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children=children_explicit,
        all_generated=False,
        parent_eval_value=0.1,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    # Step-7 explicit is Value-first: float bridge may be absent while canonical Value remains.
    explicit.value_white_direct_evaluation = None
    # Explicit policy is Value-first in Step 7; clear Value fields to model "no direct evaluation".
    explicit.direct_value = None
    explicit.minmax_value = None

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values={0, 1},
        updated_best_seq=set(),
    )


def test_equivalence_no_children_raises_assertion_full_generated() -> None:
    legacy = _build_parent_eval(
        turn=Color.WHITE,
        children={},
        all_generated=True,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.WHITE,
        children={},
        all_generated=True,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values=set(),
        updated_best_seq=set(),
    )


def test_equivalence_no_children_raises_assertion_partial_generated() -> None:
    legacy = _build_parent_eval(
        turn=Color.BLACK,
        children={},
        all_generated=False,
        parent_eval_value=0.0,
        policy=LegacyMinimaxBackupPolicy(),
    )
    explicit = _build_parent_eval(
        turn=Color.BLACK,
        children={},
        all_generated=False,
        parent_eval_value=0.0,
        policy=ExplicitMinimaxBackupPolicy(),
    )

    _assert_value_equivalent(
        legacy,
        explicit,
        updated_values=set(),
        updated_best_seq=set(),
    )
