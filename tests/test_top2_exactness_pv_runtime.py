"""Focused tests for the conservative top-2 / exactness / PV runtime."""

# ruff: noqa: D103

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from valanga import BranchKey, Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.common import FieldChange, NodeDelta
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)

if TYPE_CHECKING:
    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.tree.top2_exactness_pv_runtime import (
        Top2ExactnessPvRuntime,
    )


def _branch(branch_key: int) -> BranchKey:
    return cast("BranchKey", branch_key)


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: Color) -> bool:
        return self.winner == role

    def is_loss_for(self, role: Color) -> bool:
        return self.winner is not None and self.winner != role


@dataclass
class _FakeChildNode:
    node_id: int
    tree_evaluation: NodeMinmaxEvaluation[Any, Any]

    @property
    def id(self) -> int:
        return self.node_id

    @property
    def tree_node(self) -> Any:
        return self.tree_evaluation.tree_node


@dataclass(slots=True)
class _CountingMinimaxPolicy:
    wrapped: ExplicitMinimaxBackupPolicy = field(
        default_factory=ExplicitMinimaxBackupPolicy
    )
    calls: int = 0

    @property
    def aggregation_policy(self) -> Any:
        return self.wrapped.aggregation_policy

    @property
    def proof_policy(self) -> Any:
        return self.wrapped.proof_policy

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult[BranchKey]:
        self.calls += 1
        return self.wrapped.backup_from_children(
            node_eval=node_eval,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )


def _make_leaf_eval(
    *,
    node_id: int,
    value: Value,
    pv_tail: list[int],
) -> NodeMinmaxEvaluation[Any, Any]:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=cast("Any", tree_node))
    evaluation.direct_value = value
    evaluation.minmax_value = value
    evaluation.set_best_branch_sequence(list(pv_tail))
    return evaluation


def _make_child(
    *,
    node_id: int,
    value: Value,
    pv_tail: list[int],
) -> _FakeChildNode:
    return _FakeChildNode(
        node_id=node_id,
        tree_evaluation=_make_leaf_eval(
            node_id=node_id,
            value=value,
            pv_tail=pv_tail,
        ),
    )


def _make_parent_eval(
    *,
    children: dict[BranchKey, _FakeChildNode] | None = None,
) -> tuple[
    NodeMinmaxEvaluation[Any, Any],
    dict[BranchKey, _FakeChildNode],
    _CountingMinimaxPolicy,
]:
    child_map = children or {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.ESTIMATE),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.2, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
    }
    tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children=child_map,
        all_branches_generated=True,
    )
    backup_policy = _CountingMinimaxPolicy()
    parent = NodeMinmaxEvaluation(
        tree_node=cast("Any", tree_node),
        backup_policy=backup_policy,
    )
    parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    parent.minmax_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    parent.update_branches_values(branches_to_consider=set(child_map))
    return parent, child_map, backup_policy


def _initialize_runtime(
    parent: NodeMinmaxEvaluation[Any, Any],
    policy: _CountingMinimaxPolicy,
) -> None:
    first = parent.backup_from_children(
        branches_with_updated_value=set(parent.tree_node.branches_children),
        branches_with_updated_best_branch_seq=set(parent.tree_node.branches_children),
    )
    assert first.value_changed
    assert policy.calls == 1
    policy.calls = 0


def _set_child_value(child: _FakeChildNode, value: Value) -> None:
    child.tree_evaluation.direct_value = value
    child.tree_evaluation.minmax_value = value


def _runtime(
    parent: NodeMinmaxEvaluation[Any, Any],
) -> Top2ExactnessPvRuntime[BranchKey]:
    return cast(
        "Top2ExactnessPvRuntime[BranchKey]",
        parent.backup_runtime,
    )


def test_runtime_noops_when_child_delta_is_empty() -> None:
    parent, _, policy = _make_parent_eval()
    _initialize_runtime(parent, policy)

    value_before = parent.minmax_value
    pv_before = parent.best_branch_sequence.copy()

    result = parent.backup_from_children(
        branches_with_updated_value={1},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={_branch(1): NodeDelta()},
    )

    assert policy.calls == 0
    assert parent.minmax_value == value_before
    assert parent.best_branch_sequence == pv_before
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()


def test_runtime_ignores_non_best_child_pv_churn() -> None:
    parent, children, policy = _make_parent_eval()
    _initialize_runtime(parent, policy)

    parent_version_before = parent.pv_version
    pv_before = parent.best_branch_sequence.copy()
    child_version_before = children[_branch(1)].tree_evaluation.pv_version
    children[_branch(1)].tree_evaluation.set_best_branch_sequence([77])

    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={_branch(1)},
        child_deltas={
            _branch(1): NodeDelta(
                pv_version=FieldChange(
                    old=child_version_before,
                    new=children[_branch(1)].tree_evaluation.pv_version,
                )
            )
        },
    )

    assert policy.calls == 0
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()
    assert parent.best_branch_sequence == pv_before
    assert parent.pv_version == parent_version_before


def test_runtime_handles_non_best_child_worsening_without_fallback() -> None:
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.ESTIMATE),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.6, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
        _branch(2): _make_child(
            node_id=12,
            value=Value(score=0.1, certainty=Certainty.ESTIMATE),
            pv_tail=[6],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    old_child_value = built_children[_branch(2)].tree_evaluation.minmax_value
    new_child_value = Value(score=-0.2, certainty=Certainty.ESTIMATE)
    _set_child_value(built_children[_branch(2)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(2)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(2): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert parent.best_branch() == 0
    assert parent.second_best_branch() == 1
    assert parent.minmax_value == Value(score=0.9, certainty=Certainty.ESTIMATE)
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 1
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()


def test_runtime_handles_non_best_child_improving_below_second_without_fallback() -> (
    None
):
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.ESTIMATE),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.6, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
        _branch(2): _make_child(
            node_id=12,
            value=Value(score=0.1, certainty=Certainty.ESTIMATE),
            pv_tail=[6],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    old_child_value = built_children[_branch(2)].tree_evaluation.minmax_value
    new_child_value = Value(score=0.5, certainty=Certainty.ESTIMATE)
    _set_child_value(built_children[_branch(2)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(2)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(2): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert parent.best_branch() == 0
    assert parent.second_best_branch() == 1
    assert parent.minmax_value == Value(score=0.9, certainty=Certainty.ESTIMATE)
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 1
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()


def test_runtime_updates_second_best_when_third_child_overtakes_second() -> None:
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.ESTIMATE),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.6, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
        _branch(2): _make_child(
            node_id=12,
            value=Value(score=0.1, certainty=Certainty.ESTIMATE),
            pv_tail=[6],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    old_child_value = built_children[_branch(2)].tree_evaluation.minmax_value
    new_child_value = Value(score=0.7, certainty=Certainty.ESTIMATE)
    _set_child_value(built_children[_branch(2)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(2)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(2): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert parent.best_branch() == 0
    assert parent.second_best_branch() == 2
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 2
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()


def test_runtime_updates_best_value_when_selected_child_remains_best() -> None:
    parent, children, policy = _make_parent_eval()
    _initialize_runtime(parent, policy)

    old_child_value = children[_branch(0)].tree_evaluation.minmax_value
    new_child_value = Value(score=1.1, certainty=Certainty.ESTIMATE)
    _set_child_value(children[_branch(0)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(0)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(0): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert parent.best_branch() == 0
    assert parent.best_branch_sequence == [0, 4]
    assert parent.minmax_value == new_child_value
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 1
    assert result.node_delta.value == FieldChange(
        old=Value(score=0.9, certainty=Certainty.ESTIMATE),
        new=new_child_value,
    )
    assert result.node_delta.best_branch is None
    assert result.node_delta.pv_version is None


def test_runtime_updates_parent_selection_when_challenger_overtakes_best() -> None:
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.ESTIMATE),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.6, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
        _branch(2): _make_child(
            node_id=12,
            value=Value(score=0.1, certainty=Certainty.ESTIMATE),
            pv_tail=[6],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    parent_version_before = parent.pv_version
    old_child_value = built_children[_branch(2)].tree_evaluation.minmax_value
    new_child_value = Value(score=1.2, certainty=Certainty.ESTIMATE)
    _set_child_value(built_children[_branch(2)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(2)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(2): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert result.value_changed
    assert result.pv_changed
    assert not result.over_changed
    assert parent.best_branch() == 2
    assert parent.best_branch_sequence == [2, 6]
    assert parent.minmax_value == new_child_value
    assert runtime.best_branch == 2
    assert runtime.second_best_branch == 0
    assert result.node_delta.value == FieldChange(
        old=Value(score=0.9, certainty=Certainty.ESTIMATE),
        new=new_child_value,
    )
    assert result.node_delta.pv_version == FieldChange(
        old=parent_version_before,
        new=parent_version_before + 1,
    )


def test_runtime_updates_parent_exactness_from_cached_exact_child_count() -> None:
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(score=0.9, certainty=Certainty.FORCED),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=-0.2, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    old_child_value = built_children[_branch(1)].tree_evaluation.minmax_value
    new_child_value = Value(score=-0.2, certainty=Certainty.FORCED)
    _set_child_value(built_children[_branch(1)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(1)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(1): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 0
    assert parent.best_branch() == 0
    assert parent.best_branch_sequence == [0, 4]
    assert parent.minmax_value == Value(score=0.9, certainty=Certainty.FORCED)
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 1
    assert runtime.exact_child_count == 2
    assert result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.value == FieldChange(
        old=Value(score=0.9, certainty=Certainty.ESTIMATE),
        new=Value(score=0.9, certainty=Certainty.FORCED),
    )
    assert result.node_delta.best_branch is None
    assert result.node_delta.pv_version is None


def test_runtime_updates_parent_pv_from_selected_child_delta() -> None:
    parent, children, policy = _make_parent_eval()
    _initialize_runtime(parent, policy)

    parent_version_before = parent.pv_version
    child_version_before = children[_branch(0)].tree_evaluation.pv_version
    children[_branch(0)].tree_evaluation.set_best_branch_sequence([99, 100])

    result = parent.backup_from_children(
        branches_with_updated_value=set(),
        branches_with_updated_best_branch_seq={_branch(0)},
        child_deltas={
            _branch(0): NodeDelta(
                pv_version=FieldChange(
                    old=child_version_before,
                    new=children[_branch(0)].tree_evaluation.pv_version,
                )
            )
        },
    )

    assert policy.calls == 0
    assert not result.value_changed
    assert result.pv_changed
    assert not result.over_changed
    assert parent.best_branch() == 0
    assert parent.best_branch_sequence == [0, 99, 100]
    assert result.node_delta.value is None
    assert result.node_delta.best_branch is None
    assert result.node_delta.all_branches_generated is None
    assert result.node_delta.pv_version == FieldChange(
        old=parent_version_before,
        new=parent_version_before + 1,
    )


def test_runtime_refreshes_top2_and_exactness_cache_after_fallback() -> None:
    children = {
        _branch(0): _make_child(
            node_id=10,
            value=Value(
                score=0.9,
                certainty=Certainty.FORCED,
                over_event=_FakeOverEvent(winner=Color.WHITE),
            ),
            pv_tail=[4],
        ),
        _branch(1): _make_child(
            node_id=11,
            value=Value(score=0.6, certainty=Certainty.ESTIMATE),
            pv_tail=[5],
        ),
        _branch(2): _make_child(
            node_id=12,
            value=Value(score=0.55, certainty=Certainty.ESTIMATE),
            pv_tail=[6],
        ),
    }
    parent, built_children, policy = _make_parent_eval(children=children)
    _initialize_runtime(parent, policy)

    old_child_value = built_children[_branch(1)].tree_evaluation.minmax_value
    new_child_value = Value(score=0.1, certainty=Certainty.ESTIMATE)
    _set_child_value(built_children[_branch(1)], new_child_value)

    result = parent.backup_from_children(
        branches_with_updated_value={_branch(1)},
        branches_with_updated_best_branch_seq=set(),
        child_deltas={
            _branch(1): NodeDelta(
                value=FieldChange(old=old_child_value, new=new_child_value)
            )
        },
    )

    runtime = _runtime(parent)
    assert policy.calls == 1
    assert not result.value_changed
    assert not result.pv_changed
    assert not result.over_changed
    assert result.node_delta.is_empty()
    assert parent.best_branch() == 0
    assert parent.second_best_branch() == 2
    assert runtime.best_branch == 0
    assert runtime.second_best_branch == 2
    assert runtime.exact_child_count == 1
    assert runtime.selected_child_pv_version == parent.pv_cached_best_child_version
