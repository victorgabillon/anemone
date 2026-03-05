"""Tests proving backup policy injection is used by the updater path."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from anemone.backup_policies.types import BackupResult
from anemone.updates.minmax_evaluation_updater import MinMaxEvaluationUpdater
from anemone.updates.updates_file import UpdateInstructionsTowardsOneParentNode
from anemone.updates.value_block import ValueUpdateInstructionsTowardsOneParentNode

if TYPE_CHECKING:
    from anemone.nodes.algorithm_node import AlgorithmNode


class _FakeOverEvent:
    def is_over(self) -> bool:
        return False


@dataclass(slots=True)
class _FakePolicy:
    calls: list[tuple[set[int], set[int]]]
    result: BackupResult

    def backup_from_children(
        self,
        node_eval: object,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult:
        del node_eval
        self.calls.append(
            (branches_with_updated_value, branches_with_updated_best_branch_seq)
        )
        return self.result


@dataclass(slots=True)
class _FakeTreeEvaluation:
    backup_policy: _FakePolicy
    over_event: _FakeOverEvent = field(default_factory=_FakeOverEvent)
    terminal_candidate: bool = False

    def backup_from_children(
        self,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult:
        return self.backup_policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

    def update_over(self, branches_with_updated_over: set[int]) -> bool:
        del branches_with_updated_over
        return False

    def is_terminal_candidate(self) -> bool:
        return self.terminal_candidate


@dataclass(slots=True)
class _FakeNode:
    tree_evaluation: _FakeTreeEvaluation


def test_updater_calls_injected_backup_policy() -> None:
    """MinMaxEvaluationUpdater must route backup work through the injected policy."""
    fake_policy = _FakePolicy(
        calls=[],
        result=BackupResult(value_changed=True, pv_changed=True, over_changed=False),
    )
    node = _FakeNode(tree_evaluation=_FakeTreeEvaluation(backup_policy=fake_policy))
    updates = UpdateInstructionsTowardsOneParentNode(
        value_updates_toward_one_parent_node=ValueUpdateInstructionsTowardsOneParentNode(
            branches_with_updated_over={2},
            branches_with_updated_value={0, 1},
            branches_with_updated_best_branch_seq={1},
        ),
    )

    updater = MinMaxEvaluationUpdater()
    result = updater.perform_updates(
        node_to_update=cast("AlgorithmNode", node),
        updates_instructions=updates,
    )

    assert fake_policy.calls == [({0, 1}, {1})]
    assert result.new_value_for_node is True
    assert result.new_best_branch_for_node is True


def test_create_update_instructions_uses_terminal_candidate_predicate() -> None:
    """Birth updates should consume the terminal-from-Value predicate."""
    fake_policy = _FakePolicy(
        calls=[],
        result=BackupResult(value_changed=False, pv_changed=False, over_changed=False),
    )
    node = _FakeNode(
        tree_evaluation=_FakeTreeEvaluation(
            backup_policy=fake_policy,
            terminal_candidate=True,
        )
    )

    updater = MinMaxEvaluationUpdater()
    result = updater.create_update_instructions_after_node_birth(
        new_node=cast("AlgorithmNode", node),
    )

    assert result.is_node_newly_over is True


def test_create_update_instructions_terminal_candidate_false() -> None:
    """Birth updates should report non-over when terminal candidate is false."""
    fake_policy = _FakePolicy(
        calls=[],
        result=BackupResult(value_changed=False, pv_changed=False, over_changed=False),
    )
    node = _FakeNode(
        tree_evaluation=_FakeTreeEvaluation(
            backup_policy=fake_policy,
            terminal_candidate=False,
        )
    )

    updater = MinMaxEvaluationUpdater()
    result = updater.create_update_instructions_after_node_birth(
        new_node=cast("AlgorithmNode", node),
    )

    assert result.is_node_newly_over is False
