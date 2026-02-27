"""Tests for the single-agent objective semantics."""

from anemone.objectives import BackupInput, ChildInfo, SingleAgentObjective


def test_single_agent_sort_key_prefers_high_value() -> None:
    """Higher child value should rank before lower child value."""
    obj = SingleAgentObjective()
    key_hi = obj.child_sort_key(None, ChildInfo(value=10.0, depth=3, child_id=2))
    key_lo = obj.child_sort_key(None, ChildInfo(value=5.0, depth=1, child_id=1))
    assert key_hi < key_lo


def test_single_agent_backup_uses_max() -> None:
    """Default backup should select max child value."""
    obj = SingleAgentObjective()
    value = obj.backup(
        None,
        BackupInput(
            child_values=[1.0, 3.0, 2.0],
            prior_value=None,
            all_children_generated=True,
        ),
    )
    assert value == 3.0
