"""Focused tests for applying external scalar updates to live tree nodes."""
# ruff: noqa: D103

from typing import Any, cast

import pytest

from anemone import NodeValueUpdate, NodeValueUpdateResult
from tests.test_node_reevaluation import _build_runtime, _node_at


def _runtime_with_one_expansion() -> Any:
    """Build a small live runtime with root children already evaluated."""
    runtime = _build_runtime(
        value_by_id={
            0: 0.0,
            1: 0.1,
            2: 0.4,
            3: 0.0,
            4: 0.0,
        }
    )
    runtime.step()
    return runtime


def test_apply_node_value_updates_updates_existing_node_direct_value() -> None:
    runtime = _runtime_with_one_expansion()
    root = runtime.tree.root_node

    result = runtime.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(root.id), direct_value=0.33)]
    )

    assert result.requested_count == 1
    assert result.applied_count == 1
    assert result.missing_node_ids == ()
    assert root.tree_evaluation.direct_value is not None
    assert root.tree_evaluation.direct_value.score == pytest.approx(0.33)


def test_apply_node_value_updates_reports_missing_nodes() -> None:
    runtime = _runtime_with_one_expansion()
    child = _node_at(runtime, depth=1, node_id=1)

    result = runtime.apply_node_value_updates(
        [
            NodeValueUpdate(node_id=str(child.id), direct_value=0.2),
            NodeValueUpdate(node_id="missing-node", direct_value=0.7),
        ],
        allow_missing=True,
    )

    assert result.applied_count == 1
    assert result.missing_node_ids == ("missing-node",)


def test_apply_node_value_updates_rejects_missing_when_disallowed() -> None:
    runtime = _runtime_with_one_expansion()

    with pytest.raises(ValueError, match="missing node ids"):
        runtime.apply_node_value_updates(
            [NodeValueUpdate(node_id="missing-node", direct_value=0.7)],
            allow_missing=False,
        )


def test_apply_node_value_updates_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        NodeValueUpdate(node_id="1", direct_value=float("nan"))


def test_apply_node_value_updates_recomputes_backups() -> None:
    runtime = _runtime_with_one_expansion()
    root = runtime.tree.root_node
    child = _node_at(runtime, depth=1, node_id=1)
    root_value_before = root.tree_evaluation.get_value().score

    result = runtime.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(child.id), direct_value=1.0)],
        recompute_backups=True,
    )

    assert result.recomputed_count == 1
    assert root.tree_evaluation.get_value().score != pytest.approx(root_value_before)
    assert root.tree_evaluation.get_value().score == pytest.approx(1.0)


def test_apply_node_value_updates_can_skip_backup_recompute() -> None:
    runtime = _runtime_with_one_expansion()
    root = runtime.tree.root_node
    child = _node_at(runtime, depth=1, node_id=1)
    root_value_before = root.tree_evaluation.get_value().score

    result = runtime.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(child.id), direct_value=1.0)],
        recompute_backups=False,
    )

    assert result.recomputed_count is None
    assert child.tree_evaluation.direct_value is not None
    assert child.tree_evaluation.direct_value.score == pytest.approx(1.0)
    assert root.tree_evaluation.get_value().score == pytest.approx(root_value_before)


def test_node_value_update_result_is_stable_and_serializable_enough() -> None:
    result = NodeValueUpdateResult(
        requested_count=2,
        applied_count=1,
        missing_node_ids=cast("Any", ["missing-node"]),
    )

    assert result.missing_node_ids == ("missing-node",)
    assert result.skipped_node_ids == ()
