"""Optional alternating depth policy and persistent selector-state contracts."""

from __future__ import annotations

from random import Random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from anemone.checkpoints import (
    LinooSelectorCheckpointPayload,
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.sharded_decode import _selector_payload_from_mapping
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.node_selector.linoo import Linoo, LinooArgs
from anemone.node_selector.linoo.depth_policy import choose_depth
from anemone.node_selector.node_selector_types import NodeSelectorType
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.test_checkpoint_load import (
    _CHILDREN_BY_ID,
    _build_linoo_args,
    _build_linoo_runtime,
    _ConcreteFakeYamlState,
    _FakeIncrementalStateCheckpointCodec,
    _values_for,
)
from tests.test_linoo_selector import _FakeOpeningInstructor, _node, _tree

if TYPE_CHECKING:
    from pathlib import Path

    from anemone.node_selector.linoo import LinooSelectionReport


def _fixture(policy: str = "alternating_by_step") -> tuple[Linoo[Any], Any]:
    """Return a frontier where odd and even depth choices can disagree."""
    selector = Linoo(
        opening_instructor=_FakeOpeningInstructor(),
        random_generator=Random(4),
        depth_selection_policy=policy,
    )
    return selector, _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, opened=True),
        _node(12, 1, score=1.0),
        _node(20, 2, score=2.0),
    )


def _select(selector: Linoo[Any], tree: Any) -> LinooSelectionReport:
    """Select without expanding so a fixed frontier isolates policy parity."""
    selector.choose_node_and_branch_to_open(tree, SimpleNamespace())
    assert selector.latest_selection_report is not None
    return selector.latest_selection_report


def _signature(report: Any) -> tuple[object, ...]:
    """Compare semantic diagnostics without timing or cache-hit counters."""
    return (
        report.selected_depth,
        report.selected_node_id,
        report.depth_selection_subpolicy,
        report.depth_selection_step,
        report.depth_selection_step_parity,
        report.selected_depth_selection_index,
        report.selected_depth_selection_weight,
        report.selected_depth_selection_probability,
        report.depth_rows,
    )


def test_alternating_is_explicit_and_preserves_public_default() -> None:
    """Both public defaults remain inverse-depth while alternating opts in."""
    assert (
        LinooArgs(type=NodeSelectorType.LINOO).depth_selection_policy == "inverse_depth"
    )
    default = Linoo(_FakeOpeningInstructor(), Random(4))
    assert default.depth_selection_policy == "inverse_depth"
    selector, tree = _fixture()
    reports = [_select(selector, tree) for _ in range(4)]
    assert [r.depth_selection_step for r in reports] == [1, 2, 3, 4]
    assert [r.depth_selection_subpolicy for r in reports] == [
        "inverse_depth",
        "opened_count_depth_index",
        "inverse_depth",
        "opened_count_depth_index",
    ]
    assert [r.depth_selection_step_parity for r in reports] == [
        "odd",
        "even",
        "odd",
        "even",
    ]
    assert [r.selected_node_id for r in reports] == [12, 20, 12, 20]
    assert all(r.selected_depth_selection_probability is not None for r in reports)
    assert "index" in reports[1].format_depth_table().splitlines()[0].split()


@pytest.mark.parametrize("save_after", [1, 2])
@pytest.mark.parametrize("mapping_roundtrip", [False, True])
def test_selector_payload_preserves_sequence_and_rng(
    save_after: int, mapping_roundtrip: bool
) -> None:
    """Both saved parities preserve the next selections and RNG state."""
    selector, tree = _fixture()
    for _ in range(save_after):
        _select(selector, tree)
    payload = selector.build_checkpoint_payload(
        tree.root_node.tree_evaluation.required_objective
    )
    assert payload is not None
    assert payload.selection_step_count == save_after
    if mapping_roundtrip:
        payload = _selector_payload_from_mapping(
            checkpoint_payload_to_jsonable(payload)
        )
    restored, _ = _fixture()
    restored.random_generator.setstate(selector.random_generator.getstate())
    assert restored.restore_from_checkpoint_payload(
        tree=tree,
        objective=tree.root_node.tree_evaluation.required_objective,
        payload=payload,
    )
    for _ in range(4):
        assert _signature(_select(restored, tree)) == _signature(
            _select(selector, tree)
        )
        assert (
            restored.random_generator.getstate() == selector.random_generator.getstate()
        )


@pytest.mark.parametrize("save_after", [0, 1, 2])
def test_rng_matches_standalone_subpolicy(save_after: int) -> None:
    """Odd draws match inverse-depth and even depth choice adds no draw."""
    selector, tree = _fixture()
    for _ in range(save_after):
        _select(selector, tree)
    policy = "inverse_depth" if save_after % 2 == 0 else "opened_count_depth_index"
    standalone, _ = _fixture(policy)
    before = selector.random_generator.getstate()
    standalone.random_generator.setstate(before)
    actual = _select(selector, tree)
    expected = _select(standalone, tree)
    assert actual.selected_depth == expected.selected_depth
    assert actual.selected_node_id == expected.selected_node_id
    assert (
        selector.random_generator.getstate() == standalone.random_generator.getstate()
    )
    # Node ranking has its own draws even on a deterministic depth step.
    # Check the depth-only dispatch separately without changing that behavior.
    depth_rng = Random()
    depth_rng.setstate(before)
    choose_depth(
        depth_selection_policy="alternating_by_step",
        depth_stats_by_depth=selector._depth_stats_by_depth,
        active_depths=(1, 2),
        random_generator=depth_rng,
        step=save_after + 1,
    )
    assert (depth_rng.getstate() == before) == (policy == "opened_count_depth_index")


@pytest.mark.parametrize(
    "cache_action", ["invalidate", "stale_payload", "save_invalidated"]
)
def test_counter_survives_cache_rebuild(cache_action: str) -> None:
    """Optional cache loss must not reset the durable selection count."""
    selector, tree = _fixture()
    _select(selector, tree)
    objective = tree.root_node.tree_evaluation.required_objective
    if cache_action == "invalidate":
        selector.invalidate()
    else:
        if cache_action == "save_invalidated":
            selector.invalidate()
        payload = selector.build_checkpoint_payload(objective)
        assert payload is not None
        if cache_action == "stale_payload":
            payload.node_states[0].node_id = 999999
        restored, _ = _fixture()
        restored.random_generator.setstate(selector.random_generator.getstate())
        assert not restored.restore_from_checkpoint_payload(
            tree=tree, objective=objective, payload=payload
        )
        selector = restored
    report = _select(selector, tree)
    assert report.depth_selection_step == 2
    assert report.depth_selection_subpolicy == "opened_count_depth_index"
    fresh, fresh_tree = _fixture()
    assert _select(fresh, fresh_tree).depth_selection_step == 1


def test_checkpoint_refresh_after_invalidation_preserves_count() -> None:
    """A save between invalidation and selection rebuilds without a new step."""
    selector, tree = _fixture()
    _select(selector, tree)
    selector.invalidate()
    objective = tree.root_node.tree_evaluation.required_objective
    selector.refresh_state_for_checkpoint(
        tree=tree, objective=objective, latest_tree_expansions=SimpleNamespace()
    )
    payload = selector.build_checkpoint_payload(objective)
    assert payload is not None
    assert payload.selection_step_count == 1
    assert payload.node_states


@pytest.mark.parametrize(
    "policy", ["inverse_depth", "opened_count_depth_index", "alternating_by_step"]
)
def test_legacy_counter_absence_has_explicit_fallback(policy: str) -> None:
    """Old payloads start at zero without changing either published policy."""
    selector, tree = _fixture(policy)
    _select(selector, tree)
    objective = tree.root_node.tree_evaluation.required_objective
    payload = selector.build_checkpoint_payload(objective)
    assert payload is not None
    wire = checkpoint_payload_to_jsonable(payload)
    wire.pop("selection_step_count")
    legacy = _selector_payload_from_mapping(wire)
    assert legacy.selection_step_count == 0
    restored, _ = _fixture(policy)
    restored.random_generator.setstate(selector.random_generator.getstate())
    assert restored.restore_from_checkpoint_payload(
        tree=tree, objective=objective, payload=legacy
    )
    expected, _ = _fixture(
        "inverse_depth" if policy == "alternating_by_step" else policy
    )
    expected.random_generator.setstate(selector.random_generator.getstate())
    actual = _select(restored, tree)
    control = _select(expected, tree)
    assert actual.depth_selection_step == 1
    assert (actual.selected_depth, actual.selected_node_id) == (
        control.selected_depth,
        control.selected_node_id,
    )
    assert restored.random_generator.getstate() == expected.random_generator.getstate()


@pytest.mark.parametrize("bad_count", [-1, True, 1.5, "1", None])
def test_invalid_counters_are_rejected(bad_count: Any) -> None:
    """Malformed saved counts must not silently choose a different parity."""
    with pytest.raises(ValueError, match="selection_step_count"):
        LinooSelectorCheckpointPayload(selection_step_count=bad_count)
    with pytest.raises(ValueError, match="selection_step_count"):
        _selector_payload_from_mapping({"selection_step_count": bad_count})


def test_failed_depth_selection_does_not_advance_counter() -> None:
    """An empty frontier has no successful depth choice to count."""
    selector, _ = _fixture()
    tree = _tree(_node(0, 0, opened=True))
    with pytest.raises(ValueError):
        _select(selector, tree)
    _, usable_tree = _fixture()
    assert _select(selector, usable_tree).depth_selection_step == 1


def test_alternating_requires_positive_selection_step() -> None:
    """The dispatch API cannot guess parity when no valid step is supplied."""
    for step in [None, 0, -1]:
        with pytest.raises(ValueError):
            choose_depth(
                depth_selection_policy="alternating_by_step",
                depth_stats_by_depth={},
                active_depths=(1,),
                random_generator=Random(0),
                step=step,
            )


@pytest.mark.parametrize("save_after", [1, 2])
@pytest.mark.parametrize(
    "policy", ["alternating_by_step", "inverse_depth", "opened_count_depth_index"]
)
@pytest.mark.parametrize("path", ["typed", "sharded", "streaming", "streaming_split"])
def test_runtime_checkpoint_sequence_equivalence(
    tmp_path: Path, save_after: int, policy: str, path: str
) -> None:
    """Actual runtime codecs preserve both parities and standalone policies."""
    runtime = _build_linoo_runtime()
    runtime.node_selector.base.depth_selection_policy = policy
    for _ in range(save_after):
        runtime.step()
    codec = _FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    assert payload.selector_state is not None
    assert payload.selector_state.selection_step_count == save_after
    assert payload.rng_state is not None
    args = _build_linoo_args()
    args.node_selector.base = LinooArgs(
        type=NodeSelectorType.LINOO, depth_selection_policy=policy
    )
    kwargs = dict(
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=args,
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for(_CHILDREN_BY_ID)
        ),
        random_generator=Random(999),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    if path == "typed":
        restored = load_search_from_checkpoint_payload(payload, **kwargs)
    else:
        write_sharded_search_checkpoint(
            payload,
            tmp_path,
            node_count_per_shard=2,
            layout="split" if path == "streaming_split" else "node_records",
        )
        if path == "sharded":
            decoded = load_sharded_search_checkpoint(tmp_path)
            assert decoded.selector_state is not None
            assert decoded.selector_state.selection_step_count == save_after
            restored = load_search_from_checkpoint_payload(decoded, **kwargs)
        else:
            restored = load_search_from_sharded_checkpoint(tmp_path, **kwargs)
    # The RNG passed by the caller is deliberately different; restore must use saved state.
    assert (
        restored.node_selector.base.random_generator.getstate()
        == runtime.node_selector.base.random_generator.getstate()
    )
    actual = restored.step().selector_report
    expected = runtime.step().selector_report
    assert actual is not None and expected is not None
    assert actual.depth_selection_step == save_after + 1
    assert _signature(actual) == _signature(expected)
    assert (
        restored.node_selector.base.random_generator.getstate()
        == runtime.node_selector.base.random_generator.getstate()
    )
