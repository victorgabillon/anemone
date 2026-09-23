"""Independent rollout RNG continuation through every runtime checkpoint path."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import TYPE_CHECKING, Any

import pytest
from valanga import Color

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    write_checkpoint_json_payload,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.sharded_decode import (
    _algorithm_node_payload_from_mapping,
    _selector_payload_from_mapping,
    _tree_expansions_payload_from_mapping,
)
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.tree_manager.opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.test_checkpoint_load import (
    _build_linoo_args,
    _ConcreteFakeYamlState,
    _FakeIncrementalStateCheckpointCodec,
    _values_for,
)
from tests.test_linoo_alternating import _signature

if TYPE_CHECKING:
    from pathlib import Path

# Root has one initial edge, so max_extra_steps=1 consumes exactly one choice
# in the first step. The deeper binary tree leaves ample frontier for continuation.
_GRAPH = {0: [1], **{i: [2 * i, 2 * i + 1] for i in range(1, 512)}}
_GRAPH.update({i: [] for i in range(512, 1024)})
_RANDOM_KINDS = [
    RolloutActionSelectorKind.RANDOM_OPENABLE,
    RolloutActionSelectorKind.RANDOM_LEGAL_PREFER_OPENABLE,
]
_PATHS = [
    "typed",
    "json",
    "json.gz",
    "json.zst",
    "sharded",
    "sharded_jsonl",
    "streaming",
    "streaming_split",
]


def _args(kind: RolloutActionSelectorKind, extra_steps: int | None = 1) -> Any:
    args = _build_linoo_args()
    args.node_selector.base = replace(
        args.node_selector.base, depth_selection_policy="alternating_by_step"
    )
    args.stopping_criterion.tree_branch_limit = 10000
    args.opening_expansion = OpeningExpansionConfig(
        kind=OpeningExpansionKind.ROLLOUT,
        rollout=RolloutExpansionConfig(
            max_extra_steps=extra_steps,
            action_selector_kind=kind,
            random_seed=0,
            stop_on_existing_node=False,
        ),
    )
    return args


def _runtime(args: Any) -> Any:
    return create_search_with_tree_eval_factory(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=_ConcreteFakeYamlState(0, _GRAPH, Color.WHITE),
        args=args,
        random_generator=Random(17),
        master_state_evaluator=MasterStateValueEvaluatorFromYaml(_values_for(_GRAPH)),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


def _selector(runtime: Any) -> Any:
    return runtime.tree_manager.opening_expansion_executor.rollout_action_selector


def _payload(runtime: Any) -> SearchRuntimeCheckpointPayload:
    return build_search_checkpoint_payload(
        runtime, state_codec=_FakeIncrementalStateCheckpointCodec(_GRAPH)
    )


def _decode_json(raw: Any) -> SearchRuntimeCheckpointPayload:
    # Anemone's JSON API returns raw mappings; use its field decoders to rebuild
    # typed payloads. Chipiron's real dacite adapter is tested in integration.
    return SearchRuntimeCheckpointPayload(
        evaluator_version=raw["evaluator_version"],
        format_version=raw["format_version"],
        tree=TreeCheckpointPayload(
            root_node_id=raw["tree"]["root_node_id"],
            nodes=[
                _algorithm_node_payload_from_mapping(n) for n in raw["tree"]["nodes"]
            ],
        ),
        rng_state=raw.get("rng_state"),
        rollout_rng_state=raw.get("rollout_rng_state"),
        selector_state=(
            None
            if raw.get("selector_state") is None
            else _selector_payload_from_mapping(raw["selector_state"])
        ),
        latest_tree_expansions=(
            None
            if raw.get("latest_tree_expansions") is None
            else _tree_expansions_payload_from_mapping(raw["latest_tree_expansions"])
        ),
    )


def _restore(payload: Any, args: Any, path: str, directory: Path) -> Any:
    kwargs = dict(
        state_codec=_FakeIncrementalStateCheckpointCodec(_GRAPH),
        dynamics=FakeYamlDynamics(),
        args=args,
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for(_GRAPH)
        ),
        random_generator=Random(999),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    if path.startswith("json"):
        filename = directory / f"checkpoint.{path}"
        write_checkpoint_json_payload(payload, filename)
        raw, _ = load_checkpoint_json_payload(filename)
        payload = _decode_json(raw)
    elif path != "typed":
        write_sharded_search_checkpoint(
            payload,
            directory,
            node_count_per_shard=3,
            layout="split" if path.endswith("_split") else "node_records",
            encoding="jsonl"
            if path in {"sharded_jsonl", "streaming_split"}
            else "jsonl_zst",
        )
        if path.startswith("streaming"):
            return load_search_from_sharded_checkpoint(directory, **kwargs)
        payload = load_sharded_search_checkpoint(directory)
    return load_search_from_checkpoint_payload(payload, **kwargs)


def _rng_states(runtime: Any) -> tuple[object, object]:
    search_rng = runtime.node_selector.base.random_generator
    rollout_rng = _selector(runtime).random_generator
    assert search_rng is not rollout_rng
    return search_rng.getstate(), rollout_rng.getstate()


def _tree(runtime: Any) -> object:
    return checkpoint_payload_to_jsonable(_payload(runtime).tree)


def _step(runtime: Any) -> object:
    report = runtime.step()
    return (
        report.selected_node_id,
        report.selected_depth,
        report.nodes_before,
        report.nodes_after,
        report.nodes_added,
        report.branch_count,
        _signature(report.selector_report),
        runtime.tree_manager.latest_rollout_report,
    )


@pytest.mark.parametrize("kind", _RANDOM_KINDS)
@pytest.mark.parametrize("save_after", [0, 1, 3])
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("extra_steps", [1, None])
def test_independent_rollout_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: RolloutActionSelectorKind,
    save_after: int,
    path: str,
    extra_steps: int | None,
) -> None:
    """Compare multiple actual decisions, complete tree payloads, reports and RNGs."""
    args = _args(kind, extra_steps)
    live = _runtime(args)
    selector_type = type(_selector(live))
    choose = selector_type.choose_action
    actions: dict[int, list[object]] = {}

    def record(selector: Any, context: Any) -> object:
        action = choose(selector, context)
        actions.setdefault(id(selector), []).append((context.current_node.id, action))
        return action

    monkeypatch.setattr(selector_type, "choose_action", record)
    for _ in range(save_after):
        live.step()
    choices = len(actions.get(id(_selector(live)), []))
    if save_after == 0:
        assert choices == 0
    elif save_after == 1 and extra_steps == 1:
        assert choices == 1
    else:
        assert choices > 1
    states = _rng_states(live)
    payload = _payload(live)
    assert _rng_states(live) == states  # saving consumes neither stream
    assert payload.rng_state == states[0]
    assert payload.rollout_rng_state == states[1]
    restored = _restore(payload, args, path, tmp_path)
    assert _tree(restored) == _tree(live)
    assert _rng_states(restored) == states
    assert _selector(restored).random_generator is not _selector(live).random_generator
    # Factory-created live selectors are not the optional manager override.
    assert restored.tree_manager.rollout_action_selector is None
    continued_choices = 0
    for _ in range(4):
        actions.clear()
        assert _step(restored) == _step(live)
        actual = actions.get(id(_selector(restored)), [])
        assert actual == actions.get(id(_selector(live)), [])
        continued_choices += len(actual)
        assert _tree(restored) == _tree(live)
        assert _rng_states(restored) == _rng_states(live)
    assert continued_choices > 0


@pytest.mark.parametrize("kind", _RANDOM_KINDS)
@pytest.mark.parametrize("path", _PATHS)
def test_legacy_missing_rollout_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: RolloutActionSelectorKind,
    path: str,
) -> None:
    """Absent legacy metadata keeps historical seed reset without losing search RNG."""
    import anemone.checkpoints.sharded_encode as sharded_encode

    args = _args(kind)
    live = _runtime(args)
    for _ in range(3):
        live.step()
    payload = _payload(live)
    payload.rollout_rng_state = None
    if path.startswith("json"):
        payload = checkpoint_payload_to_jsonable(payload)
        del payload["rollout_rng_state"]
    elif path != "typed":
        write = sharded_encode._write_json_shard

        def legacy_write(data: Any, **kwargs: Any) -> Any:
            if kwargs.get("kind") == "metadata":
                data = {k: v for k, v in data.items() if k != "rollout_rng_state"}
            return write(data, **kwargs)

        monkeypatch.setattr(sharded_encode, "_write_json_shard", legacy_write)
    restored = _restore(payload, args, path, tmp_path)
    assert _rng_states(restored)[0] == _rng_states(live)[0]
    assert _rng_states(restored)[1] == Random(0).getstate()
    assert _rng_states(restored)[1] != _rng_states(live)[1]
    assert _tree(restored) == _tree(live)
    diverged = False
    for _ in range(4):
        reports_differ = _step(restored) != _step(live)
        diverged |= reports_differ or _tree(restored) != _tree(live)
    assert diverged  # reset RNG must alter the fixture trajectory, not just state


@pytest.mark.parametrize(
    "kind",
    [
        RolloutActionSelectorKind.FIRST_OPENABLE,
        RolloutActionSelectorKind.FIRST_LEGAL_PREFER_OPENABLE,
        RolloutActionSelectorKind.NO_ROLLOUT,
        None,
    ],
)
@pytest.mark.parametrize("path", _PATHS)
def test_deterministic_and_one_ply_have_no_rollout_rng(
    tmp_path: Path,
    kind: RolloutActionSelectorKind | None,
    path: str,
) -> None:
    args = _args(kind or RolloutActionSelectorKind.NO_ROLLOUT)
    if kind is None:
        args.opening_expansion = OpeningExpansionConfig()
    live = _runtime(args)
    live.step()
    payload = _payload(live)
    assert payload.rollout_rng_state is None
    restored = _restore(payload, args, path, tmp_path)
    for _ in range(3):
        assert _step(live) == _step(restored)
        assert _tree(live) == _tree(restored)


@pytest.mark.parametrize("kind", _RANDOM_KINDS)
def test_export_does_not_change_uninterrupted_search(
    kind: RolloutActionSelectorKind,
) -> None:
    """Frequent saves cannot add/remove RNG draws or alter ordinary execution."""
    live, control = _runtime(_args(kind)), _runtime(_args(kind))
    for _ in range(6):
        _payload(live)
        assert _step(live) == _step(control)
        assert _rng_states(live) == _rng_states(control)


def test_saved_rollout_state_requires_random_selector(tmp_path: Path) -> None:
    live = _runtime(_args(_RANDOM_KINDS[0]))
    with pytest.raises(ValueError, match="live selector has no RNG"):
        _restore(
            _payload(live),
            _args(RolloutActionSelectorKind.NO_ROLLOUT),
            "typed",
            tmp_path,
        )


@pytest.mark.parametrize("kind", _RANDOM_KINDS)
def test_export_reads_live_executor_not_manager_override(
    kind: RolloutActionSelectorKind,
) -> None:
    live = _runtime(_args(kind))
    live.step()
    actual_selector = _selector(live)
    decoy = type(actual_selector)(Random(123))
    live.tree_manager.rollout_action_selector = decoy
    payload = _payload(live)
    assert payload.rollout_rng_state == actual_selector.random_generator.getstate()
    assert payload.rollout_rng_state != decoy.random_generator.getstate()
