"""Tests for training-export builders and model behavior."""
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PACKAGE_ROOT = _REPO_ROOT / "src" / "anemone"

try:
    import anemone as _anemone  # noqa: F401
except ModuleNotFoundError:
    _stub_package = ModuleType("anemone")
    _stub_package.__path__ = [str(_SRC_PACKAGE_ROOT)]
    sys.modules["anemone"] = _stub_package

from anemone.checkpoints import AnchorCheckpointStatePayload
from anemone.checkpoints.state_handles import CheckpointBackedStateHandle
from anemone.node_evaluation.common import (
    NodeTargetSource,
    ValueCandidate,
    ValueCandidateSource,
)
from anemone.training_export import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
    EffectiveValueSourceMissingError,
    build_training_node_snapshot,
    build_training_tree_snapshot,
    state_ref_payload_without_resolving,
)
from anemone.utils.logger import anemone_logger


class _DummyOverEvent:
    """Tiny over-event-like object with a stable string tag."""

    def get_over_tag(self) -> str:
        """Return a compact marker."""
        return "forced-win"


@dataclass(frozen=True, slots=True)
class _ScoreValue:
    """Tiny score-carrying value-like object."""

    score: float


@dataclass(slots=True)
class _RichDummyNode:
    """Dummy node exposing the direct attribute surface used by the builder."""

    node_id: str
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    depth: int
    state: object
    direct_value: object | None
    backed_up_value: object | None
    is_terminal: bool
    is_exact: bool
    visit_count: int
    over_event: object | None = None


@dataclass(slots=True)
class _SourceAwareEvaluation:
    """Evaluation-like object exposing direct/tree/effective provenance."""

    direct_value: _ScoreValue | None
    backed_up_value: _ScoreValue | None
    effective_source: ValueCandidateSource

    @property
    def tree_value(self) -> _ScoreValue | None:
        """Return the tree-derived value."""
        return self.backed_up_value

    def get_effective_value_candidate(self) -> ValueCandidate:
        """Return a source-aware candidate."""
        if self.effective_source is ValueCandidateSource.DIRECT_SELF:
            assert self.direct_value is not None
            return ValueCandidate.direct(self.direct_value)
        if self.effective_source is ValueCandidateSource.TREE_CHILD:
            assert self.backed_up_value is not None
            return ValueCandidate.tree(self.backed_up_value)
        return ValueCandidate.none()


@dataclass(slots=True)
class _EvaluationNode:
    """Node-like object with a tree evaluation."""

    node_id: str
    tree_evaluation: _SourceAwareEvaluation
    all_branches_generated: bool = False


@dataclass(slots=True)
class _EvaluationWithSourceMissing:
    """Evaluation-like object with invalid effective provenance."""

    direct_value: _ScoreValue | None
    backed_up_value: _ScoreValue | None
    effective_value: _ScoreValue | None

    @property
    def tree_value(self) -> _ScoreValue | None:
        """Return the tree-derived value."""
        return self.backed_up_value


class _MinimalDummyNode:
    """Dummy node exposing only a stable node id."""

    node_id = "solo"


@dataclass(slots=True)
class _PayloadResolver:
    """Minimal resolver shape for reusable checkpoint payload tests."""

    payloads_by_node_id: dict[int, object]

    def payload_for_node_id_or_none(self, node_id: int) -> object | None:
        """Return one configured raw payload."""
        return self.payloads_by_node_id.get(node_id)


class _StateTrackingNode:
    """Node that records state access and can reject unexpected resolution."""

    def __init__(
        self,
        *,
        state_handle: object | None = None,
        state: object | None = None,
        allow_state_access: bool = True,
    ) -> None:
        """Initialize the node with optional state and lazy handle."""
        self.state_handle = state_handle
        self._state = state
        self.allow_state_access = allow_state_access
        self.state_access_count = 0

    @property
    def state(self) -> object | None:
        """Return configured state or fail when access is forbidden."""
        self.state_access_count += 1
        if not self.allow_state_access:
            raise AssertionError
        return self._state


@dataclass(slots=True)
class _TreeNodeWrapper:
    """Wrapper exposing the real state handle under ``tree_node``."""

    state_handle: object


@dataclass(slots=True)
class _StateRefProfile:
    """Small profiler spy for checkpoint-aware state-ref export."""

    observed_nodes: int = 0
    state_access_calls: int = 0
    state_ref_conversions: int = 0
    state_present_values: list[bool] = field(default_factory=list)

    def observe_state_handle(self, node: object) -> None:
        """Record one raw-handle observation."""
        del node
        self.observed_nodes += 1

    def record_state_access(self, elapsed_s: float, *, state_present: bool) -> None:
        """Record one materialized state access."""
        assert elapsed_s >= 0.0
        self.state_access_calls += 1
        self.state_present_values.append(state_present)

    def record_state_ref_conversion(self, elapsed_s: float) -> None:
        """Record one state-ref conversion."""
        assert elapsed_s >= 0.0
        self.state_ref_conversions += 1


def test_state_ref_payload_without_resolving_uses_checkpoint_payload_fast_path() -> (
    None
):
    """Checkpoint-backed reusable payloads should avoid ``node.state``."""
    payload = AnchorCheckpointStatePayload(anchor_ref={"anchor": 1})
    resolver = _PayloadResolver(payloads_by_node_id={7: payload})
    node = _StateTrackingNode(
        state_handle=CheckpointBackedStateHandle(
            resolver=resolver,
            node_id=7,
        ),
        allow_state_access=False,
    )
    profile = _StateRefProfile()

    state_ref_payload = state_ref_payload_without_resolving(
        node,
        checkpoint_payload_to_state_ref=lambda handle, raw_payload: {
            "node_id": handle.node_id,
            "payload": raw_payload,
        },
        materialized_state_to_state_ref=lambda state: {"state": state},
        profile=profile,
    )

    assert state_ref_payload == {"node_id": 7, "payload": payload}
    assert node.state_access_count == 0
    assert profile.observed_nodes == 1
    assert profile.state_access_calls == 0
    assert profile.state_ref_conversions == 1


def test_state_ref_payload_without_resolving_falls_back_to_materialized_state() -> None:
    """Missing checkpoint payloads should fall back to one state access."""
    resolver = _PayloadResolver(payloads_by_node_id={})
    node = _StateTrackingNode(
        state_handle=CheckpointBackedStateHandle(
            resolver=resolver,
            node_id=7,
        ),
        state={"board": [1]},
    )
    profile = _StateRefProfile()

    state_ref_payload = state_ref_payload_without_resolving(
        node,
        checkpoint_payload_to_state_ref=lambda handle, payload: {
            "node_id": handle.node_id,
            "payload": payload,
        },
        materialized_state_to_state_ref=lambda state: {"state": state},
        profile=profile,
    )

    assert state_ref_payload == {"state": {"board": [1]}}
    assert node.state_access_count == 1
    assert profile.observed_nodes == 1
    assert profile.state_access_calls == 1
    assert profile.state_present_values == [True]
    assert profile.state_ref_conversions == 1


def test_state_ref_payload_without_resolving_falls_back_when_converter_returns_none() -> (
    None
):
    """Unsupported checkpoint payloads should allow materialized fallback."""
    payload = AnchorCheckpointStatePayload(anchor_ref={"anchor": 1})
    resolver = _PayloadResolver(payloads_by_node_id={7: payload})
    node = _StateTrackingNode(
        state_handle=CheckpointBackedStateHandle(
            resolver=resolver,
            node_id=7,
        ),
        state={"board": [2]},
    )

    state_ref_payload = state_ref_payload_without_resolving(
        node,
        checkpoint_payload_to_state_ref=lambda handle, raw_payload: None,
        materialized_state_to_state_ref=lambda state: {"state": state},
    )

    assert state_ref_payload == {"state": {"board": [2]}}
    assert node.state_access_count == 1


def test_state_ref_payload_without_resolving_returns_none_for_missing_state() -> None:
    """Fallback with no materialized state should return ``None``."""
    resolver = _PayloadResolver(payloads_by_node_id={})
    node = _StateTrackingNode(
        state_handle=CheckpointBackedStateHandle(
            resolver=resolver,
            node_id=7,
        ),
        state=None,
    )
    profile = _StateRefProfile()

    state_ref_payload = state_ref_payload_without_resolving(
        node,
        checkpoint_payload_to_state_ref=lambda handle, payload: payload,
        materialized_state_to_state_ref=lambda state: {"state": state},
        profile=profile,
    )

    assert state_ref_payload is None
    assert node.state_access_count == 1
    assert profile.state_access_calls == 1
    assert profile.state_present_values == [False]
    assert profile.state_ref_conversions == 0


def test_state_ref_payload_without_resolving_supports_tree_node_handle() -> None:
    """Wrapped nodes can expose checkpoint handles through ``tree_node``."""
    payload = AnchorCheckpointStatePayload(anchor_ref={"anchor": 1})
    resolver = _PayloadResolver(payloads_by_node_id={7: payload})
    handle = CheckpointBackedStateHandle(resolver=resolver, node_id=7)
    node = _StateTrackingNode(allow_state_access=False)
    node.tree_node = _TreeNodeWrapper(state_handle=handle)

    state_ref_payload = state_ref_payload_without_resolving(
        node,
        checkpoint_payload_to_state_ref=lambda current_handle, raw_payload: {
            "node_id": current_handle.node_id,
            "payload": raw_payload,
        },
        materialized_state_to_state_ref=lambda state: {"state": state},
    )

    assert state_ref_payload == {"node_id": 7, "payload": payload}
    assert node.state_access_count == 0


def test_build_training_tree_snapshot_with_dummy_nodes() -> None:
    """Builder should preserve ids, payloads, scalars, flags, and metadata."""
    root = _RichDummyNode(
        node_id="root",
        parent_ids=(),
        child_ids=("child",),
        depth=0,
        state={"raw_state": [1, 2, 3]},
        direct_value=1,
        backed_up_value=2.5,
        is_terminal=True,
        is_exact=True,
        visit_count=7,
        over_event=_DummyOverEvent(),
    )
    child = _RichDummyNode(
        node_id="child",
        parent_ids=("root",),
        child_ids=(),
        depth=1,
        state={"raw_state": [4]},
        direct_value=0.25,
        backed_up_value=None,
        is_terminal=False,
        is_exact=False,
        visit_count=2,
    )

    snapshot = build_training_tree_snapshot(
        [root, child],
        state_ref_dumper=lambda state: {"state_key": state},
        direct_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
        backed_up_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
        created_at_unix_s=123.5,
        metadata={"source": "unit-test"},
    )

    assert snapshot.root_node_id == "root"
    assert snapshot.created_at_unix_s == 123.5
    assert snapshot.metadata["format_kind"] == TRAINING_TREE_SNAPSHOT_FORMAT_KIND
    assert snapshot.metadata["format_version"] == (
        TRAINING_TREE_SNAPSHOT_FORMAT_VERSION
    )
    assert snapshot.metadata["default_target_source"] == "tree_value"
    assert snapshot.metadata["source"] == "unit-test"
    assert snapshot.nodes[0].state_ref_payload == {
        "state_key": {"raw_state": [1, 2, 3]}
    }
    assert snapshot.nodes[0].direct_value_scalar == 1.0
    assert snapshot.nodes[0].tree_value_scalar == 2.5
    assert snapshot.nodes[0].backed_up_value_scalar == 2.5
    assert snapshot.nodes[0].effective_value_scalar is None
    assert snapshot.nodes[0].effective_value_source == "none"
    assert snapshot.nodes[0].target_value_scalar == 2.5
    assert snapshot.nodes[0].is_terminal is True
    assert snapshot.nodes[0].is_exact is True
    assert snapshot.nodes[0].visit_count == 7
    assert snapshot.nodes[0].over_event_label == "forced-win"
    assert snapshot.nodes[1].parent_ids == ("root",)
    assert snapshot.nodes[1].child_ids == ()


def test_build_training_tree_snapshot_preserves_explicit_root_node_id() -> None:
    """Builder should keep an explicit root override unchanged."""
    root = _RichDummyNode(
        node_id="root",
        parent_ids=(),
        child_ids=(),
        depth=0,
        state={"raw_state": [1]},
        direct_value=None,
        backed_up_value=None,
        is_terminal=False,
        is_exact=False,
        visit_count=1,
    )

    snapshot = build_training_tree_snapshot([root], root_node_id="manual-root")

    assert snapshot.root_node_id == "manual-root"


def test_build_training_tree_snapshot_logs_structured_phases(
    caplog: object,
) -> None:
    """Builder should emit structured training-export phase logs."""
    root = _RichDummyNode(
        node_id="root",
        parent_ids=(),
        child_ids=("child",),
        depth=0,
        state={"raw_state": [1, 2, 3]},
        direct_value=1,
        backed_up_value=2.5,
        is_terminal=True,
        is_exact=True,
        visit_count=7,
    )
    child = _RichDummyNode(
        node_id="child",
        parent_ids=("root",),
        child_ids=(),
        depth=1,
        state={"raw_state": [4]},
        direct_value=0.25,
        backed_up_value=None,
        is_terminal=False,
        is_exact=False,
        visit_count=2,
    )

    old_propagate = anemone_logger.propagate
    anemone_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=anemone_logger.name):
            snapshot = build_training_tree_snapshot(
                [root, child],
                state_ref_dumper=lambda state: {"state_key": state},
            )
    finally:
        anemone_logger.propagate = old_propagate

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[training-export] phase=snapshot_build status=start node_count=2" in message
        for message in messages
    )
    assert any(
        "[training-export] phase=payload_build status=done" in message
        and "elapsed_s=" in message
        and "node_count=2" in message
        for message in messages
    )
    assert any(
        "[training-export] phase=state_ref_serialization status=done" in message
        and "state_ref_count=2" in message
        for message in messages
    )
    assert snapshot.root_node_id == "root"


def test_build_training_node_snapshot_handles_missing_optional_fields() -> None:
    """Builder should default missing optional fields cleanly."""
    snapshot = build_training_node_snapshot(_MinimalDummyNode())

    assert snapshot.node_id == "solo"
    assert snapshot.parent_ids == ()
    assert snapshot.child_ids == ()
    assert snapshot.depth == 0
    assert snapshot.state_ref_payload is None
    assert snapshot.direct_value_scalar is None
    assert snapshot.tree_value_scalar is None
    assert snapshot.backed_up_value_scalar is None
    assert snapshot.effective_value_scalar is None
    assert snapshot.effective_value_source == "none"
    assert snapshot.target_value_scalar is None
    assert snapshot.is_terminal is False
    assert snapshot.is_exact is False
    assert snapshot.over_event_label is None
    assert snapshot.visit_count is None
    assert snapshot.metadata == {}


def test_build_training_node_snapshot_without_value_extractors_keeps_scalars_none() -> (
    None
):
    """Builder should leave scalar fields unset when no extractors are given."""
    node = _RichDummyNode(
        node_id="rich",
        parent_ids=(),
        child_ids=(),
        depth=2,
        state={"nested": {"board": [0, 1]}},
        direct_value=3,
        backed_up_value=4,
        is_terminal=False,
        is_exact=True,
        visit_count=9,
    )

    snapshot = build_training_node_snapshot(
        node,
        state_ref_dumper=lambda state: {"state_key": state},
    )

    assert snapshot.state_ref_payload == {"state_key": {"nested": {"board": [0, 1]}}}
    assert snapshot.direct_value_scalar is None
    assert snapshot.tree_value_scalar is None
    assert snapshot.backed_up_value_scalar is None
    assert snapshot.effective_value_scalar is None
    assert snapshot.target_value_scalar is None


def test_build_training_node_snapshot_keeps_structured_payloads_and_numeric_scalars() -> (
    None
):
    """Builder should keep payloads structured and value scalars numeric."""
    node = _RichDummyNode(
        node_id="rich",
        parent_ids=(),
        child_ids=(),
        depth=2,
        state={"nested": {"board": [0, 1]}},
        direct_value=3,
        backed_up_value=4,
        is_terminal=False,
        is_exact=True,
        visit_count=9,
    )

    snapshot = build_training_node_snapshot(
        node,
        state_ref_dumper=lambda state: {"state_key": state},
        direct_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
        backed_up_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
    )

    assert isinstance(snapshot.state_ref_payload, dict)
    assert snapshot.state_ref_payload == {"state_key": {"nested": {"board": [0, 1]}}}
    assert isinstance(snapshot.direct_value_scalar, float)
    assert isinstance(snapshot.tree_value_scalar, float)
    assert isinstance(snapshot.backed_up_value_scalar, float)
    assert snapshot.direct_value_scalar == 3.0
    assert snapshot.tree_value_scalar == 4.0
    assert snapshot.backed_up_value_scalar == 4.0
    assert snapshot.effective_value_scalar is None
    assert snapshot.target_value_scalar == 4.0


def test_training_snapshot_separates_direct_tree_and_effective_values() -> None:
    """Partial-node snapshots expose value provenance distinctly."""
    node = _EvaluationNode(
        node_id="partial",
        tree_evaluation=_SourceAwareEvaluation(
            direct_value=_ScoreValue(0.7),
            backed_up_value=_ScoreValue(0.2),
            effective_source=ValueCandidateSource.DIRECT_SELF,
        ),
    )

    snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
    )

    assert snapshot.direct_value_scalar == 0.7
    assert snapshot.tree_value_scalar == 0.2
    assert snapshot.backed_up_value_scalar == 0.2
    assert snapshot.effective_value_scalar == 0.7
    assert snapshot.effective_value_source == "direct_self"
    assert snapshot.target_value_scalar == 0.2


def test_training_snapshot_uses_tree_value_when_fully_opened() -> None:
    """Fully opened diagnostics still show direct and tree separately."""
    node = _EvaluationNode(
        node_id="full",
        all_branches_generated=True,
        tree_evaluation=_SourceAwareEvaluation(
            direct_value=_ScoreValue(0.9),
            backed_up_value=_ScoreValue(0.4),
            effective_source=ValueCandidateSource.TREE_CHILD,
        ),
    )

    snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
    )

    assert snapshot.direct_value_scalar == 0.9
    assert snapshot.tree_value_scalar == 0.4
    assert snapshot.effective_value_scalar == 0.4
    assert snapshot.effective_value_source == "tree_child"
    assert snapshot.target_value_scalar == 0.4


def test_training_target_does_not_fallback_to_direct_without_tree_value() -> None:
    """Tree targets stay absent instead of self-distilling direct estimates."""
    node = _EvaluationNode(
        node_id="direct-only",
        tree_evaluation=_SourceAwareEvaluation(
            direct_value=_ScoreValue(0.7),
            backed_up_value=None,
            effective_source=ValueCandidateSource.DIRECT_SELF,
        ),
    )

    snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
    )

    assert snapshot.direct_value_scalar == 0.7
    assert snapshot.tree_value_scalar is None
    assert snapshot.effective_value_scalar == 0.7
    assert snapshot.effective_value_source == "direct_self"
    assert snapshot.target_value_scalar is None


def test_effective_value_without_source_is_rejected() -> None:
    """Effective values must carry explicit provenance."""
    node = _EvaluationNode(
        node_id="invalid",
        tree_evaluation=_EvaluationWithSourceMissing(
            direct_value=_ScoreValue(0.7),
            backed_up_value=_ScoreValue(0.2),
            effective_value=_ScoreValue(0.7),
        ),
    )

    with pytest.raises(EffectiveValueSourceMissingError):
        build_training_node_snapshot(
            node,
            direct_value_extractor=lambda value: (
                value.score if value is not None else None
            ),
            tree_value_extractor=lambda value: (
                value.score if value is not None else None
            ),
        )


def test_training_target_source_must_be_explicit_for_effective_or_direct() -> None:
    """Non-tree target choices are opt-in."""
    node = _EvaluationNode(
        node_id="partial",
        tree_evaluation=_SourceAwareEvaluation(
            direct_value=_ScoreValue(0.7),
            backed_up_value=_ScoreValue(0.2),
            effective_source=ValueCandidateSource.DIRECT_SELF,
        ),
    )

    default_snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
    )
    effective_snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
        target_source=NodeTargetSource.EFFECTIVE_VALUE,
    )
    direct_snapshot = build_training_node_snapshot(
        node,
        direct_value_extractor=lambda value: value.score if value is not None else None,
        tree_value_extractor=lambda value: value.score if value is not None else None,
        target_source=NodeTargetSource.DIRECT_VALUE,
    )

    assert default_snapshot.target_value_scalar == 0.2
    assert effective_snapshot.target_value_scalar == 0.7
    assert direct_snapshot.target_value_scalar == 0.7
