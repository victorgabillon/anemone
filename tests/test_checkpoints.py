"""Tests for checkpoint payloads, codecs, and low-level value serialization."""

from __future__ import annotations

from dataclasses import replace

import pytest
from valanga import Color, Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    DecisionOrderingCheckpointPayload,
    ExplorationIndexCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
    deserialize_value,
    serialize_value,
)
from anemone.node_evaluation.common import canonical_value


def test_node_checkpoint_can_store_opaque_external_state_ref() -> None:
    """Node checkpoint payloads should treat domain state refs as opaque."""
    state_ref = {"domain": "morpion", "variant": "5T", "played_moves": []}
    node = AlgorithmNodeCheckpointPayload(
        node_id=1,
        parent_node_id=None,
        branch_from_parent=None,
        depth=0,
        state_ref=state_ref,
        generated_all_branches=False,
    )

    assert node.state_ref == state_ref


def test_valanga_checkpoint_protocol_import_path_when_available() -> None:
    """The generic state-checkpoint protocol should come from Valanga."""
    checkpoints = pytest.importorskip(
        "valanga.checkpoints",
        reason="Valanga checkpoint protocol is not installed in this environment.",
    )

    assert hasattr(checkpoints, "StateCheckpointCodec")


def test_value_serialization_round_trip_preserves_semantics() -> None:
    """Checkpoint value helpers should preserve real runtime value semantics."""
    values = [
        replace(canonical_value.make_estimate_value(score=0.125), line=[1, 2]),
        replace(
            canonical_value.make_forced_value(
                score=0.75,
                over_event=OverEvent(outcome=Outcome.DRAW, termination="solved"),
            ),
            line=[3],
        ),
        replace(
            canonical_value.make_terminal_value(
                score=1.0,
                over_event=OverEvent(
                    outcome=Outcome.WIN,
                    winner=Color.WHITE,
                    termination="mate",
                ),
            ),
            line=[4, 5],
        ),
    ]

    for value in values:
        payload = serialize_value(value)
        restored = deserialize_value(payload)

        assert restored == value
        assert serialize_value(restored) == payload


def test_checkpoint_payload_dataclasses_support_nested_runtime_state() -> None:
    """Checkpoint payload dataclasses should compose into one nested runtime tree."""
    direct_value = serialize_value(
        replace(canonical_value.make_estimate_value(score=0.4), line=[0, 1])
    )
    backed_up_value = serialize_value(
        replace(
            canonical_value.make_forced_value(
                score=0.9,
                over_event=OverEvent(outcome=Outcome.DRAW, termination="solved"),
            ),
            line=[2],
        )
    )
    checkpoint = SearchRuntimeCheckpointPayload(
        evaluator_version=3,
        tree=TreeCheckpointPayload(
            root_node_id=1,
            nodes=[
                AlgorithmNodeCheckpointPayload(
                    node_id=1,
                    parent_node_id=None,
                    branch_from_parent=None,
                    depth=0,
                    state_ref={"tag": "root"},
                    generated_all_branches=True,
                    unopened_branches=[2, 3],
                    linked_children_by_branch={0: 2},
                    evaluation=NodeEvaluationCheckpointPayload(
                        direct_value=direct_value,
                        direct_evaluation_version=5,
                        backed_up_value=backed_up_value,
                        decision_ordering=DecisionOrderingCheckpointPayload(
                            branch_ordering=[
                                BranchOrderingCheckpointPayload(
                                    branch_key=0,
                                    primary_score=0.9,
                                    tactical_tiebreak=1,
                                    stable_tiebreak_id=2,
                                )
                            ]
                        ),
                        principal_variation=PrincipalVariationCheckpointPayload(
                            best_branch_sequence=[0, 1],
                            pv_version=4,
                            cached_best_child_version=11,
                        ),
                        branch_frontier=BranchFrontierCheckpointPayload(
                            frontier_branches=[0, 2]
                        ),
                        backup_runtime=BackupRuntimeCheckpointPayload(
                            best_branch=0,
                            second_best_branch=2,
                            exact_child_count=1,
                            selected_child_pv_version=11,
                            is_initialized=True,
                        ),
                    ),
                    exploration_index=ExplorationIndexCheckpointPayload(
                        kind="interval",
                        payload={"index": 1.5},
                    ),
                )
            ],
        ),
    )

    assert checkpoint.format_version == CHECKPOINT_FORMAT_VERSION
    assert checkpoint.tree.root_node_id == 1
    assert checkpoint.tree.nodes[0].evaluation is not None
    assert checkpoint.tree.nodes[0].evaluation.direct_value == direct_value
    assert checkpoint.tree.nodes[0].evaluation.backup_runtime is not None
    assert checkpoint.tree.nodes[0].evaluation.backup_runtime.best_branch == 0
    assert checkpoint.tree.nodes[0].linked_children_by_branch == {0: 2}
    assert checkpoint.tree.nodes[0].exploration_index is not None
    assert checkpoint.tree.nodes[0].exploration_index.kind == "interval"


def test_terminal_value_payload_preserves_over_event_fields() -> None:
    """Terminal-value payloads should keep explicit over-event metadata fields."""
    value = Value(
        score=1.0,
        certainty=Certainty.TERMINAL,
        over_event=OverEvent(
            outcome=Outcome.WIN,
            winner=Color.WHITE,
            termination="mate",
        ),
        line=[9],
    )

    payload = serialize_value(value)

    assert payload.over_event is not None
    assert payload.over_event.outcome == "WIN"
    assert payload.over_event.termination == "mate"
    assert payload.line == [9]
