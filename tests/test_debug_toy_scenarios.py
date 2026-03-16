"""Tests for the toy domains that exercise the real Anemone search stack."""

# ruff: noqa: D103
# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from dataclasses import replace
from random import Random
from typing import TYPE_CHECKING, Any

from valanga.evaluations import Certainty

from anemone.debug import LiveDebugEnvironment
from anemone.debug.toy_scenarios import (
    ToyScenarioSpec,
    all_toy_scenario_specs,
    build_deceptive_trap_scenario_spec,
    build_live_toy_debug_environment,
    build_minimax_micro_scenario_spec,
    build_minimax_semantic_stress_scenario_spec,
    build_single_agent_backup_scenario_spec,
    build_toy_tree_exploration,
)
from anemone.objectives import AdversarialZeroSumObjective, SingleAgentMaxObjective
from anemone.tree_exploration import TreeExploration

if TYPE_CHECKING:
    from pathlib import Path


def test_toy_scenario_specs_encode_expected_outcomes() -> None:
    scenario_specs = {
        scenario_spec.name: scenario_spec for scenario_spec in all_toy_scenario_specs()
    }

    assert set(scenario_specs) == {
        "single_agent_backup",
        "minimax_micro",
        "deceptive_trap",
        "minimax_semantic_stress",
    }
    assert scenario_specs["single_agent_backup"].expected_root_value == 8.0
    assert scenario_specs["single_agent_backup"].expected_pv == ("A", "A2")
    assert scenario_specs["minimax_micro"].expected_root_value == 4.0
    assert scenario_specs["minimax_micro"].expected_pv == ("A", "A1")
    assert scenario_specs["deceptive_trap"].expected_root_value == 6.0
    assert scenario_specs["deceptive_trap"].expected_pv == ("B", "B1")
    assert scenario_specs["minimax_semantic_stress"].expected_root_value is None
    assert scenario_specs["minimax_semantic_stress"].expected_pv == ()


def test_build_toy_tree_exploration_uses_real_engine_collaborators() -> None:
    exploration = build_toy_tree_exploration(build_single_agent_backup_scenario_spec())

    assert isinstance(exploration, TreeExploration)
    assert exploration.__class__.__module__ == "anemone.tree_exploration"
    assert exploration.node_selector.__class__.__module__.startswith(
        "anemone.node_selector"
    )
    assert exploration.tree_manager.__class__.__module__.startswith(
        "anemone.tree_manager"
    )
    assert exploration.tree.root_node.state.tag == "root"
    assert exploration.tree.root_node.tree_evaluation.direct_value is not None
    assert exploration.tree.root_node.tree_evaluation.backed_up_value is None
    assert isinstance(
        exploration.tree.root_node.tree_evaluation.objective,
        SingleAgentMaxObjective,
    )


def test_build_toy_tree_exploration_uses_adversarial_objective_for_minimax() -> None:
    exploration = build_toy_tree_exploration(build_minimax_micro_scenario_spec())

    assert isinstance(
        exploration.tree.root_node.tree_evaluation.objective,
        AdversarialZeroSumObjective,
    )


def test_build_toy_tree_exploration_supports_semantic_stress_minimax() -> None:
    exploration = build_toy_tree_exploration(
        build_minimax_semantic_stress_scenario_spec()
    )

    assert isinstance(
        exploration.tree.root_node.tree_evaluation.objective,
        AdversarialZeroSumObjective,
    )


def test_build_live_toy_debug_environment_creates_live_environment(
    tmp_path: Path,
) -> None:
    environment = build_live_toy_debug_environment(
        build_minimax_micro_scenario_spec(),
        tmp_path / "minimax-micro",
        snapshot_format="dot",
    )
    payload = json.loads(
        (environment.session_directory / "session.json").read_text(encoding="utf-8")
    )

    assert isinstance(environment, LiveDebugEnvironment)
    assert environment.session_directory == tmp_path / "minimax-micro"
    assert environment.session_directory.is_dir()
    assert (environment.session_directory / "session.json").exists()
    assert (environment.session_directory / "snapshots").is_dir()
    assert payload["is_live"] is True
    assert payload["entries"] == []


def test_single_agent_backup_scenario_runs_to_expected_result(tmp_path: Path) -> None:
    scenario_spec = build_single_agent_backup_scenario_spec()
    result, environment, payload = _run_scenario(
        scenario_spec,
        tmp_path / scenario_spec.name,
    )

    root_evaluation = environment.controlled_exploration.tree.root_node.tree_evaluation
    best_child = environment.controlled_exploration.tree.root_node.branches_children[
        "A"
    ]
    assert best_child is not None
    best_child_evaluation = best_child.tree_evaluation
    best_leaf = best_child.branches_children["A2"]
    assert best_leaf is not None

    assert result.branch_recommendation.recommended_name == "A"
    assert environment.controlled_exploration.tree.nodes_count > 1
    assert payload["entry_count"] > 0
    assert payload["is_live"] is True
    assert payload["is_complete"] is True
    assert root_evaluation.backed_up_value is not None
    assert root_evaluation.backed_up_value.score == scenario_spec.expected_root_value
    assert root_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert root_evaluation.has_exact_value()
    assert not root_evaluation.is_terminal()
    assert best_child_evaluation.backed_up_value is not None
    assert best_child_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert best_leaf.tree_evaluation.direct_value is not None
    assert best_leaf.tree_evaluation.direct_value.certainty is Certainty.TERMINAL
    assert tuple(root_evaluation.best_branch_sequence) == scenario_spec.expected_pv
    _assert_tree_fully_explored(environment, scenario_spec)


def test_minimax_micro_scenario_runs_to_expected_result(tmp_path: Path) -> None:
    scenario_spec = build_minimax_micro_scenario_spec()
    result, environment, _payload = _run_scenario(
        scenario_spec,
        tmp_path / scenario_spec.name,
    )

    root = environment.controlled_exploration.tree.root_node
    root_evaluation = root.tree_evaluation
    child_a = root.branches_children["A"]
    child_b = root.branches_children["B"]

    assert child_a is not None
    assert child_b is not None
    leaf_a1 = child_a.branches_children["A1"]
    leaf_b2 = child_b.branches_children["B2"]
    assert leaf_a1 is not None
    assert leaf_b2 is not None
    assert result.branch_recommendation.recommended_name == "A"
    assert child_a.tree_evaluation.backed_up_value is not None
    assert child_b.tree_evaluation.backed_up_value is not None
    assert child_a.tree_evaluation.backed_up_value.score == 4.0
    assert child_b.tree_evaluation.backed_up_value.score == 2.0
    assert child_a.tree_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert child_b.tree_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert root_evaluation.backed_up_value is not None
    assert root_evaluation.backed_up_value.score == scenario_spec.expected_root_value
    assert root_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert root_evaluation.has_exact_value()
    assert not root_evaluation.is_terminal()
    assert leaf_a1.tree_evaluation.direct_value is not None
    assert leaf_a1.tree_evaluation.direct_value.certainty is Certainty.TERMINAL
    assert leaf_b2.tree_evaluation.direct_value is not None
    assert leaf_b2.tree_evaluation.direct_value.certainty is Certainty.TERMINAL
    assert tuple(root_evaluation.best_branch_sequence) == scenario_spec.expected_pv
    _assert_tree_fully_explored(environment, scenario_spec)


def test_deceptive_trap_scenario_runs_to_expected_result(tmp_path: Path) -> None:
    scenario_spec = build_deceptive_trap_scenario_spec()
    result, environment, payload = _run_scenario(
        scenario_spec,
        tmp_path / scenario_spec.name,
    )

    root_evaluation = environment.controlled_exploration.tree.root_node.tree_evaluation
    backup_events = [
        entry
        for entry in payload["entries"]
        if entry["event_type"] == "BackupFinished"
        and entry["event_fields"].get("node_id")
        == str(environment.controlled_exploration.tree.root_node.id)
    ]

    assert result.branch_recommendation.recommended_name == "B"
    assert root_evaluation.backed_up_value is not None
    assert root_evaluation.backed_up_value.score == scenario_spec.expected_root_value
    assert root_evaluation.backed_up_value.certainty is Certainty.FORCED
    assert tuple(root_evaluation.best_branch_sequence) == scenario_spec.expected_pv
    assert any(entry["event_fields"].get("value_changed") for entry in backup_events)
    assert any(entry["event_fields"].get("pv_changed") for entry in backup_events)
    _assert_tree_fully_explored(environment, scenario_spec)


def test_same_tree_shape_differs_between_single_agent_and_minimax() -> None:
    minimax_spec = build_minimax_micro_scenario_spec()
    single_agent_spec = ToyScenarioSpec(
        name="single_agent_same_shape",
        root_id=minimax_spec.root_id,
        nodes={
            node_id: replace(node_spec, player="single")
            for node_id, node_spec in minimax_spec.nodes.items()
        },
        description="Same tree shape as minimax_micro but with single-agent backup.",
        expected_root_value=9.0,
        expected_pv=("B", "B2"),
    )

    single_agent_exploration = build_toy_tree_exploration(single_agent_spec)
    single_agent_result = single_agent_exploration.explore(random_generator=Random(0))
    minimax_exploration = build_toy_tree_exploration(minimax_spec)
    minimax_result = minimax_exploration.explore(random_generator=Random(0))

    single_agent_root = single_agent_exploration.tree.root_node.tree_evaluation
    minimax_root = minimax_exploration.tree.root_node.tree_evaluation

    assert single_agent_result.branch_recommendation.recommended_name == "B"
    assert minimax_result.branch_recommendation.recommended_name == "A"
    assert single_agent_root.backed_up_value is not None
    assert minimax_root.backed_up_value is not None
    assert single_agent_root.backed_up_value.score == 9.0
    assert minimax_root.backed_up_value.score == 4.0
    assert single_agent_root.backed_up_value.certainty is Certainty.FORCED
    assert minimax_root.backed_up_value.certainty is Certainty.FORCED
    assert tuple(single_agent_root.best_branch_sequence) == ("B", "B2")
    assert tuple(minimax_root.best_branch_sequence) == ("A", "A1")


def test_real_search_events_and_snapshots_are_recorded_for_toy_domain(
    tmp_path: Path,
) -> None:
    scenario_spec = build_single_agent_backup_scenario_spec()
    _result, environment, payload = _run_scenario(
        scenario_spec,
        tmp_path / "events-and-snapshots",
    )

    event_types = {entry["event_type"] for entry in payload["entries"]}
    metadata_file = next(
        entry["snapshot_metadata_file"]
        for entry in payload["entries"]
        if entry["snapshot_metadata_file"] is not None
    )
    snapshot_metadata = json.loads(
        (environment.session_directory / metadata_file).read_text(encoding="utf-8")
    )
    event_order = [entry["event_type"] for entry in payload["entries"]]

    assert {
        "NodeSelected",
        "ChildLinked",
        "DirectValueAssigned",
        "BackupStarted",
        "BackupFinished",
        "SearchIterationCompleted",
    }.issubset(event_types)
    assert event_order.index("NodeSelected") < event_order.index("ChildLinked")
    assert event_order.index("ChildLinked") < event_order.index("DirectValueAssigned")
    assert event_order.index("DirectValueAssigned") < event_order.index(
        "BackupFinished"
    )
    assert environment.recorder.to_trace().entries
    assert len(snapshot_metadata["nodes"]) > 1


def test_final_snapshot_metadata_marks_forced_root_and_terminal_leaf(
    tmp_path: Path,
) -> None:
    scenario_spec = build_single_agent_backup_scenario_spec()
    _result, environment, payload = _run_scenario(
        scenario_spec,
        tmp_path / "final-snapshot-certainty",
    )

    snapshot_metadata = _load_last_snapshot_metadata(environment, payload)
    root_node = _snapshot_node_by_state_tag(snapshot_metadata, "root")
    forced_leaf_parent = _snapshot_node_by_state_tag(snapshot_metadata, "A")
    terminal_leaf = _snapshot_node_by_state_tag(snapshot_metadata, "A2")

    assert root_node["is_exact"] is True
    assert root_node["is_terminal"] is False
    assert "certainty=FORCED" in (root_node["backed_up_value"] or "")
    assert root_node["over_event"] is not None

    assert forced_leaf_parent["is_exact"] is True
    assert forced_leaf_parent["is_terminal"] is False
    assert "certainty=FORCED" in (forced_leaf_parent["backed_up_value"] or "")

    assert terminal_leaf["is_exact"] is True
    assert terminal_leaf["is_terminal"] is True
    assert "certainty=TERMINAL" in (terminal_leaf["direct_value"] or "")
    assert terminal_leaf["over_event"] is not None


def test_minimax_final_snapshot_metadata_marks_forced_interior_nodes(
    tmp_path: Path,
) -> None:
    scenario_spec = build_minimax_micro_scenario_spec()
    _result, environment, payload = _run_scenario(
        scenario_spec,
        tmp_path / "minimax-final-snapshot-certainty",
    )

    snapshot_metadata = _load_last_snapshot_metadata(environment, payload)
    root_node = _snapshot_node_by_state_tag(snapshot_metadata, "root")
    interior_node = _snapshot_node_by_state_tag(snapshot_metadata, "A")
    terminal_leaf = _snapshot_node_by_state_tag(snapshot_metadata, "A1")

    assert root_node["is_exact"] is True
    assert root_node["is_terminal"] is False
    assert "certainty=FORCED" in (root_node["backed_up_value"] or "")

    assert interior_node["is_exact"] is True
    assert interior_node["is_terminal"] is False
    assert "certainty=FORCED" in (interior_node["backed_up_value"] or "")

    assert terminal_leaf["is_exact"] is True
    assert terminal_leaf["is_terminal"] is True
    assert "certainty=TERMINAL" in (terminal_leaf["direct_value"] or "")


def _run_scenario(
    scenario_spec: ToyScenarioSpec,
    session_directory: Path,
) -> tuple[Any, LiveDebugEnvironment, dict[str, Any]]:
    environment = build_live_toy_debug_environment(
        scenario_spec,
        session_directory,
        snapshot_format="dot",
    )
    result = environment.controlled_exploration.explore(random_generator=Random(0))
    environment.finalize()
    payload = json.loads(
        (environment.session_directory / "session.json").read_text(encoding="utf-8")
    )
    return result, environment, payload


def _assert_tree_fully_explored(
    environment: LiveDebugEnvironment,
    scenario_spec: ToyScenarioSpec,
) -> None:
    tree = environment.controlled_exploration.tree

    assert tree.nodes_count == len(scenario_spec.nodes)
    assert tree.branch_count == _scenario_edge_count(scenario_spec)


def _scenario_edge_count(scenario_spec: ToyScenarioSpec) -> int:
    return sum(len(node_spec.children) for node_spec in scenario_spec.nodes.values())


def _load_last_snapshot_metadata(
    environment: LiveDebugEnvironment,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metadata_file = next(
        entry["snapshot_metadata_file"]
        for entry in reversed(payload["entries"])
        if entry["snapshot_metadata_file"] is not None
    )
    return json.loads(
        (environment.session_directory / metadata_file).read_text(encoding="utf-8")
    )


def _snapshot_node_by_state_tag(
    snapshot_metadata: dict[str, Any],
    state_tag: str,
) -> dict[str, Any]:
    return next(
        node
        for node in snapshot_metadata["nodes"]
        if node.get("state_tag") == state_tag
    )
