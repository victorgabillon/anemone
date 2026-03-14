"""Predefined deterministic toy scenarios for manual GUI inspection."""

from __future__ import annotations

from .model import ToyNodeSpec, ToyScenarioSpec


def build_single_agent_backup_scenario_spec() -> ToyScenarioSpec:
    """Return a tiny single-agent max-backup scenario."""
    nodes = {
        "root": ToyNodeSpec(
            node_id="root",
            player="single",
            children={"A": "A", "B": "B"},
            state_tag="root",
        ),
        "A": ToyNodeSpec(
            node_id="A",
            player="single",
            children={"A1": "A1", "A2": "A2"},
            state_tag="A",
        ),
        "A1": ToyNodeSpec(
            node_id="A1",
            player="single",
            terminal_value=3.0,
            state_tag="A1",
        ),
        "A2": ToyNodeSpec(
            node_id="A2",
            player="single",
            terminal_value=8.0,
            state_tag="A2",
        ),
        "B": ToyNodeSpec(
            node_id="B",
            player="single",
            children={"B1": "B1", "B2": "B2"},
            state_tag="B",
        ),
        "B1": ToyNodeSpec(
            node_id="B1",
            player="single",
            terminal_value=5.0,
            state_tag="B1",
        ),
        "B2": ToyNodeSpec(
            node_id="B2",
            player="single",
            terminal_value=6.0,
            state_tag="B2",
        ),
    }
    return ToyScenarioSpec(
        name="single_agent_backup",
        root_id="root",
        nodes=nodes,
        description=(
            "Single-agent max backup where the best line should resolve to A -> A2."
        ),
        expected_root_value=8.0,
        expected_pv=("A", "A2"),
    )


def build_minimax_micro_scenario_spec() -> ToyScenarioSpec:
    """Return a tiny deterministic minimax scenario."""
    nodes = {
        "root": ToyNodeSpec(
            node_id="root",
            player="max",
            children={"A": "A", "B": "B"},
            state_tag="root",
        ),
        "A": ToyNodeSpec(
            node_id="A",
            player="min",
            children={"A1": "A1", "A2": "A2"},
            state_tag="A",
        ),
        "A1": ToyNodeSpec(
            node_id="A1",
            player="single",
            terminal_value=4.0,
            state_tag="A1",
        ),
        "A2": ToyNodeSpec(
            node_id="A2",
            player="single",
            terminal_value=7.0,
            state_tag="A2",
        ),
        "B": ToyNodeSpec(
            node_id="B",
            player="min",
            children={"B1": "B1", "B2": "B2"},
            state_tag="B",
        ),
        "B1": ToyNodeSpec(
            node_id="B1",
            player="single",
            terminal_value=2.0,
            state_tag="B1",
        ),
        "B2": ToyNodeSpec(
            node_id="B2",
            player="single",
            terminal_value=9.0,
            state_tag="B2",
        ),
    }
    return ToyScenarioSpec(
        name="minimax_micro",
        root_id="root",
        nodes=nodes,
        description=(
            "Tiny minimax tree where both children are MIN nodes and the root "
            "should prefer A -> A1."
        ),
        expected_root_value=4.0,
        expected_pv=("A", "A1"),
    )


def build_deceptive_trap_scenario_spec() -> ToyScenarioSpec:
    """Return a scenario whose best branch flips after deeper backup."""
    nodes = {
        "root": ToyNodeSpec(
            node_id="root",
            player="single",
            children={"A": "A", "B": "B"},
            state_tag="root",
        ),
        "A": ToyNodeSpec(
            node_id="A",
            player="single",
            children={"A1": "A1", "A2": "A2"},
            heuristic_value=9.0,
            state_tag="A",
        ),
        "A1": ToyNodeSpec(
            node_id="A1",
            player="single",
            terminal_value=1.0,
            state_tag="A1",
        ),
        "A2": ToyNodeSpec(
            node_id="A2",
            player="single",
            terminal_value=0.0,
            state_tag="A2",
        ),
        "B": ToyNodeSpec(
            node_id="B",
            player="single",
            children={"B1": "B1", "B2": "B2"},
            heuristic_value=5.0,
            state_tag="B",
        ),
        "B1": ToyNodeSpec(
            node_id="B1",
            player="single",
            terminal_value=6.0,
            state_tag="B1",
        ),
        "B2": ToyNodeSpec(
            node_id="B2",
            player="single",
            terminal_value=4.0,
            state_tag="B2",
        ),
    }
    return ToyScenarioSpec(
        name="deceptive_trap",
        root_id="root",
        nodes=nodes,
        description=(
            "Deceptive shallow heuristic where branch A starts attractive but "
            "deeper expansion reveals branch B as the final best line."
        ),
        expected_root_value=6.0,
        expected_pv=("B", "B1"),
    )


def all_toy_scenario_specs() -> tuple[ToyScenarioSpec, ...]:
    """Return all built-in toy scenario specifications."""
    return (
        build_single_agent_backup_scenario_spec(),
        build_minimax_micro_scenario_spec(),
        build_deceptive_trap_scenario_spec(),
    )


__all__ = [
    "all_toy_scenario_specs",
    "build_deceptive_trap_scenario_spec",
    "build_minimax_micro_scenario_spec",
    "build_single_agent_backup_scenario_spec",
]
