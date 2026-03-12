"""Tests for live debug control helpers."""

# ruff: noqa: D103

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from anemone.debug import (
    BackupFinished,
    BackupFlagBreakpoint,
    ControlledTreeExploration,
    DebugCommand,
    DebugSessionController,
    EventTypeBreakpoint,
    LiveDebugSessionRecorder,
    NodeSelected,
    write_debug_command,
)
from anemone.node_selector.opening_instructions import OpeningInstructions

if TYPE_CHECKING:
    from pathlib import Path


class _FakeStoppingCriterion:
    def __init__(self) -> None:
        self.checks = 0

    def should_we_continue(self, *, tree: Any) -> bool:
        del tree
        self.checks += 1
        return self.checks == 1


class _FakeTreeManager:
    def __init__(self) -> None:
        self.update_calls = 0
        self.dynamics: Any = None

    def update_indices(self, *, tree: Any) -> None:
        del tree
        self.update_calls += 1


@dataclass
class _FakeExploration:
    tree: Any
    tree_manager: Any
    stopping_criterion: Any
    node_selector: Any | None = None
    explore_calls: list[Random] = field(default_factory=list)
    result: Any = field(default_factory=lambda: SimpleNamespace(marker="done"))

    def explore(self, random_generator: Random) -> Any:
        self.explore_calls.append(random_generator)
        while self.stopping_criterion.should_we_continue(tree=self.tree):
            if self.node_selector is not None:
                self.node_selector.choose_node_and_branch_to_open(
                    tree=self.tree,
                    latest_tree_expansions=SimpleNamespace(),
                )
            self.tree_manager.update_indices(tree=self.tree)
        return self.result


@dataclass(eq=False)
class _FakeNode:
    id: int
    state: Any
    tree_depth: int
    branches_children: dict[str, Any | None] = field(default_factory=dict)
    parent_nodes: dict[Any, str] = field(default_factory=dict)
    all_branches_generated: bool = False


class _FakeLegalActions:
    def __init__(self, branches: list[str]) -> None:
        self._branches = branches

    def get_all(self) -> list[str]:
        return list(self._branches)


class _FakeDynamics:
    def __init__(self, branches_by_state: dict[str, list[str]]) -> None:
        self._branches_by_state = branches_by_state

    def legal_actions(self, state: str) -> _FakeLegalActions:
        return _FakeLegalActions(self._branches_by_state[state])


class _FakeNodeSelector:
    def __init__(self) -> None:
        self.calls = 0

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: Any,
    ) -> OpeningInstructions[Any]:
        del tree, latest_tree_expansions
        self.calls += 1
        return OpeningInstructions()


def test_debug_session_controller_initializes_command_file(tmp_path: Path) -> None:
    controller = DebugSessionController(tmp_path)

    assert controller.poll() == DebugCommand.NONE
    payload = json.loads((tmp_path / "commands.json").read_text(encoding="utf-8"))
    assert payload["command"] == "none"
    assert (tmp_path / "breakpoints.json").exists()
    assert (tmp_path / "control_state.json").exists()


def test_debug_session_controller_applies_pause_and_resume_commands(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)

    write_debug_command(tmp_path, DebugCommand.PAUSE, timestamp=1)
    assert controller.poll() == DebugCommand.PAUSE
    assert controller.should_pause() is True

    write_debug_command(tmp_path, DebugCommand.RESUME, timestamp=2)
    assert controller.poll() == DebugCommand.RESUME
    assert controller.should_pause() is False


def test_debug_session_controller_consumes_one_step_then_repauses(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)

    write_debug_command(tmp_path, DebugCommand.STEP, timestamp=1)
    assert controller.poll() == DebugCommand.STEP
    assert controller.should_step() is True
    assert controller.should_pause() is False

    controller.complete_iteration()

    assert controller.should_step() is False
    assert controller.should_pause() is True


def test_debug_session_controller_runs_until_matching_node_event(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)

    controller.run_until_node_event("7")
    hit = controller.handle_event(NodeSelected(node_id="7"))

    assert hit is not None
    assert hit.id == "node-event:7"
    assert controller.should_pause() is True
    control_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert control_state["last_breakpoint_hit"] == "node-event:7"
    assert control_state["run_until_node_event_node_id"] is None


def test_debug_session_controller_runs_until_value_change_for_node(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)

    controller.run_until_node_value_change("11")
    hit = controller.handle_event(
        BackupFinished(
            node_id="11",
            value_changed=True,
            pv_changed=False,
            over_changed=False,
        )
    )

    assert hit is not None
    assert hit.id == "node-value-change:11"
    assert controller.should_pause() is True
    control_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert control_state["run_until_node_value_change_node_id"] is None


def test_debug_session_controller_persists_timeline_focus(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)

    controller.focus_node_timeline("17")
    focused_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert focused_state["focused_node_id"] == "17"

    controller.clear_timeline_focus()
    cleared_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert cleared_state["focused_node_id"] is None


def test_debug_session_controller_saves_and_loads_breakpoints(tmp_path: Path) -> None:
    controller = DebugSessionController(tmp_path)
    breakpoints = (
        EventTypeBreakpoint(id="bp-1", enabled=True, event_type="NodeSelected"),
        BackupFlagBreakpoint(id="bp-2", enabled=True, flag_name="pv_changed"),
    )

    controller.save_breakpoints(breakpoints)

    assert controller.load_breakpoints() == breakpoints


def test_debug_session_controller_auto_pauses_on_matching_breakpoint(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)
    controller.save_breakpoints(
        (
            BackupFlagBreakpoint(
                id="bp-3",
                enabled=True,
                flag_name="pv_changed",
            ),
        )
    )

    hit = controller.handle_event(
        BackupFinished(
            node_id="11",
            value_changed=False,
            pv_changed=True,
            over_changed=False,
        )
    )

    assert hit is not None
    assert hit.id == "bp-3"
    assert controller.should_pause() is True
    control_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert control_state["paused"] is True
    assert control_state["last_breakpoint_hit"] == "bp-3"


def test_debug_session_controller_does_not_pause_on_non_matching_event(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)
    controller.save_breakpoints(
        (EventTypeBreakpoint(id="bp-1", enabled=True, event_type="NodeSelected"),)
    )

    hit = controller.handle_event(
        BackupFinished(
            node_id="11",
            value_changed=True,
            pv_changed=False,
            over_changed=False,
        )
    )

    assert hit is None
    assert controller.should_pause() is False


def test_controlled_tree_exploration_runs_exactly_one_iteration_for_step(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)
    write_debug_command(tmp_path, DebugCommand.STEP, timestamp=1)

    tree_manager = _FakeTreeManager()
    stopping_criterion = _FakeStoppingCriterion()
    base_exploration = _FakeExploration(
        tree=SimpleNamespace(root_node=SimpleNamespace()),
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
    )
    exploration = ControlledTreeExploration.from_tree_exploration(
        base_exploration,
        controller=controller,
    )

    result = exploration.explore(random_generator=Random(0))

    assert result.marker == "done"
    assert tree_manager.update_calls == 1
    assert stopping_criterion.checks == 2
    assert controller.should_step() is False
    assert controller.should_pause() is True


def test_controlled_tree_exploration_forces_best_effort_node_expansion(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)
    root = _FakeNode(id=1, state="root", tree_depth=0, branches_children={"a": None})
    tree_manager = _FakeTreeManager()
    tree_manager.dynamics = _FakeDynamics({"root": ["a", "b"]})
    stopping_criterion = _FakeStoppingCriterion()
    node_selector = _FakeNodeSelector()
    base_exploration = _FakeExploration(
        tree=SimpleNamespace(root_node=root),
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
        node_selector=node_selector,
    )
    controlled_exploration = ControlledTreeExploration.from_tree_exploration(
        base_exploration,
        controller=controller,
    )

    controller.expand_node("1")
    opening_instructions = (
        controlled_exploration.node_selector.choose_node_and_branch_to_open(
            tree=base_exploration.tree,
            latest_tree_expansions=SimpleNamespace(),
        )
    )

    assert node_selector.calls == 0
    assert [
        opening_instruction.branch
        for opening_instruction in opening_instructions.values()
    ] == ["a"]
    control_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert control_state["requested_expansion_node_id"] is None


def test_live_debug_session_recorder_notifies_controller_about_breakpoints(
    tmp_path: Path,
) -> None:
    controller = DebugSessionController(tmp_path)
    controller.save_breakpoints(
        (EventTypeBreakpoint(id="bp-7", enabled=True, event_type="NodeSelected"),)
    )
    recorder = LiveDebugSessionRecorder(tmp_path, controller=controller)

    recorder.emit(NodeSelected(node_id="7"))

    assert controller.should_pause() is True
    control_state = json.loads(
        (tmp_path / "control_state.json").read_text(encoding="utf-8")
    )
    assert control_state["last_breakpoint_hit"] == "bp-7"
    session_payload = json.loads(
        (tmp_path / "session.json").read_text(encoding="utf-8")
    )
    assert session_payload["entries"][0]["breakpoint_hit"] == "bp-7"
