"""Tests for debug breakpoint models and matchers."""

# ruff: noqa: D103

from __future__ import annotations

from anemone.debug import (
    BackupFinished,
    BackupFlagBreakpoint,
    ChildLinked,
    EventTypeBreakpoint,
    IterationBreakpoint,
    NodeIdBreakpoint,
    NodeSelected,
    SearchIterationStarted,
    any_breakpoint_matches,
    breakpoint_from_json,
    breakpoint_matches,
    breakpoint_to_json,
)


def test_event_type_breakpoint_matches_correct_event() -> None:
    breakpoint_spec = EventTypeBreakpoint(
        id="bp-1",
        enabled=True,
        event_type="NodeSelected",
    )

    assert breakpoint_matches(breakpoint_spec, NodeSelected(node_id="7")) is True
    assert (
        breakpoint_matches(
            breakpoint_spec,
            SearchIterationStarted(iteration_index=7),
        )
        is False
    )


def test_node_id_breakpoint_matches_supported_events_only() -> None:
    breakpoint_spec = NodeIdBreakpoint(id="bp-2", enabled=True, node_id="42")

    assert breakpoint_matches(breakpoint_spec, NodeSelected(node_id="42")) is True
    assert breakpoint_matches(breakpoint_spec, NodeSelected(node_id="7")) is False
    assert (
        breakpoint_matches(
            breakpoint_spec,
            ChildLinked(
                parent_id="42",
                child_id="99",
                branch_key="a",
                was_already_present=False,
            ),
        )
        is False
    )


def test_backup_flag_breakpoint_matches_only_backup_finished() -> None:
    breakpoint_spec = BackupFlagBreakpoint(
        id="bp-3",
        enabled=True,
        flag_name="pv_changed",
    )

    assert (
        breakpoint_matches(
            breakpoint_spec,
            BackupFinished(
                node_id="17",
                value_changed=False,
                pv_changed=True,
                over_changed=False,
            ),
        )
        is True
    )
    assert (
        breakpoint_matches(
            breakpoint_spec,
            BackupFinished(
                node_id="17",
                value_changed=True,
                pv_changed=False,
                over_changed=False,
            ),
        )
        is False
    )


def test_iteration_breakpoint_matches_exact_iteration() -> None:
    breakpoint_spec = IterationBreakpoint(
        id="bp-4",
        enabled=True,
        iteration_index=17,
    )

    assert (
        breakpoint_matches(
            breakpoint_spec,
            SearchIterationStarted(iteration_index=17),
        )
        is True
    )
    assert (
        breakpoint_matches(
            breakpoint_spec,
            SearchIterationStarted(iteration_index=18),
        )
        is False
    )


def test_breakpoint_json_round_trip() -> None:
    original = BackupFlagBreakpoint(
        id="bp-5",
        enabled=True,
        flag_name="value_changed",
    )

    payload = breakpoint_to_json(original)
    restored = breakpoint_from_json(payload)

    assert restored == original


def test_any_breakpoint_matches_returns_first_enabled_match() -> None:
    breakpoints = (
        EventTypeBreakpoint(id="bp-1", enabled=False, event_type="NodeSelected"),
        NodeIdBreakpoint(id="bp-2", enabled=True, node_id="9"),
        EventTypeBreakpoint(id="bp-3", enabled=True, event_type="NodeSelected"),
    )

    assert (
        any_breakpoint_matches(breakpoints, NodeSelected(node_id="9")) == breakpoints[1]
    )
