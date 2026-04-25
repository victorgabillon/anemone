"""Tests for opt-in descendants/checkpoint invariant diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from anemone.trees.debug_invariants import (
    DESCENDANTS_INVARIANT_ENV_VAR,
    DESCENDANTS_INVARIANT_LOG_PREFIX,
    validate_descendants_tags,
)
from anemone.utils.logger import anemone_logger


@dataclass(slots=True)
class _FakeState:
    tag: str


@dataclass
class _FakeNode:
    id: int
    tag: str
    state: _FakeState
    tree_depth: int


class _FakeDescendants:
    def __init__(self, nodes_by_depth: dict[int, dict[str, _FakeNode]]) -> None:
        self._nodes_by_depth = nodes_by_depth

    def range(self) -> range:
        return range(min(self._nodes_by_depth), max(self._nodes_by_depth) + 1)

    def __getitem__(self, tree_depth: int) -> dict[str, _FakeNode]:
        return self._nodes_by_depth[tree_depth]


@dataclass(slots=True)
class _FakeTree:
    descendants: _FakeDescendants


def _tree(nodes_by_depth: dict[int, dict[str, _FakeNode]]) -> _FakeTree:
    return _FakeTree(descendants=_FakeDescendants(nodes_by_depth))


def test_validate_descendants_tags_passes_when_tags_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching stored, node, and state tags should pass in debug mode."""
    monkeypatch.setenv(DESCENDANTS_INVARIANT_ENV_VAR, "1")
    node = _FakeNode(id=1, tag="a", state=_FakeState(tag="a"), tree_depth=0)

    validate_descendants_tags(_tree({0: {"a": node}}), phase="test")  # type: ignore[arg-type]


def test_validate_descendants_tags_fails_when_stored_tag_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale descendants key should fail loudly in debug mode."""
    monkeypatch.setenv(DESCENDANTS_INVARIANT_ENV_VAR, "1")
    node = _FakeNode(id=1, tag="new", state=_FakeState(tag="new"), tree_depth=0)

    with pytest.raises(AssertionError, match="descendants invariant failed"):
        validate_descendants_tags(_tree({0: {"old": node}}), phase="test")  # type: ignore[arg-type]


def test_validate_descendants_tags_fails_on_duplicate_state_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two nodes at one depth with one state tag should fail even if keys differ."""
    monkeypatch.setenv(DESCENDANTS_INVARIANT_ENV_VAR, "1")
    first_node = _FakeNode(id=1, tag="a", state=_FakeState(tag="same"), tree_depth=0)
    second_node = _FakeNode(id=2, tag="b", state=_FakeState(tag="same"), tree_depth=0)

    with pytest.raises(AssertionError, match="descendants invariant failed"):
        validate_descendants_tags(
            _tree({0: {"a": first_node, "b": second_node}}),  # type: ignore[arg-type]
            phase="test",
        )


def test_validate_descendants_tags_is_noop_without_env_flag(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env flag, validation should not inspect or log anything."""
    monkeypatch.delenv(DESCENDANTS_INVARIANT_ENV_VAR, raising=False)
    node = _FakeNode(id=1, tag="new", state=_FakeState(tag="old"), tree_depth=1)

    validate_descendants_tags(_tree({0: {"stale": node}}), phase="test")  # type: ignore[arg-type]

    assert DESCENDANTS_INVARIANT_LOG_PREFIX not in caplog.text


def test_validate_descendants_tags_logs_registered_snapshot_for_mutated_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutated node tag should log the insertion-time debug snapshot."""
    monkeypatch.setenv(DESCENDANTS_INVARIANT_ENV_VAR, "1")
    node = _FakeNode(id=1, tag="current", state=_FakeState(tag="current"), tree_depth=0)
    node._debug_registered_tag = "registered"
    node._debug_registered_state_tag = "registered_state"
    node._debug_registered_depth = 0
    node._debug_registered_node_id = 1
    node._debug_registered_parent_node_id = 99
    error_messages: list[str] = []

    def capture_error(message: str, *args: Any) -> None:
        error_messages.append(message % args)

    monkeypatch.setattr(anemone_logger, "error", capture_error)

    with pytest.raises(AssertionError, match="descendants invariant failed"):
        validate_descendants_tags(_tree({0: {"registered": node}}), phase="test")  # type: ignore[arg-type]

    assert any(
        "first_seen_mismatch=stored_tag!=node.tag" in msg for msg in error_messages
    )
    assert any("registered_parent_node_id=99" in msg for msg in error_messages)
