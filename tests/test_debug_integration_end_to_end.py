"""End-to-end integration coverage for the public debug GUI setup path."""

# ruff: noqa: D103

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from anemone.debug import build_live_debug_environment

if TYPE_CHECKING:
    from pathlib import Path


def _empty_object_list() -> list[Any]:
    return []


def _empty_branches_children() -> dict[str, Any | None]:
    return {}


def _empty_parent_nodes() -> dict[Any, str]:
    return {}


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
    explore_calls: list[Any] = field(default_factory=_empty_object_list)
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
    branches_children: dict[str, Any | None] = field(
        default_factory=_empty_branches_children
    )
    parent_nodes: dict[Any, str] = field(default_factory=_empty_parent_nodes)


def test_build_live_debug_environment_happy_path(tmp_path: Path) -> None:
    session_directory = tmp_path / "debug-session"
    exploration = _FakeExploration(
        tree=SimpleNamespace(root_node=_FakeNode(id=1, state="root", tree_depth=0)),
        tree_manager=_FakeTreeManager(),
        stopping_criterion=_FakeStoppingCriterion(),
    )

    environment = build_live_debug_environment(
        tree_exploration=exploration,
        session_directory=session_directory,
        snapshot_format="dot",
    )
    result = environment.controlled_exploration.explore(random_generator=Random(0))
    environment.finalize()

    payload = json.loads(
        (session_directory / "session.json").read_text(encoding="utf-8")
    )

    assert result.marker == "done"
    assert (session_directory / "index.html").exists()
    assert (session_directory / "snapshots").is_dir()
    assert payload["is_complete"] is True
    assert payload["entry_count"] >= 1


def test_public_live_debug_api_shape(tmp_path: Path) -> None:
    exploration = _FakeExploration(
        tree=SimpleNamespace(root_node=_FakeNode(id=1, state="root", tree_depth=0)),
        tree_manager=_FakeTreeManager(),
        stopping_criterion=_FakeStoppingCriterion(),
    )

    environment = build_live_debug_environment(
        tree_exploration=exploration,
        session_directory=tmp_path / "api-shape",
    )

    assert hasattr(environment, "controlled_exploration")
    assert hasattr(environment, "controller")
    assert hasattr(environment, "recorder")
    assert hasattr(environment, "session_directory")
    assert callable(environment.finalize)
