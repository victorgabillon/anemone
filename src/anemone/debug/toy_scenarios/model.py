"""Declarative model types for tiny debug-oriented toy search scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def _empty_children() -> dict[str, str]:
    """Return an empty action-to-child mapping."""
    return {}


type ToyPlayerRole = Literal["single", "max", "min"]


@dataclass(frozen=True)
class ToyNodeSpec:
    """Declarative description of one node in a toy scenario."""

    node_id: str
    player: ToyPlayerRole
    children: dict[str, str] = field(default_factory=_empty_children)
    terminal_value: float | None = None
    heuristic_value: float | None = None
    state_tag: str | None = None


@dataclass(frozen=True)
class ToyScenarioSpec:
    """Declarative description of one full toy exploration scenario."""

    name: str
    root_id: str
    nodes: dict[str, ToyNodeSpec]
    description: str
    expected_root_value: float | None = None
    expected_pv: tuple[str, ...] = ()


__all__ = ["ToyNodeSpec", "ToyPlayerRole", "ToyScenarioSpec"]
