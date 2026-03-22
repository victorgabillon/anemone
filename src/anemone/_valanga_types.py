"""Local aliases for strict type-checking against Valanga generics."""

from typing import Any

from valanga import OverEvent, TurnState

type AnyOverEvent = OverEvent[Any]
type AnyTurnState = TurnState[Any]
