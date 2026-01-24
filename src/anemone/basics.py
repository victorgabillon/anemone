"""Basic types and protocols for Anemone."""

from typing import Annotated, Protocol

from valanga import Color, HasTurn, State

type Seed = Annotated[int, "seed"]
type TreeDepth = Annotated[int, "Depth level of a node in a tree structure"]


class StateWithTurn(State, HasTurn, Protocol):
    """A `valanga.State` that also exposes turn information."""

    ...


class HasBlackAndWhiteTurn(Protocol):
    """Protocol for state that has black and white turns."""

    @property
    def turn(self) -> Color:
        """Return the current player's turn color."""
        ...
