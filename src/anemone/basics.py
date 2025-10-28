from dataclasses import dataclass
from typing import Annotated, Protocol

from valanga import BoardEvaluation, BranchKey, Color, HasTurn, State

type Seed = Annotated[int, "seed"]
type TreeDepth = Annotated[int, "Depth level of a node in a tree structure"]


class StateWithTurn(State, HasTurn, Protocol):
    """Protocol for State that has turn information."""

    ...


@dataclass
class BranchRecommendation:
    """
    Represents a recommended move to play along with an optional evaluation score.
    """

    branch_key: BranchKey
    evaluation: BoardEvaluation | None = None


class HasBlackAndWhiteTurn(Protocol):
    """Protocol for content that has black and white turns."""

    @property
    def is_black_to_move(self) -> bool:
        """Returns True if it's black's turn to move, False otherwise."""
        ...

    def is_white_to_move(self) -> bool:
        """Returns True if it's white's turn to move, False otherwise."""
        ...

    def turn(self) -> Color:
        """Returns the color of the player whose turn it is."""
        ...
