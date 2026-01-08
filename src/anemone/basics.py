from dataclasses import dataclass
from typing import Annotated, Mapping, Protocol

from valanga import BoardEvaluation, BranchKey, Color, HasTurn, State

from anemone.recommender_rule.recommender_rule import BranchPolicy

type Seed = Annotated[int, "seed"]
type TreeDepth = Annotated[int, "Depth level of a node in a tree structure"]


class StateWithTurn(State, HasTurn, Protocol):
    """A `valanga.State` that also exposes turn information."""

    ...


@dataclass(frozen=True, slots=True)
class BranchRecommendation:
    branch_key: BranchKey
    evaluation: BoardEvaluation | None = None
    policy: BranchPolicy | None = None
    child_evals: Mapping[BranchKey, BoardEvaluation] | None = None


class HasBlackAndWhiteTurn(Protocol):
    """Protocol for state that has black and white turns."""

    @property
    def turn(self) -> Color: ...
