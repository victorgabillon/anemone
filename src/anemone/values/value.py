"""Value objects carrying score metadata for direct evaluation."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valanga import OverEvent


class Certainty(Enum):
    """Certainty levels associated with an evaluation score."""

    ESTIMATE = auto()
    TERMINAL = auto()
    FORCED = auto()


@dataclass(frozen=True, slots=True)
class Value:
    """Score plus certainty and optional terminal over-event metadata."""

    score: float
    certainty: Certainty = Certainty.ESTIMATE
    over_event: "OverEvent | None" = None
