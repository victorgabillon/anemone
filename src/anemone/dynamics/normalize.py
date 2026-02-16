"""Utilities to normalize search dynamics inputs."""

from collections.abc import Hashable
from typing import cast

import valanga

from .search_dynamics import SearchDynamics
from .stateless_adapter import StatelessDynamicsAdapter

type DynamicsOrSearch[StateT: valanga.State, ActionT: Hashable] = (
    valanga.Dynamics[StateT] | SearchDynamics[StateT, ActionT]
)


def normalize_search_dynamics[StateT: valanga.State, ActionT: Hashable](
    dynamics: DynamicsOrSearch[StateT, ActionT],
) -> SearchDynamics[StateT, ActionT]:
    """Return a SearchDynamics regardless of the provided dynamics type."""
    if getattr(dynamics, "__anemone_search_dynamics__", False):
        return cast(SearchDynamics[StateT, ActionT], dynamics)
    return StatelessDynamicsAdapter(cast(valanga.Dynamics[StateT], dynamics))
