"""Utilities to normalize search dynamics inputs."""

from collections.abc import Hashable
from typing import cast

import valanga

from .search_dynamics import SearchDynamics
from .stateless_adapter import StatelessDynamicsAdapter

type DynamicsOrSearch[StateT: valanga.State, ActionT: Hashable] = (
    valanga.Dynamics[StateT] | SearchDynamics[StateT, ActionT]
)


def normalize_search_dynamics[S: valanga.State, A: Hashable](
    dynamics: DynamicsOrSearch[S, A],
) -> SearchDynamics[S, A]:
    """Return a SearchDynamics regardless of the provided dynamics type."""
    if getattr(dynamics, "__anemone_search_dynamics__", False):
        return cast("SearchDynamics[S, A]", dynamics)
    return StatelessDynamicsAdapter(cast("valanga.Dynamics[S]", dynamics))
