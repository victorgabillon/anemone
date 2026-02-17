"""Utilities to normalize search dynamics inputs."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast, overload

import valanga

from .search_dynamics import SearchDynamics
from .stateless_adapter import StatelessDynamicsAdapter

type DynamicsOrSearch[S: valanga.State, A: Hashable] = (
    valanga.Dynamics[S] | SearchDynamics[S, A]
)


@overload
def normalize_search_dynamics[StateT: valanga.State, ActionT: Hashable](
    dynamics: SearchDynamics[StateT, ActionT],
) -> SearchDynamics[StateT, ActionT]: ...


@overload
def normalize_search_dynamics[StateT: valanga.State](
    dynamics: valanga.Dynamics[StateT],
) -> SearchDynamics[StateT, valanga.BranchKey]: ...


def normalize_search_dynamics(dynamics: Any) -> Any:
    """Return a SearchDynamics regardless of the provided dynamics type."""
    if getattr(dynamics, "__anemone_search_dynamics__", False):
        return dynamics
    # When it is plain valanga.Dynamics, the action key type is valanga.BranchKey.
    return cast(
        "SearchDynamics[Any, valanga.BranchKey]",
        StatelessDynamicsAdapter(cast("valanga.Dynamics[Any]", dynamics)),
    )
