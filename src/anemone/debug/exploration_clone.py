"""Shared shallow-clone helpers for debug exploration facades."""

from __future__ import annotations

from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any


def clone_exploration(tree_exploration: Any, **updates: Any) -> Any:
    """Return a shallow clone of ``tree_exploration`` with updated collaborators.

    This helper is intended for debug observation and assumes collaborator
    replacement is sufficient for safe delegated execution.
    """
    if is_dataclass(tree_exploration) and not isinstance(tree_exploration, type):
        return replace(tree_exploration, **updates)

    cloned_exploration = copy(tree_exploration)
    for attribute_name, value in updates.items():
        setattr(cloned_exploration, attribute_name, value)
    return cloned_exploration


__all__ = ["clone_exploration"]
