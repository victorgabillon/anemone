"""Source-aware value candidates for node-evaluation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Keep Value in module globals so runtime introspection via get_type_hints works.
from valanga.evaluations import Value  # noqa: TC002


class ValueCandidateSource(StrEnum):
    """Provenance for a value candidate returned by a node evaluation."""

    NONE = "none"
    DIRECT_SELF = "direct_self"
    TREE_CHILD = "tree_child"


@dataclass(frozen=True, slots=True)
class ValueCandidate:
    """A maybe-present ``Value`` plus the source it came from."""

    value: Value | None
    source: ValueCandidateSource

    @classmethod
    def none(cls) -> ValueCandidate:
        """Return an absent value candidate."""
        return cls(value=None, source=ValueCandidateSource.NONE)

    @classmethod
    def direct(cls, value: Value) -> ValueCandidate:
        """Return a candidate sourced from this node's direct evaluator."""
        return cls(value=value, source=ValueCandidateSource.DIRECT_SELF)

    @classmethod
    def tree(cls, value: Value) -> ValueCandidate:
        """Return a candidate sourced from child/subtree backup."""
        return cls(value=value, source=ValueCandidateSource.TREE_CHILD)

    @property
    def has_value(self) -> bool:
        """Return whether this candidate carries a concrete value."""
        return self.value is not None
