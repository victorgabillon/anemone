from collections.abc import Hashable
from typing import Annotated, Iterator, Protocol, Self

type NodeTag = Annotated[Hashable, "A label or identifier for a node in a tree"]
type ContentTag = Annotated[Hashable, "A label or identifier for a node in a tree"]


type TreeDepth = Annotated[int, "Depth level of a node in a tree structure"]


type BranchKey = Annotated[Hashable, "A label or identifier for a branch in a tree"]


type ContentRepresentation = Annotated[
    Hashable, "A representation of the content of a node"
]


class BranchKeyGeneratorP(Protocol):
    """Protocol for a branch key generator that yields branch keys."""

    all_generated_keys: list[BranchKey] | None

    # whether to sort the branch keys by their respective uci for easy comparison of various implementations
    sort_branch_keys: bool = False

    def __iter__(self) -> Iterator[BranchKey]:
        """Returns an iterator over the branch keys."""
        ...

    def __next__(self) -> BranchKey:
        """Returns the next branch key."""
        ...

    def more_than_one_branch(self) -> bool:
        """Checks if there is more than one branch available.

        Returns:
            bool: True if there is more than one branch, False otherwise.
        """
        ...

    def get_all(self) -> list[BranchKey]:
        """Returns a list of all branch keys."""
        ...

    def copy_with_reset(self) -> Self:
        """Creates a copy of the legal move generator with an optional reset of generated moves.

        Returns:
            Self: A new instance of the legal move generator with the specified generated moves.
        """
        ...


class NodeContent(Protocol):
    """Protocol for a content object that has a tag."""

    @property
    def tag(self) -> ContentTag:
        """Returns the tag of the content.

        Returns:
            ContentTag: The tag of the content.
        """
        ...

    @property
    def branch_keys(self) -> BranchKeyGeneratorP:
        """Returns the branch keys associated with the content.

        Returns:
            BranchKeyGeneratorP: The branch keys associated with the content.
        """
        ...
