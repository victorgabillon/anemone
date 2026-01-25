"""Basic types and protocols for Anemone."""

from typing import Annotated

type Seed = Annotated[int, "seed"]
type TreeDepth = Annotated[int, "Depth level of a node in a tree structure"]
