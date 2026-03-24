# Node selector

This package decides which node or branch should be opened next during search.

## Key concepts

- selector: chooses where the search goes next
- opening instructions: the concrete node/branch openings returned by a selector
- opening instructor: turns "open this node" into the actual branches to open

## What selectors read and return

- selectors read evaluation state and, depending on strategy, may also read
  branch-frontier or exploration-index state
- selectors return `OpeningInstructions` for the structural expansion phase

## Composition

- `composed/`: optional priority override plus base selector
- `uniform/`, `recurzipf/`, `sequool/`: concrete strategy families

## Out of scope

- selectors do not evaluate nodes
- selectors do not back up values
- selectors do not recompute exploration indices

For the end-to-end runtime flow, see
[`docs/source/search_iteration_architecture.rst`](../../../docs/source/search_iteration_architecture.rst).
