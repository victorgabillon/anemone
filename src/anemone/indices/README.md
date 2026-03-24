# Exploration indices

This package defines and updates exploration indices used to prioritize search.

## Key concepts

- exploration-index payload: the data object stored on each node
- strategy: the rule used to compute or update that payload
- manager: the tree-wide coordinator that recomputes indices across the tree

## Structure

- `node_indices/`: per-node payload classes and payload factories
- `index_manager/`: strategy selection plus tree-wide recomputation

## Current strategy families

- recursive Zipf / factored probability
- interval / local-min-change
- global-min-change

Important: exploration indices are stored per node, but the current update flow
recomputes them tree-wide from the root using the configured strategy.

For the end-to-end runtime flow, see
[`docs/source/search_iteration_architecture.rst`](../../../docs/source/search_iteration_architecture.rst).
