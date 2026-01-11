# Node indices

Node index data structures describe per-node exploration signals used by
selection strategies.

## Key pieces

- `index_data.py`: `NodeExplorationData` implementations (including
  `MaxDepthDescendants`).
- `index_types.py`: `IndexComputationType` enum that selects index strategies.
- `factory.py`: `ExplorationIndexDataFactory` and helpers to instantiate index
  data for new nodes.
