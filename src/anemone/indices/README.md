# Exploration indices

Indices are optional structures used by node selectors to guide exploration.

## Key pieces

- `node_indices/`: Index data structures and factories (for example,
  depth-based indices).
- `index_manager/`: Manager for updating indices across the tree.

Index computation can be enabled via `SearchFactory` and is wired through
`AlgorithmNodeFactory` and `tree_manager`.
