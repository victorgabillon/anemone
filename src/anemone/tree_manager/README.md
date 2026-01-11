# Tree manager

Tree managers coordinate how nodes are opened, expanded, and updated.

## Key pieces

- `tree_manager.py`: Base `TreeManager` for opening nodes and tracking
  expansions.
- `algorithm_node_tree_manager.py`: `AlgorithmNodeTreeManager` implementation
  for `AlgorithmNode` trees.
- `tree_expander.py`: `TreeExpansion` and `TreeExpansions` helpers describing
  node creation events.
- `factory.py`: `create_algorithm_node_tree_manager` convenience constructor.
