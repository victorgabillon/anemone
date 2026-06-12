# Tree manager

Tree managers coordinate how nodes are opened, expanded, and updated.

Nodes can be unopened, partially opened, or fully opened. A node having children
does not imply that all legal branches are open. `all_branches_generated` means
no legal branch remains unopened, and opening all children opens only the
remaining legal branches.

## Key pieces

- `tree_manager.py`: Base `TreeManager` for opening nodes and tracking
  expansions.
- `algorithm_node_tree_manager.py`: `AlgorithmNodeTreeManager` implementation
  for `AlgorithmNode` trees.
- `tree_expander.py`: `TreeExpansion` and `TreeExpansions` helpers describing
  node creation events.
- `factory.py`: `create_algorithm_node_tree_manager` convenience constructor.
