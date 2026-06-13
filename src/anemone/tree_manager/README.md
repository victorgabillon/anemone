# Tree manager

Tree managers coordinate how nodes are opened, expanded, and updated.

Nodes can be unopened, partially opened, or fully opened. A node having children
does not imply that all legal branches are open. `all_branches_generated` means
no legal branch remains unopened, and opening all children opens only the
remaining legal branches.

## Key pieces

- `tree_manager.py`: Base `TreeManager` for opening nodes and tracking
  expansions.
- `branch_opening_service.py`: one-branch opening primitive that records
  `TreeExpansion` objects and runs per-branch callbacks such as branch-frontier
  bookkeeping.
- `opening_expansion_executor.py`: one-ply `OpeningInstructions` executor that
  opens requested branches and synchronizes touched parents once per batch.
- `algorithm_node_tree_manager.py`: `AlgorithmNodeTreeManager` implementation
  for `AlgorithmNode` trees.
- `tree_expander.py`: `TreeExpansion` and `TreeExpansions` helpers describing
  node creation events.
- `factory.py`: `create_algorithm_node_tree_manager` convenience constructor.

This split keeps `TreeManager` as the structural single-branch open/link
primitive while allowing future expansion executors, such as rollout expansion,
to reuse the same branch-opening path.
