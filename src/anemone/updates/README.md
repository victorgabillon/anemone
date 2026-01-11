# Update pipeline

This folder hosts the logic for propagating updates after node expansions.

## Key pieces

- `algorithm_node_updater.py`: Updates algorithm nodes after expansions.
- `minmax_evaluation_updater.py`: Propagates minmax values up the tree.
- `index_updater.py`: Updates exploration indices when depth indexing is enabled.
- `updates_file.py` and `index_block.py`: Instruction objects describing which
  nodes/branches need updates.

These updates are triggered by `tree_manager` after new nodes are opened.
