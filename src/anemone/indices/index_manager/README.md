# Index manager

The index manager coordinates recalculating exploration indices after tree
expansions.

- `node_exploration_manager.py` updates indices on nodes across the tree.
- `factory.py` exposes `create_exploration_index_manager` for wiring into the
  broader search pipeline.
