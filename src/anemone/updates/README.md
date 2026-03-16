# Update pipeline

This folder hosts the remaining helpers for propagating updates after node
expansions.

## Key pieces

- `value_propagator.py`: depth-batched dirty-ancestor propagation for value-only
  updates.
- `depth_index_propagator.py`: depth-batched dirty-ancestor propagation for
  `MaxDepthDescendants` metadata when depth indexing is enabled.

The active architecture is:

- structural expansion
- direct evaluation
- value propagation via `ValuePropagator`
- descendant-depth propagation via `DepthIndexPropagator` when needed
- exploration-index refresh as a separate explicit phase in `tree_manager`
