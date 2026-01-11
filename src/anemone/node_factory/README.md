# Node factories

Factories are responsible for constructing tree nodes and their algorithm-specific
wrappers.

## Key pieces

- `base.py`: `TreeNodeFactory` creates the core `TreeNode` instances.
- `algorithm_node_factory.py`: `AlgorithmNodeFactory` wraps `TreeNode` objects
  with tree evaluation, exploration indices, and optional state representations.

Factories are composed in `factory.py` to build the full tree-and-value
pipeline.
