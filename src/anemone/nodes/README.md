# Nodes

This package defines the core tree data structures.

## Key pieces

- `tree_node.py`: `TreeNode` stores the shared graph structure, including parent
  links, child branches, and the associated `valanga` state.
- `itree_node.py`: `ITreeNode` protocol used by wrappers to expose navigation
  consistently.
- `algorithm_node/`: Higher-level wrappers that attach evaluation and exploration
  data to a `TreeNode`.
