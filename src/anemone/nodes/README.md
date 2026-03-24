# Nodes

This package defines the node-layer vocabulary used across the search runtime.

- `ITreeNode`: structural/navigation protocol
- `TreeNode`: concrete structural implementation
- `AlgorithmNode`: runtime/search wrapper in `anemone.nodes.algorithm_node`

`TreeNode` owns structure only. `AlgorithmNode` adds tree evaluation,
exploration-index payloads, and optional evaluator representations on top of
that structure.

For the end-to-end runtime flow, see
[`docs/source/search_iteration_architecture.rst`](../../../docs/source/search_iteration_architecture.rst).
