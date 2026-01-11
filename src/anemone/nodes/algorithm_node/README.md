# Algorithm nodes

`AlgorithmNode` is the primary wrapper used by the tree-and-value search. It
wraps a `TreeNode` and adds:

- A `NodeTreeEvaluation` instance for minmax and best-branch tracking.
- Optional exploration index data used by node selectors.
- Optional state representations for evaluation (e.g., neural inputs).

See `algorithm_node.py` for the full wrapper interface.
