# anemone

`anemone` is a Python library for tree search.

## Design

This codebase follows a “core node + wrappers” pattern.

- **`TreeNode` (core)**
	- `TreeNode` is the canonical, shared data structure.
	- It stores the graph structure: `branches_children` and `parent_nodes`.
	- There is conceptually a single tree/graph of `TreeNode`s.

- **Wrappers implement `ITreeNode`**
	- Higher-level nodes (e.g. `AlgorithmNode`) wrap a `TreeNode` and add algorithm-specific state:
		evaluation, indices, representations, etc.
	- Wrappers expose navigation by delegating to the underlying `TreeNode`.

- **Homogeneity at the wrapper level**
	- Even though `TreeNode` is the core place where connections are stored, each wrapper is intended to be
		*closed under parent/child links*:
		- a wrapper’s `branches_children` and `parent_nodes` contain that same wrapper type.
		- today this is typically either “all `TreeNode`” or “all `AlgorithmNode`”.
		- in the future, another wrapper can exist (still implementing `ITreeNode`), and it should also be
			homogeneous within itself.

The practical motivation is:
- algorithms can be written against `ITreeNode` (for navigation) and against wrappers like `AlgorithmNode`
	(for algorithm-specific fields),
- while keeping a single shared underlying structure that can be accessed consistently from any wrapper.
