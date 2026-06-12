# Node evaluation

This package owns Anemone's value semantics and tree-evaluation state.

## Key concepts

- `direct_value`: immediate evaluator output for one node
- `tree_value`: child/subtree-derived value propagated from children
- `backed_up_value`: current storage property for `tree_value`
- effective value: search-facing value; direct when no tree value exists,
  objective-best of direct and tree for partially opened nodes, and tree value
  for fully opened nodes
- candidate value: a maybe-present value, optionally with source provenance
- canonical value: the required concrete `Value` returned to consumers that
  need one

## Structure

- `common/`: shared value semantics and protocols
- `direct/`: immediate evaluation of individual nodes
- `tree/`: backup, branch ordering, branch frontier, and principal variation

## Out of scope

- no exploration-index strategy logic
- no selector/opening policy logic

For the end-to-end runtime flow, see
[`docs/source/search_iteration_architecture.rst`](../../../docs/source/search_iteration_architecture.rst).
