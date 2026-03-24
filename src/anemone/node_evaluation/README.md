# Node evaluation

This package owns Anemone's value semantics and tree-evaluation state.

## Key concepts

- `direct_value`: immediate evaluator output for one node
- `backed_up_value`: subtree-derived value propagated from children
- candidate value: best currently available value, preferring backed-up over
  direct when both exist
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
