# Search factory

This package is an internal assembly layer.

- `SearchFactory` keeps selector creation and exploration-index payload creation
  coherent for one runtime configuration.
- `_runtime_assembly` uses it while building the runnable search runtime.
- `anemone.factory` remains the public entry point for callers.

If you are new to the codebase, start from `anemone.factory`, not here.

For the end-to-end runtime flow, see
[`docs/source/search_iteration_architecture.rst`](../../../docs/source/search_iteration_architecture.rst).
