# Node selection

Node selectors choose which node and branch to open next during tree exploration.

## Key pieces

- `factory.py`: `create()` builds a selector from configuration arguments.
- `opening_instructions.py`: Defines how branches are opened (and which
  branches to open).
- `node_selector.py` / `node_selector_args.py`: Base classes and argument types.
- `node_selector_types.py`: Enumerates supported selector types.

Concrete strategies are in subfolders:

- `uniform/` for uniform selection.
- `recurzipf/` for RecurZipf-based sampling.
- `sequool/` for Sequool-style depth selection.
