# Search factories

Search factories create coordinated components that must share the same
configuration:

- Node selectors (`node_selector/`).
- Exploration index data (`indices/`).

`search_factory.py` exposes `SearchFactory` and `SearchFactoryP` to keep selector,
index creation, and search configuration in sync.
