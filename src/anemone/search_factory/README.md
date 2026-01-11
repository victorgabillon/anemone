# Search factories

Search factories create coordinated components that must share the same
configuration:

- Node selectors (`node_selector/`).
- Exploration index data (`indices/`).
- Index update helpers (`updates/`).

`search_factory.py` exposes `SearchFactory` and `SearchFactoryP` to keep selector,
index creation, and index updates in sync.
