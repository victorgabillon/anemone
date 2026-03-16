"""Internal update helpers.

Value propagation now routes through ``anemone.updates.value_propagator``.
Descendant-depth propagation lives in
``anemone.updates.depth_index_propagator``.
Exploration-index refresh is orchestrated separately by tree-manager code.
"""

__all__: list[str] = []
