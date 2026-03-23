"""Public entrypoints for assembling and running Anemone searches.

Preferred public vocabulary:

* ``SearchArgs`` configures a search
* ``create_search(...)`` builds the runnable runtime
* ``SearchRuntime`` exposes ``step()`` and ``explore(...)``

``SearchRecommender`` remains available as the secondary recommend-only API.
Legacy ``TreeAndValue...`` names remain available for compatibility.
"""

from .factory import (
    SearchArgs,
    TreeAndValuePlayerArgs,
    create_search,
    create_search_with_tree_eval_factory,
    create_tree_and_value_branch_selector,
    create_tree_and_value_branch_selector_with_tree_eval_factory,
    create_tree_and_value_exploration,
    create_tree_and_value_exploration_with_tree_eval_factory,
)
from .tree_and_value_branch_selector import SearchRecommender
from .tree_exploration import SearchRuntime

__all__ = [
    "SearchArgs",
    "SearchRecommender",
    "SearchRuntime",
    "TreeAndValuePlayerArgs",
    "create_search",
    "create_search_with_tree_eval_factory",
    "create_tree_and_value_branch_selector",
    "create_tree_and_value_branch_selector_with_tree_eval_factory",
    "create_tree_and_value_exploration",
    "create_tree_and_value_exploration_with_tree_eval_factory",
]
