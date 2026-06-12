"""Tree-evaluation families built on top of shared backup/runtime helpers.

This subpackage is where tree values, branch ordering, frontier tracking, and
principal-variation updates live. ``backed_up_value`` remains the legacy
internal name for a child/subtree-derived ``tree_value``. Public factory
entrypoints are exposed from :mod:`anemone.node_evaluation.tree.factory`.
"""
