"""Tree-evaluation families built on top of shared backup/runtime helpers.

This subpackage is where tree values, branch ordering, frontier tracking, and
principal-variation updates live. ``tree_value`` is the child/subtree-derived
value currently stored through ``backed_up_value``. Public factory entrypoints
are exposed from :mod:`anemone.node_evaluation.tree.factory`.
"""
