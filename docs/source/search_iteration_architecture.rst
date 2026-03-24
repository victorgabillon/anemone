Search Iteration Architecture
=============================

This document follows one search runtime from construction through one
``TreeExploration.step()`` iteration. It is intentionally practical: the goal is
to show which layers own which responsibilities in the current implementation,
not to propose a new design.

Canonical vocabulary
--------------------

- ``ITreeNode``: the structural/navigation protocol shared by tree helpers.
- ``TreeNode``: the concrete structural node that stores state, parent/child
  links, depth, and branch-opening bookkeeping.
- ``AlgorithmNode``: the runtime/search wrapper around a ``TreeNode``. It adds
  tree evaluation, exploration-index payloads, and optional evaluator
  representations.
- ``direct_value``: the immediate evaluator output for one node.
- ``backed_up_value``: the subtree-derived value propagated from children.
- candidate value: the best currently available value, preferring
  ``backed_up_value`` over ``direct_value`` when both exist.
- canonical value: the required concrete ``Value`` returned to consumers that
  need one; it is obtained from the current candidate value and therefore
  raises if no value is available yet.
- tree evaluation: the runtime state attached to an ``AlgorithmNode`` that owns
  candidate/canonical value access, branch ordering, branch frontier tracking,
  and principal variation (PV).
- exploration-index payload: the per-node data object stored on an
  ``AlgorithmNode`` for search-priority strategies.
- selector: the object that decides which node or branch to open next.
- opening instructor: the object that turns "open this node" into concrete
  branch-opening instructions.
- runtime assembly: the internal wiring that builds a runnable search runtime
  from factories and strategy configuration.

Main architectural layers
-------------------------

Public runtime creation
~~~~~~~~~~~~~~~~~~~~~~~

``anemone.factory`` is the canonical public entry point. ``create_search(...)``
builds a runnable ``TreeExploration`` runtime. ``TreeAndValueBranchSelector``
is the secondary convenience wrapper for callers who only want
``recommend(...)``.

Internal runtime assembly
~~~~~~~~~~~~~~~~~~~~~~~~~

``anemone._runtime_assembly`` wires together the direct evaluator, tree
evaluation factory, selector factory, tree factory, and tree manager.
``anemone.search_factory`` is an assembly helper, not the preferred public
entry point.

Structural and runtime node layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TreeNode`` owns only structural concerns. ``AlgorithmNode`` wraps it and
adds runtime state. Generic tree helpers stay on ``ITreeNode``; search logic
works with ``AlgorithmNode``.

Direct evaluation
~~~~~~~~~~~~~~~~~

``anemone.node_evaluation.direct`` owns the immediate evaluator pass that fills
``direct_value`` on newly created nodes.

Tree evaluation
~~~~~~~~~~~~~~~

``anemone.node_evaluation.tree`` owns the search-time evaluation state attached
to each node: candidate/canonical value access, branch ordering, branch
frontier, and PV tracking. Family-specific backup semantics are injected through
backup policies.

Exploration indices
~~~~~~~~~~~~~~~~~~~

``anemone.indices.node_indices`` defines per-node exploration payload classes.
``anemone.indices.index_manager`` chooses the update strategy and recomputes
payloads across the tree.

Selector and opening flow
~~~~~~~~~~~~~~~~~~~~~~~~~

``anemone.node_selector`` chooses where the search goes next. Selectors return
``OpeningInstructions``, and the ``OpeningInstructor`` expands a selected node
into the concrete branches to open. Depending on configuration, selectors may
use only evaluations, or they may also consult frontier and exploration-index
state.

One search iteration: step by step
----------------------------------

The current implementation is easiest to read as this flow:

.. code-block:: text

   create_search(...)
     -> assemble_search_runtime_dependencies(...)
     -> create_tree_exploration(...)
     -> TreeExploration.step()
        1. choose openings
        2. expand structurally
        3. direct-evaluate new nodes
        4. propagate backup / PV / branch-order updates upward
        5. refresh exploration indices

1. Runtime creation starts at the public factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``anemone.factory.create_search(...)`` creates a default
``NodeTreeMinmaxEvaluationFactory`` and then delegates to
``create_search_with_tree_eval_factory(...)``. That function calls
``anemone._runtime_assembly.assemble_search_runtime_dependencies(...)``, which
builds:

- a ``NodeDirectEvaluator``
- a ``SearchFactory`` that can create the configured selector and per-node
  exploration payloads
- an ``AlgorithmNodeFactory``
- a ``ValueTreeFactory``
- an ``AlgorithmNodeTreeManager``

Finally, ``create_tree_exploration(...)`` builds the runtime object itself.

2. Tree creation evaluates the root before the first step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ValueTreeFactory.create(...)`` creates the root ``AlgorithmNode`` and
immediately runs the direct evaluator on it. So the tree handed to
``TreeExploration`` already contains:

- a structural root ``TreeNode``
- a runtime root ``AlgorithmNode``
- a tree-evaluation object
- an exploration-index payload shell (if configured)
- a root ``direct_value``

This happens before the first call to ``TreeExploration.step()``.

3. The selector proposes what to open next
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TreeExploration.step()`` starts by calling
``node_selector.choose_node_and_branch_to_open(...)``. The selector sees:

- the current tree
- the latest ``TreeExpansions`` log from the previous iteration

In the current implementation, selectors are often composed from:

- an optional priority override
- a base selector such as Uniform, RecurZipf, or Sequool

The selector decides which node to open next. The ``OpeningInstructor`` then
turns that choice into concrete branches to open. With the current
``OpeningType.ALL_CHILDREN`` mode, this means "open every legal action from the
selected node".

``TreeExploration`` then lets the stopping criterion shrink that batch if the
current search budget requires it.

4. Structural expansion opens or reuses child nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmNodeTreeManager.expand_instructions(...)`` does two things:

- it marks the selected branches as opened in the node's branch-frontier state
- it delegates to ``TreeManager.expand_instructions(...)`` for structural work

``TreeManager`` applies ``dynamics.step(...)`` for each opening instruction,
then either:

- creates a new child node via ``AlgorithmNodeFactory`` and ``TreeNodeFactory``,
  or
- reuses an existing node if the reached state already exists at that depth

Each structural change is recorded as a ``TreeExpansion`` inside a
``TreeExpansions`` log. Newly created ``AlgorithmNode`` objects already contain
their tree-evaluation object and exploration-index payload object, but they have
not been directly evaluated yet.

5. Direct evaluation fills ``direct_value`` on newly created nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmNodeTreeManager.evaluate_expansions(...)`` queues each newly created
node in ``NodeDirectEvaluator``.

``NodeDirectEvaluator`` first checks for obvious terminal outcomes on the node's
own state. Non-terminal nodes are then batch-evaluated by the configured master
state evaluator. The result is written into ``node.tree_evaluation.direct_value``.

At this point, a newly created node has immediate local evaluation data, but its
ancestors have not been recomputed yet.

6. Upward propagation recomputes backed-up values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmNodeTreeManager.update_backward(...)`` triggers
``ValuePropagator.propagate_from_changed_nodes(...)`` from the changed child
nodes. The propagator only owns scheduling: it processes dirty ancestors in
descending depth order and recomputes each dirty node from a full snapshot of
its currently open children.

For each dirty ancestor, the propagator calls
``node.tree_evaluation.backup_from_children(...)``. In the default minimax path,
that delegates to ``ExplicitTreeBackupPolicy`` through the configured backup
policy.

7. Branch ordering and PV updates happen inside backup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runtime loop does not update branch ordering or PV directly. That happens as
part of backup:

- the backup policy refreshes cached branch-ordering keys for the changed child
  branches
- it computes the new ``backed_up_value``
- it updates frontier membership
- it rebuilds or refreshes the principal variation when the best branch or a
  best-child PV changed

The generic mechanics live in:

- ``anemone.node_evaluation.tree.node_tree_evaluation``
- ``anemone.node_evaluation.tree.branch_ordering_runtime``
- ``anemone.node_evaluation.tree.principal_variation_runtime``
- ``anemone.backup_policies.*``

8. Optional depth metadata and exploration-index payloads are refreshed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After value propagation, ``TreeExploration`` runs two more refresh phases:

- ``propagate_depth_index(...)`` updates descendant-depth payloads when that
  feature is enabled
- ``refresh_exploration_indices(...)`` calls
  ``anemone.indices.index_manager.update_all_indices(...)``

``update_all_indices(...)`` initializes the root exploration payload, then walks
the tree depth by depth. For each parent, it uses the parent's current decision
ordering to recompute child exploration data with the configured strategy.
Depending on configuration, that strategy may be:

- global-min-change
- interval/local-min-change
- recursive Zipf/factored-probability
- or the null/no-op strategy

9. The next iteration sees the updated runtime state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``TreeExploration.step()`` finishes, the tree is richer in three ways:

- the structure may contain newly opened children
- direct and backed-up values may have changed
- branch frontier, branch ordering, PV, and exploration-index payloads may have
  changed

The selector on the next iteration reads this updated tree. Which parts it
actually uses depends on the configured selector strategy.

Where to look in the code
-------------------------

- public runtime creation:
  ``anemone.factory`` and ``anemone.tree_and_value_branch_selector``
- internal wiring:
  ``anemone._runtime_assembly`` and ``anemone.search_factory``
- runtime loop:
  ``anemone.tree_exploration``
- structural vs runtime nodes:
  ``anemone.nodes.itree_node``, ``anemone.nodes.tree_node``,
  ``anemone.nodes.algorithm_node.algorithm_node``
- node creation:
  ``anemone.node_factory.algorithm_node_factory`` and ``anemone.trees.factory``
- direct evaluation:
  ``anemone.node_evaluation.direct.factory`` and
  ``anemone.node_evaluation.direct.node_direct_evaluator``
- tree evaluation, branch ordering, and PV:
  ``anemone.node_evaluation.tree.node_tree_evaluation``,
  ``anemone.node_evaluation.tree.branch_ordering_runtime``, and
  ``anemone.node_evaluation.tree.principal_variation_runtime``
- backup sequencing:
  ``anemone.backup_policies.explicit_tree`` and the family-specific backup
  policies in ``anemone.backup_policies``
- exploration indices:
  ``anemone.indices.node_indices`` and
  ``anemone.indices.index_manager.node_exploration_manager``
- selector/opening flow:
  ``anemone.node_selector.factory``,
  ``anemone.node_selector.composed.composed_node_selector``, and
  ``anemone.node_selector.opening_instructions``
- tree-wide structural and algorithm-aware phases:
  ``anemone.tree_manager.tree_manager`` and
  ``anemone.tree_manager.algorithm_node_tree_manager``

Common confusions and notes
---------------------------

- ``anemone.factory`` is the public entry point; ``anemone.search_factory`` is
  assembly infrastructure behind it.
- ``TreeNode`` stores structure; ``AlgorithmNode`` is the object search code
  works with at runtime.
- ``direct_value`` is local evaluator output, ``backed_up_value`` is subtree
  backup output, the candidate value is the best available one, and the
  canonical value is the required concrete ``Value`` returned to callers.
- Exploration-index payloads are stored per node, but recomputation is a
  tree-wide strategy pass.
- Branch ordering and PV updates are not a separate top-level runtime phase;
  they are triggered inside backup.
- The search loop typically stops when the root value becomes exact, even if the
  root state is not terminal and some siblings remain unopened.
