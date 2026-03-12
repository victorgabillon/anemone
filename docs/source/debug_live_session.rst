Live debug session
==================

The recommended way to run the browser debugger around a real exploration is
through :func:`anemone.debug.build_live_debug_environment`.

Basic workflow
--------------

.. code-block:: python

   from random import Random

   from anemone.debug import (
       build_live_debug_environment,
       serve_live_debug_session,
   )

   debug_environment = build_live_debug_environment(
       tree_exploration=tree_exploration,
       session_directory="debug-session",
   )

   result = debug_environment.controlled_exploration.explore(
       random_generator=Random(0)
   )
   debug_environment.finalize()

   serve_live_debug_session("debug-session", port=8000)

What gets written
-----------------

The live recorder writes a session directory containing:

* ``index.html`` for the browser viewer
* ``session.json`` with the current timeline payload
* ``control_state.json`` and ``commands.json`` for live interaction state
* ``snapshots/`` with exported graph snapshots and metadata sidecars

Lifecycle
---------

The session recorder writes incrementally during the search run. Calling
``debug_environment.finalize()`` marks ``session.json`` as complete. If the
process exits early, the browser viewer can still open the session, but it will
remain marked incomplete.

Reference example
-----------------

See ``examples/debug_live_session_example.py`` for a small helper-oriented
reference workflow.
