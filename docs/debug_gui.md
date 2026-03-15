# Debug GUI

## What the debug GUI is

The Anemone debug GUI records exploration events and tree snapshots, then
renders them in a browser for live or offline inspection. It is designed to
help debug tree growth, direct evaluation, backup propagation, breakpoints, and
node-level state.

The recommended workflow starts with
`anemone.debug.build_live_debug_environment(...)`.

## Recommended quickstart

```python
from random import Random

from anemone.debug import (
    build_live_debug_environment,
    serve_live_debug_session,
)

env = build_live_debug_environment(
    tree_exploration=tree_exploration,
    session_directory="runs/debug-session",
)

result = env.controlled_exploration.explore(random_generator=Random(0))
env.finalize()

serve_live_debug_session("runs/debug-session", port=8000)
```

During the run, the live debug environment writes the session directory
incrementally. The snippet above shows post-run viewing. For true live viewing,
start `serve_live_debug_session(...)` from another terminal or process while
the exploration is still running, because the server call blocks the current
process. After the run, the same directory can still be opened in the browser
as a completed live session.

## What files are written

A typical live session directory contains:

```text
index.html
session.json
control_state.json
commands.json
breakpoints.json
snapshots/
```

Semantics:

- `session.json`: the current live session timeline payload.
- `control_state.json`: pause, step, focus, and breakpoint UI state.
- `commands.json`: the latest browser-to-controller command.
- `breakpoints.json`: persisted breakpoint configuration.
- `snapshots/`: exported graph snapshots plus `.snapshot.json` metadata sidecars.

Browser modes:

- `serve_live_debug_session(...)`: serve a mutable live session directory while
  exploration is running or after it completes.
- `serve_debug_browser(...)`: serve a browser root that can launch built-in toy
  scenarios directly from the GUI.
- `serve_replay_bundle(...)`: serve a static replay/export bundle built from a
  finished trace.

Replay bundles typically replace `session.json` with a static `trace.json`
payload while reusing the browser-facing snapshot directory.

Related terms:

- session: live on-disk state while exploration is running
- trace: ordered event history in memory or serialized form
- snapshot: one captured tree state
- bundle: a browser-viewable exported replay directory
- replay: offline browsing of a recorded run

## Live browser usage

The browser UI supports:

- timeline browsing with search, event filtering, and jump controls
- live pause, resume, and step controls
- breakpoint configuration and auto-pause
- node list and graph node selection
- structured node inspection
- node-focused timeline filtering and node commands
- graph highlighting for root path, neighborhood, PV path, and dimming

## Important semantics and caveats

- `Expand Node` is best-effort, not a hard guarantee.
- node-focused `run until` actions pause when a matching future event is observed.
- if an event has no attached snapshot, the browser shows the nearest prior snapshot.
- if the process exits before `env.finalize()`, the session remains marked incomplete.
- timeline focus is a browser-side view/filter concern, not a search semantic concern.

## Replay / offline usage

For offline workflows, export or serve replay artifacts instead of a mutable
live session:

```python
from anemone.debug import export_and_serve_trace, serve_replay_bundle
```

Use replay bundles when you already have a completed trace or want a static
browser-viewable directory.

## API reference summary

Recommended public entrypoints:

- `build_live_debug_environment`
- `serve_debug_browser`
- `serve_live_debug_session`
- `serve_replay_bundle`
- `export_and_serve_trace`

Supported replay helpers:

- `load_debug_trace`
- `save_debug_trace`
- `TraceReplayView`
