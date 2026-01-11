# Progress monitoring

Progress monitors implement stopping criteria for tree exploration.

## Key pieces

- `progress_monitor.py` defines the `ProgressMonitor` base class and
  `StoppingCriterionTypes` enum.
- Concrete criteria include `TreeMoveLimit` and `DepthLimit`, plus data classes
  for configuring them.

These monitors are injected into `TreeExploration` to control when the search
halts.
