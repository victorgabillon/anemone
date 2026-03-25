# Profiling Foundation

This document describes the standalone profiling foundation for Anemone.

## Purpose

The profiling package can:

- run a named scenario
- measure total wall-clock time
- capture basic execution metadata
- persist a stable JSON artifact for each run

The profiling foundation is intentionally separate from the core search engine.

## Constraints

This first version does not modify core search semantics and does not add timing
or profiler hooks inside the search loop.

In particular, PR1 does not touch:

- `TreeExploration.step()`
- `AlgorithmNodeTreeManager`

## What Exists Today

- `anemone.profiling` package under `src/anemone/profiling/`
- stable `RunResult` JSON schema with `schema_version = "1"`
- run directory helpers
- a standalone scenario runner
- a minimal CLI
- a tiny real `smoke` scenario using public Anemone APIs
- lazy scenario loading so package import stays lightweight
- optional external profiler modes: `none`, `cprofile`, `pyinstrument`
- wrapper-based component timing for evaluator and dynamics
- `component_summary.json` artifact output when enabled
- `cprofile.pstats` and `cprofile_top.txt` artifacts for `cprofile`
- `pyinstrument.txt` artifact when `pyinstrument` is requested and installed
- deterministic synthetic profiling scenarios:
  - `cheap_eval`
  - `expensive_eval`
  - `wide_tree`
  - `deep_tree`
  - `reuse_heavy`
- repeatable profiling suites:
  - `baseline`
  - `quick`
- `suite.json` artifact output for repeated suite runs

## What Is Still Out Of Scope

- process-level profilers such as py-spy
- viztracer integration
- GUI tooling, charts, or HTML reports
- CI performance regression gates

## Running It

Preferred CLI:

```bash
python -m anemone.profiling.cli list-scenarios
python -m anemone.profiling.cli list-suites
python -m anemone.profiling.cli run --scenario smoke --output-dir profiling_runs
python -m anemone.profiling.cli run --scenario smoke --output-dir profiling_runs --profiler cprofile --component-summary
python -m anemone.profiling.cli run-suite --suite baseline --output-dir profiling_runs --repetitions 5 --component-summary
```

GUI launcher:

```bash
pip install -e .[gui]
python -m anemone.profiling.gui
```

Convenience runner entrypoint:

```bash
python -m anemone.profiling.runner --scenario smoke --output-dir profiling_runs
```

## Output Layout

Each run creates a timestamped folder under the selected base directory:

```text
profiling_runs/
  2026-03-24T14-32-10_smoke/
    run.json
    component_summary.json
    cprofile.pstats
    cprofile_top.txt
```

If a run id collides, the storage helper appends a deterministic suffix such as
`_2`, `_3`, and so on.

The `run.json` artifact includes scenario metadata, execution metadata, top-level
wall time, run status, and placeholder artifact references for future tooling.
The recorded wall time tracks scenario execution itself and intentionally
excludes profiler artifact writing or profiler post-processing.

Suite runs create a separate suite-level directory with nested scenario runs:

```text
profiling_runs/
  2026-03-24T15-10-00_baseline_suite/
    suite.json
    scenario_runs/
      2026-03-24T15-10-00_cheap_eval_rep1/
        run.json
        component_summary.json
      2026-03-24T15-10-01_cheap_eval_rep2/
        run.json
      2026-03-24T15-10-05_wide_tree_rep1/
        run.json
```

`suite.json` records:

- suite metadata
- requested repetition count
- profiler and component-summary settings
- per-scenario aggregate wall-time statistics across successful repetitions
- per-repetition `run.json` paths and statuses

This keeps the suite artifact comparison-ready without requiring later tooling to
re-scan directories.

## GUI

PR4 adds a local Streamlit dashboard under `anemone.profiling.gui`.

The dashboard can:

- launch scenarios and suites
- browse existing runs and suites
- display component timing breakdowns
- show readable profiler text artifacts
- compare two runs or two suites

The GUI reads the existing profiling artifacts only. It does not add new
profiling hooks or modify core search behavior.

When component summaries are enabled, the summary approximates framework
overhead as:

```text
residual_framework_wall_time_seconds
    = total_run_wall_time_seconds - wrapped_component_wall_times
```

This residual is useful, but intentionally approximate.

## Scenario Roles

- `smoke` validates profiling plumbing and public API integration.
- `cheap_eval` keeps evaluator work near zero so framework overhead is easier to see.
- `expensive_eval` makes evaluator CPU cost dominant while keeping the tree shape similar.
- `wide_tree` stresses broader opening pressure and legal-action generation.
- `deep_tree` stresses deeper narrow traversal.
- `reuse_heavy` stresses repeated shared-state patterns.
