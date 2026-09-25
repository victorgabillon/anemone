# CI performance and preserved checks

CI still runs all four tox environments: installed-package tests, Ruff/format and
Pylint, Mypy/Pyright, and build/Twine. Two environments run concurrently; the
required `ci (3.13)` job fails if any environment fails. Tool versions, analyzer
options, runtime dependencies, and release gates are unchanged.

## Test equivalence

- The 149 checkpoint RNG cases retain all graphs, seeds, checkpoint formats,
  continuation comparisons, and assertions. A function-scoped plain logging
  handler avoids Rich terminal rendering while preserving log levels and messages.
  The original handler is restored after each case. Before and after, these tests
  prove the same continuous/save/restore behavior; lost assertions and behavior
  coverage: zero.
- Browser tests still serve real HTTP requests and shut down/join real threads.
  The server's idle polling interval is 0.01 seconds rather than the stdlib's
  0.5-second default; request and test timeouts are unchanged. An added assertion
  checks that shutdown left no live thread. Lost assertions and behavior coverage:
  zero.
- One explicit branch-comparison regression covers both stable fallback directions
  and equality. Previously these paths were covered incidentally depending on sort
  input order. All 899 original node IDs remain; the new regression raises the
  population to 900.

`scripts/ci_test_metrics.py` records selected/deselected node IDs, every test phase,
Junit results, and line/branch coverage in the `ci-validation-evidence` artifact.
It observes pytest and returns its actual exit status. GitHub job logs retain all
four environments' output (`tox -p 2 -o --parallel-no-spinner`).

The performance audit compares the exact covered production lines/branches and
original per-test outcomes, not only rounded percentages. Release CI continues to
run the full quality suite at the tagged SHA; no latest-main success is substituted.
