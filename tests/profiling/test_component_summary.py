"""Tests for component-summary artifact helpers."""

from anemone.profiling.component_summary import (
    ComponentSummary,
    TimedCallStats,
    load_component_summary,
    save_component_summary,
)


def test_component_summary_round_trip_preserves_fields(tmp_path) -> None:
    """Component summary artifacts should survive a save/load round trip."""
    summary = ComponentSummary(
        total_run_wall_time_seconds=0.024,
        total_profiled_component_wall_time_seconds=0.011,
        residual_framework_wall_time_seconds=0.013,
        evaluator=TimedCallStats(
            call_count=3,
            total_wall_time_seconds=0.004,
            max_wall_time_seconds=0.002,
            min_wall_time_seconds=0.001,
            mean_wall_time_seconds=0.0013333333,
        ),
        dynamics_step=TimedCallStats(
            call_count=2,
            total_wall_time_seconds=0.005,
            max_wall_time_seconds=0.003,
            min_wall_time_seconds=0.002,
            mean_wall_time_seconds=0.0025,
        ),
        dynamics_legal_actions=TimedCallStats(
            call_count=3,
            total_wall_time_seconds=0.002,
            max_wall_time_seconds=0.001,
            min_wall_time_seconds=0.0005,
            mean_wall_time_seconds=0.0006666667,
        ),
        notes={
            "residual_definition": "total_run_wall_time - wrapped_component_wall_times"
        },
    )

    path = tmp_path / "component_summary.json"
    save_component_summary(summary, path)

    loaded = load_component_summary(path)

    assert loaded == summary
