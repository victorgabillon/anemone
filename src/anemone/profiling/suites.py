"""Registry for repeatable profiling suites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfilingSuite:
    """Named bundle of profiling scenarios intended to run together."""

    name: str
    description: str
    scenario_names: tuple[str, ...]


_SUITES: dict[str, ProfilingSuite] = {}


def register_suite(suite: ProfilingSuite) -> ProfilingSuite:
    """Register one profiling suite."""
    _SUITES[suite.name] = suite
    return suite


def get_suite(name: str) -> ProfilingSuite:
    """Return one registered profiling suite by name."""
    try:
        return _SUITES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SUITES))
        raise _unknown_suite_error(name, available) from exc


def list_suites() -> list[ProfilingSuite]:
    """Return all registered profiling suites sorted by name."""
    return [_SUITES[name] for name in sorted(_SUITES)]


register_suite(
    ProfilingSuite(
        name="baseline",
        description="Baseline deterministic profiling suite for Anemone.",
        scenario_names=(
            "cheap_eval",
            "expensive_eval",
            "wide_tree",
            "deep_tree",
            "reuse_heavy",
        ),
    )
)

register_suite(
    ProfilingSuite(
        name="quick",
        description="Small deterministic suite for fast local profiling checks.",
        scenario_names=("cheap_eval", "wide_tree"),
    )
)


def _unknown_suite_error(name: str, available: str) -> ValueError:
    return ValueError(
        f"Unknown profiling suite {name!r}. Available suites: {available}"
    )
