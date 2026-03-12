"""Canonical example for running a live Anemone debug session.

This example keeps application-specific exploration construction outside of the
debug package. Pass an already-built exploration object plus a random
generator, and this module demonstrates the recommended public API:

* ``build_live_debug_environment(...)``
* ``env.controlled_exploration.explore(...)``
* ``env.finalize()``
* ``serve_live_debug_session(...)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.debug import build_live_debug_environment, serve_live_debug_session

if TYPE_CHECKING:
    from pathlib import Path
    from random import Random


_MODULE_USAGE_MESSAGE = (
    "Import run_live_debug_session(...) into a project-specific script that "
    "constructs a real tree exploration object, then call "
    "print_serving_instructions(...) or serve_debug_session(...) to view the "
    "live session."
)


def run_live_debug_session(
    tree_exploration: Any,
    *,
    session_directory: str | Path,
    random_generator: Random,
) -> Any:
    """Run ``tree_exploration`` inside the supported live debug environment."""
    debug_environment = build_live_debug_environment(
        tree_exploration=tree_exploration,
        session_directory=session_directory,
    )
    try:
        return debug_environment.controlled_exploration.explore(
            random_generator=random_generator
        )
    finally:
        debug_environment.finalize()


def print_serving_instructions(
    session_directory: str | Path,
    *,
    port: int = 8000,
) -> None:
    """Print the recommended command for serving the finished live session."""
    print(
        "Serve the live debug session with "
        f"serve_live_debug_session({session_directory!r}, port={port})."
    )


def serve_debug_session(session_directory: str | Path, *, port: int = 8000) -> None:
    """Serve an existing live debug session directory in the local browser UI."""
    serve_live_debug_session(session_directory, port=port)


if __name__ == "__main__":
    raise SystemExit(_MODULE_USAGE_MESSAGE)
