"""Launcher for ``python -m anemone.profiling.gui``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the local profiling dashboard via Streamlit."""
    try:
        get_streamlit()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    argv_list = list(sys.argv[1:] if argv is None else argv)
    app_path = Path(__file__).with_name("app.py")
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *argv_list],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
