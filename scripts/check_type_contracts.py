"""Require both installed type checkers to reject each marked invalid use."""

import json
import re
import subprocess
import sys
from pathlib import Path

fixture = Path("tests/typing/invalid.py")
expected = {
    number
    for number, line in enumerate(fixture.read_text().splitlines(), start=1)
    if "# expect-error" in line
}
commands = {
    "mypy": ["mypy", "--strict", "--no-pretty", str(fixture)],
    "pyright": ["pyright", "--outputjson", str(fixture)],
}
for name, command in commands.items():
    result = subprocess.run(
        [sys.executable, "-m", *command], capture_output=True, text=True, check=False
    )
    if name == "mypy":
        errors = {
            int(match.group(1))
            for match in re.finditer(r"invalid\.py:(\d+): error:", result.stdout)
        }
        unexpected = [
            line
            for line in result.stdout.splitlines()
            if ": error:" in line and "invalid.py:" not in line
        ]
    else:
        diagnostics = json.loads(result.stdout)["generalDiagnostics"]
        errors = {
            entry["range"]["start"]["line"] + 1
            for entry in diagnostics
            if entry["severity"] == "error" and entry["file"].endswith("invalid.py")
        }
        unexpected = [
            entry
            for entry in diagnostics
            if entry["severity"] == "error" and not entry["file"].endswith("invalid.py")
        ]
    if result.returncode != 1 or errors != expected or unexpected:
        print(result.stdout, result.stderr)
        print(
            f"{name}: expected errors on {sorted(expected)}, got {sorted(errors)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"{name}: rejected all {len(expected)} invalid generic contracts")
