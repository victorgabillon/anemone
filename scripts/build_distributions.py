"""Build distributions without generating metadata or caches in the checkout."""

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Export tracked build inputs into temporary PEP 517 source staging."""
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    with TemporaryDirectory(prefix="build-source-", dir=output.parent) as temporary:
        source = Path(temporary)
        for relative in filter(None, files):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(output),
                str(source),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
