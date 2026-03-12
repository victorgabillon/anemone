"""Local browser serving helpers for static debug replay bundles."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from .replay_bundle import export_replay_bundle

if TYPE_CHECKING:
    from .recording import DebugTrace


def serve_replay_bundle(
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a replay bundle over HTTP until interrupted."""
    bundle_dir = Path(directory).resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(str(bundle_dir))

    handler = partial(SimpleHTTPRequestHandler, directory=str(bundle_dir))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Serving debug replay bundle at http://{host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping debug replay bundle server.")


def export_and_serve_trace(
    trace: DebugTrace,
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    snapshot_format: str = "svg",
) -> None:
    """Export ``trace`` as a replay bundle and serve it locally."""
    bundle_dir = export_replay_bundle(
        trace,
        directory,
        snapshot_format=snapshot_format,
    )
    serve_replay_bundle(bundle_dir, host=host, port=port)


__all__ = ["export_and_serve_trace", "serve_replay_bundle"]
