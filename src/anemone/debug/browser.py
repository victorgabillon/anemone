"""Local browser serving helpers for debug replay bundles and live sessions."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

from .breakpoints import breakpoint_from_json
from .live_control import DebugCommand, DebugSessionController
from .replay_bundle import export_replay_bundle

if TYPE_CHECKING:
    from socket import socket
    from socketserver import BaseServer

    from .recording import DebugTrace


class DebugSessionHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static debug files and accept simple live control commands."""

    def __init__(
        self,
        request: socket,
        client_address: tuple[str, int],
        server: BaseServer,
        directory: str,
        session_directory: str,
    ) -> None:
        """Initialize the handler with a static directory and command target."""
        self._controller = DebugSessionController(session_directory)
        super().__init__(
            request,
            client_address,
            server,
            directory=directory,
        )

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle live control commands posted by the browser."""
        route = urlparse(self.path).path
        if route == "/command":
            self._handle_command_post()
            return
        if route == "/breakpoints":
            self._handle_breakpoints_post()
            return
        self.send_error(404, "Unknown debug command endpoint")

    def _handle_command_post(self) -> None:
        """Handle live control commands posted by the browser."""
        payload = self._read_request_payload()
        if payload is None:
            return

        command_name = payload.get("command")
        if not isinstance(command_name, str):
            self.send_error(400, "Unknown debug command")
            return
        try:
            command = DebugCommand(command_name)
        except ValueError:
            self.send_error(400, "Unknown debug command")
            return

        match command:
            case DebugCommand.PAUSE:
                self._controller.request_pause()
            case DebugCommand.RESUME:
                self._controller.request_resume()
            case DebugCommand.STEP:
                self._controller.request_step()
            case DebugCommand.EXPAND_NODE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                self._controller.expand_node(node_id)
            case DebugCommand.RUN_UNTIL_NODE_EVENT:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                self._controller.run_until_node_event(node_id)
            case DebugCommand.RUN_UNTIL_NODE_VALUE_CHANGE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                self._controller.run_until_node_value_change(node_id)
            case DebugCommand.FOCUS_NODE_TIMELINE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                self._controller.focus_node_timeline(node_id)
            case DebugCommand.CLEAR_TIMELINE_FOCUS:
                self._controller.clear_timeline_focus()
            case DebugCommand.NONE:
                self.send_error(400, "Unknown debug command")
                return

        self.send_response(204)
        self.end_headers()

    def _handle_breakpoints_post(self) -> None:
        """Handle breakpoint add/clear requests posted by the browser."""
        payload = self._read_request_payload()
        if payload is None:
            return

        action = payload.get("action")
        if action == "clear":
            self._controller.clear_breakpoints()
            self.send_response(204)
            self.end_headers()
            return

        if action != "add":
            self.send_error(400, "Unknown breakpoint action")
            return

        breakpoint_payload = payload.get("breakpoint")
        if not isinstance(breakpoint_payload, dict):
            self.send_error(400, "Invalid breakpoint payload")
            return

        try:
            breakpoint_spec = breakpoint_from_json(
                cast("dict[str, object]", breakpoint_payload)
            )
        except (TypeError, ValueError):
            self.send_error(400, "Invalid breakpoint payload")
            return

        self._controller.add_breakpoint(breakpoint_spec)
        self.send_response(204)
        self.end_headers()

    def _read_request_payload(self) -> dict[str, object] | None:
        """Read and validate one JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid content length")
            return None

        try:
            payload = json.loads(
                self.rfile.read(content_length).decode("utf-8") or "{}"
            )
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON payload")
            return None

        if not isinstance(payload, dict):
            self.send_error(400, "Invalid JSON payload")
            return None
        return cast("dict[str, object]", payload)

    def _read_required_node_id(self, payload: dict[str, object]) -> str | None:
        """Return a non-empty ``node_id`` field or emit ``400``."""
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            self.send_error(400, "Invalid or missing node_id")
            return None
        return node_id


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


def serve_live_debug_session(
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a live debug session directory over HTTP until interrupted."""
    bundle_dir = Path(directory).resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(str(bundle_dir))

    handler = partial(
        DebugSessionHTTPRequestHandler,
        directory=str(bundle_dir),
        session_directory=str(bundle_dir),
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Serving live debug session at http://{host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping live debug session server.")


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


__all__ = [
    "DebugSessionHTTPRequestHandler",
    "export_and_serve_trace",
    "serve_live_debug_session",
    "serve_replay_bundle",
]
