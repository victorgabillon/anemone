"""Local browser serving helpers for debug replay bundles and live sessions."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

from .breakpoints import breakpoint_from_json
from .html_templates import render_replay_index_html
from .live_control import DebugCommand, DebugSessionController
from .replay_bundle import export_replay_bundle
from .toy_scenarios import run_registered_scenario, serialize_registered_scenarios

if TYPE_CHECKING:
    from socket import socket
    from socketserver import BaseServer

    from .recording import DebugTrace


class DebugSessionHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static debug files, scenario APIs, and live control commands."""

    def __init__(
        self,
        request: socket,
        client_address: tuple[str, int],
        server: BaseServer,
        directory: str,
        session_directory: str | None = None,
        browser_session_root: str | None = None,
    ) -> None:
        """Initialize the handler with a static directory and command target."""
        self._directory_root = Path(directory).resolve()
        self._default_session_directory = (
            Path(session_directory).resolve() if session_directory is not None else None
        )
        self._browser_session_root = (
            Path(browser_session_root).resolve()
            if browser_session_root is not None
            else self._directory_root / "sessions"
        )
        super().__init__(
            request,
            client_address,
            server,
            directory=str(self._directory_root),
        )

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handle browser API reads before falling back to static file serving."""
        route = urlparse(self.path).path
        if route == "/api/scenarios":
            self._handle_scenarios_get()
            return
        super().do_GET()

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle live control commands posted by the browser."""
        route = urlparse(self.path).path
        if route == "/api/run_scenario":
            self._handle_run_scenario_post()
            return
        command_controller = self._controller_for_route(route, endpoint_name="command")
        if command_controller is not None:
            self._handle_command_post(command_controller)
            return
        breakpoint_controller = self._controller_for_route(
            route,
            endpoint_name="breakpoints",
        )
        if breakpoint_controller is not None:
            self._handle_breakpoints_post(breakpoint_controller)
            return
        self.send_error(404, "Unknown debug command endpoint")

    def _handle_command_post(self, controller: DebugSessionController) -> None:
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
                controller.request_pause()
            case DebugCommand.RESUME:
                controller.request_resume()
            case DebugCommand.STEP:
                controller.request_step()
            case DebugCommand.EXPAND_NODE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                controller.expand_node(node_id)
            case DebugCommand.RUN_UNTIL_NODE_EVENT:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                controller.run_until_node_event(node_id)
            case DebugCommand.RUN_UNTIL_NODE_VALUE_CHANGE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                controller.run_until_node_value_change(node_id)
            case DebugCommand.FOCUS_NODE_TIMELINE:
                node_id = self._read_required_node_id(payload)
                if node_id is None:
                    return
                controller.focus_node_timeline(node_id)
            case DebugCommand.CLEAR_TIMELINE_FOCUS:
                controller.clear_timeline_focus()
            case DebugCommand.NONE:
                self.send_error(400, "Unknown debug command")
                return

        self.send_response(204)
        self.end_headers()

    def _handle_breakpoints_post(self, controller: DebugSessionController) -> None:
        """Handle breakpoint add/clear requests posted by the browser."""
        payload = self._read_request_payload()
        if payload is None:
            return

        action = payload.get("action")
        if action == "clear":
            controller.clear_breakpoints()
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

        controller.add_breakpoint(breakpoint_spec)
        self.send_response(204)
        self.end_headers()

    def _handle_scenarios_get(self) -> None:
        """Return the curated toy scenarios that may be launched in-browser."""
        self._send_json_response(
            200,
            {"scenarios": serialize_registered_scenarios()},
        )

    def _handle_run_scenario_post(self) -> None:
        """Run one registered toy scenario and return the created session path."""
        payload = self._read_request_payload()
        if payload is None:
            return

        scenario_name = payload.get("name")
        if not isinstance(scenario_name, str) or not scenario_name:
            self.send_error(400, "Invalid or missing scenario name")
            return

        run_result = run_registered_scenario(
            scenario_name,
            output_root_directory=self._browser_session_root,
        )
        response_payload = {
            "ok": run_result.ok,
            "scenario_name": run_result.scenario_name,
            "session_directory": run_result.session_directory,
            "session_path": run_result.session_path,
            "message": run_result.message,
            "error": run_result.error,
        }
        if run_result.ok:
            self._send_json_response(200, response_payload)
            return

        status_code = 404 if run_result.error == "unknown_scenario" else 500
        self._send_json_response(status_code, response_payload)

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

    def _controller_for_route(
        self,
        route: str,
        *,
        endpoint_name: str,
    ) -> DebugSessionController | None:
        """Return the live-session controller targeted by ``route``."""
        session_directory = self._resolve_session_directory(route, endpoint_name)
        if session_directory is None:
            return None
        return DebugSessionController(session_directory)

    def _resolve_session_directory(
        self,
        route: str,
        endpoint_name: str,
    ) -> Path | None:
        """Resolve the session directory targeted by one control endpoint route."""
        if route == f"/{endpoint_name}" and self._default_session_directory is not None:
            return self._default_session_directory

        route_parts = [part for part in route.split("/") if part]
        if len(route_parts) < 3:
            return None
        if route_parts[0] != "sessions" or route_parts[-1] != endpoint_name:
            return None

        candidate = (self._directory_root / Path(*route_parts[:-1])).resolve()
        try:
            candidate.relative_to(self._directory_root)
        except ValueError:
            return None
        if not candidate.is_dir():
            return None
        return candidate

    def _read_required_node_id(self, payload: dict[str, object]) -> str | None:
        """Return a non-empty ``node_id`` field or emit ``400``."""
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            self.send_error(400, "Invalid or missing node_id")
            return None
        return node_id

    def _send_json_response(
        self,
        status_code: int,
        payload: dict[str, object],
    ) -> None:
        """Write one JSON response body."""
        response_body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def serve_replay_bundle(
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a finalized replay bundle over HTTP until interrupted.

    A replay bundle is a static browser-viewable directory exported from a
    completed trace.
    """
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
    """Serve a live session directory over HTTP until interrupted.

    A live session may be viewed while exploration is still running or after it
    has been finalized.
    """
    bundle_dir = Path(directory).resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(str(bundle_dir))

    _ensure_debug_browser_index(bundle_dir)
    handler = partial(
        DebugSessionHTTPRequestHandler,
        directory=str(bundle_dir),
        session_directory=str(bundle_dir),
        browser_session_root=str(bundle_dir / "sessions"),
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Serving live debug session at http://{host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping live debug session server.")


def serve_debug_browser(
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the debug browser root with built-in toy-scenario launching enabled."""
    browser_dir = Path(directory).resolve()
    browser_dir.mkdir(parents=True, exist_ok=True)
    _ensure_debug_browser_index(browser_dir)

    handler = partial(
        DebugSessionHTTPRequestHandler,
        directory=str(browser_dir),
        session_directory=None,
        browser_session_root=str(browser_dir / "sessions"),
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Serving debug browser at http://{host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping debug browser server.")


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


def _ensure_debug_browser_index(directory: Path) -> Path:
    """Ensure the self-contained browser viewer exists at ``directory``."""
    index_path = directory / "index.html"
    if not index_path.exists():
        index_path.write_text(render_replay_index_html(), encoding="utf-8")
    return index_path


__all__ = [
    "DebugSessionHTTPRequestHandler",
    "export_and_serve_trace",
    "serve_debug_browser",
    "serve_live_debug_session",
    "serve_replay_bundle",
]
