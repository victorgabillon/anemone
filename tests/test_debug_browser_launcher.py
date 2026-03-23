"""Tests for the in-browser toy scenario launcher backend."""

# ruff: noqa: D103

from __future__ import annotations

import json
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any

import pytest
from graphviz.backend.execute import ExecutableNotFound

import anemone.debug.export as debug_export
from anemone.debug.browser import DebugSessionHTTPRequestHandler
from anemone.debug.toy_scenarios import (
    list_registered_scenarios,
    run_registered_scenario,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_list_registered_scenarios_exposes_expected_names() -> None:
    assert [scenario.name for scenario in list_registered_scenarios()] == [
        "single_agent_backup",
        "minimax_micro",
        "deceptive_trap",
        "minimax_semantic_stress",
    ]


def test_run_registered_scenario_creates_stable_session_output(tmp_path: Path) -> None:
    run_result = run_registered_scenario(
        "single_agent_backup",
        output_root_directory=tmp_path / "sessions",
        snapshot_format="dot",
    )

    assert run_result.ok is True
    assert run_result.session_path == "/sessions/single_agent_backup/"
    assert run_result.session_directory is not None
    session_directory = tmp_path / "sessions" / "single_agent_backup"
    assert run_result.session_directory == str(session_directory)
    assert (session_directory / "index.html").exists()
    assert (session_directory / "session.json").exists()


def test_run_registered_scenario_rejects_unknown_scenario(tmp_path: Path) -> None:
    run_result = run_registered_scenario(
        "does_not_exist",
        output_root_directory=tmp_path / "sessions",
    )

    assert run_result.ok is False
    assert run_result.error_code == "unknown_scenario"
    assert run_result.session_directory is None
    assert run_result.session_path is None


def test_debug_browser_api_lists_registered_scenarios(tmp_path: Path) -> None:
    with _running_browser_server(tmp_path) as address:
        status_code, payload = _request_json(
            address,
            "GET",
            "/api/scenarios",
        )

    assert status_code == 200
    assert [scenario["name"] for scenario in payload["scenarios"]] == [
        "single_agent_backup",
        "minimax_micro",
        "deceptive_trap",
        "minimax_semantic_stress",
    ]


def test_debug_browser_api_runs_registered_scenario(tmp_path: Path) -> None:
    with _running_browser_server(tmp_path) as address:
        status_code, payload = _request_json(
            address,
            "POST",
            "/api/run_scenario",
            {"name": "single_agent_backup"},
        )
        session_status, session_payload = _request_json(
            address,
            "GET",
            "/sessions/single_agent_backup/session.json",
        )

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["session_path"] == "/sessions/single_agent_backup/"
    assert session_status == 200
    assert session_payload["is_live"] is True
    assert session_payload["is_complete"] is True
    assert session_payload["entry_count"] > 0


def test_debug_browser_api_rejects_unknown_scenario_name(tmp_path: Path) -> None:
    with _running_browser_server(tmp_path) as address:
        status_code, payload = _request_json(
            address,
            "POST",
            "/api/run_scenario",
            {"name": "unknown_scenario"},
        )

    assert status_code == 404
    assert payload["ok"] is False
    assert payload["error_code"] == "unknown_scenario"


def test_debug_browser_api_runs_scenario_without_graphviz_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRenderGraph:
        source = "digraph {}"

        def render(self, **_: object) -> str:
            raise ExecutableNotFound(["dot"])

    monkeypatch.setattr(
        debug_export,
        "render_snapshot",
        lambda snapshot, format_str=None: _FailingRenderGraph(),
    )

    with _running_browser_server(tmp_path) as address:
        status_code, payload = _request_json(
            address,
            "POST",
            "/api/run_scenario",
            {"name": "single_agent_backup"},
        )
        session_status, session_payload = _request_json(
            address,
            "GET",
            "/sessions/single_agent_backup/session.json",
        )

    assert status_code == 200
    assert payload["ok"] is True
    assert session_status == 200
    assert session_payload["is_complete"] is True
    assert session_payload["entry_count"] > 0
    assert any(
        entry["snapshot_metadata_file"] is not None
        for entry in session_payload["entries"]
    )
    assert all(entry["snapshot_file"] is None for entry in session_payload["entries"])


def test_debug_browser_api_can_launch_two_scenarios_and_control_each(
    tmp_path: Path,
) -> None:
    with _running_browser_server(tmp_path) as address:
        first_status, first_payload = _request_json(
            address,
            "POST",
            "/api/run_scenario",
            {"name": "single_agent_backup"},
        )
        second_status, second_payload = _request_json(
            address,
            "POST",
            "/api/run_scenario",
            {"name": "minimax_micro"},
        )
        first_session_status, first_session_payload = _request_json(
            address,
            "GET",
            "/sessions/single_agent_backup/session.json",
        )
        second_session_status, second_session_payload = _request_json(
            address,
            "GET",
            "/sessions/minimax_micro/session.json",
        )
        pause_status, _pause_body = _request(
            address,
            "POST",
            "/sessions/single_agent_backup/command",
            {"command": "pause"},
        )
        control_status, control_payload = _request_json(
            address,
            "GET",
            "/sessions/single_agent_backup/control_state.json",
        )

    assert first_status == 200
    assert second_status == 200
    assert first_payload["session_path"] == "/sessions/single_agent_backup/"
    assert second_payload["session_path"] == "/sessions/minimax_micro/"
    assert first_session_status == 200
    assert second_session_status == 200
    assert first_session_payload["is_complete"] is True
    assert second_session_payload["is_complete"] is True
    assert pause_status == 204
    assert control_status == 200
    assert control_payload["paused"] is True


class _BrowserServerContext:
    def __init__(self, root_directory: Path) -> None:
        self._root_directory = root_directory
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> tuple[str, int]:
        handler = partial(
            DebugSessionHTTPRequestHandler,
            directory=str(self._root_directory),
            session_directory=None,
            browser_session_root=str(self._root_directory / "sessions"),
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        assert self._server is not None
        assert self._thread is not None
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _running_browser_server(root_directory: Path) -> _BrowserServerContext:
    root_directory.mkdir(parents=True, exist_ok=True)
    return _BrowserServerContext(root_directory)


def _request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    status_code, response_body = _request(address, method, path, payload)
    return status_code, json.loads(response_body)


def _request(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, str]:
    connection = HTTPConnection(address[0], address[1], timeout=10)
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    response_body = response.read().decode("utf-8")
    connection.close()
    return response.status, response_body
