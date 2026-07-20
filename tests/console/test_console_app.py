import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
from streamlit.testing.v1 import AppTest

from app.agent.state import RunStatus
from app.api.schemas import (
    ApprovalRequestView,
    CreateRunResponse,
    RunEventsPage,
    RunListResponse,
    RunSummaryView,
    RunView,
    TraceEventView,
)
from app.console.client import ApprovalDecision, TaskType
from app.console.state import ConsoleErrorCode, ConsolePhase, ConsoleState

_APP_PATH = Path(__file__).parents[2] / "app" / "console" / "app.py"
_NOW = datetime(2026, 7, 20, 8, 0, tzinfo=UTC)
_DIFF = "--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-old\n+new\n"


class AppFakeClient:
    def __init__(self) -> None:
        self.create_result: CreateRunResponse | Exception = CreateRunResponse(
            run_id="run-main",
            status=RunStatus.PLANNING,
        )
        self.archive_result: RunListResponse | Exception = RunListResponse(runs=[])
        self.event_results: list[RunEventsPage | Exception] = []
        self.pending_results: list[list[ApprovalRequestView] | Exception] = []
        self.detail_results: list[RunView | Exception] = []
        self.decision_results: list[ApprovalRequestView | Exception] = []
        self.calls: list[tuple[str, object]] = []

    def create_run(
        self,
        task_type: TaskType,
        prompt: str,
        repo: str,
        max_steps: int | None = None,
    ) -> CreateRunResponse:
        self.calls.append(("create_run", (task_type, prompt, repo, max_steps)))
        if isinstance(self.create_result, Exception):
            raise self.create_result
        return self.create_result

    def get_run(self, run_id: str) -> RunView:
        self.calls.append(("get_run", run_id))
        result = self.detail_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def list_runs(self) -> RunListResponse:
        self.calls.append(("list_runs", None))
        if isinstance(self.archive_result, Exception):
            raise self.archive_result
        return self.archive_result

    def get_events(self, run_id: str, after_seq: int) -> RunEventsPage:
        self.calls.append(("get_events", (run_id, after_seq)))
        result = self.event_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def list_pending(self, run_id: str) -> list[ApprovalRequestView]:
        self.calls.append(("list_pending", run_id))
        result = self.pending_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def decide(
        self,
        request_id: str,
        decision: ApprovalDecision,
        note: str | None,
    ) -> ApprovalRequestView:
        self.calls.append(("decide", (request_id, decision, note)))
        result = self.decision_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _summary(run_id: str = "run-main", *, status: RunStatus = RunStatus.DONE) -> RunSummaryView:
    return RunSummaryView(
        run_id=run_id,
        task_type="fix",
        prompt="Apply the verified correction.",
        repo="C:/work/repo",
        status=status,
        step_count=2,
        steps_used=2,
        replans_used=0,
        fix_cycles_used=1,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _event(seq: int, *, kind: str = "plan", tool_name: str | None = None) -> TraceEventView:
    return TraceEventView(
        seq=seq,
        ts=_NOW,
        kind=kind,
        summary=None if tool_name else f"Event {seq}",
        tool_name=tool_name,
    )


def _page(
    status: RunStatus,
    events: list[TraceEventView],
    *,
    terminal: bool = False,
    cursor: int | None = None,
) -> RunEventsPage:
    next_cursor = cursor if cursor is not None else max((event.seq for event in events), default=-1)
    return RunEventsPage(
        run_id="run-main",
        status=status,
        terminal=terminal,
        next_cursor=next_cursor,
        events=events,
    )


def _approval(
    *,
    status: str = "pending",
    actor: str | None = None,
    note: str | None = None,
) -> ApprovalRequestView:
    return ApprovalRequestView(
        request_id="approval-main",
        run_id="run-main",
        tool_name="apply_patch",
        risk_level="high",
        args={"rationale": "Correct the verified defect.", "diff": _DIFF},
        status=status,
        actor=actor,
        note=note,
        created_at=_NOW,
        decided_at=_NOW if status != "pending" else None,
    )


def _detail(*, summary: str | None = "Verified correction completed.") -> RunView:
    return RunView(
        **_summary().model_dump(),
        summary=summary,
        tool_calls=[],
    )


def _http_error(status_code: int, *, method: str = "GET") -> httpx.HTTPStatusError:
    request = httpx.Request(method, "http://api.test/resource")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


def _run_app(
    client: AppFakeClient,
    *,
    state: ConsoleState | None = None,
) -> AppTest:
    app = AppTest.from_file(_APP_PATH, default_timeout=5)
    app.session_state["_client"] = client
    app.session_state["_auto_refresh"] = False
    if state is not None:
        app.session_state["_console_state"] = state
    return app.run()


def _click(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"Button {label!r} was not rendered.")


def _input(app: AppTest, label: str, value: str, *, run: bool = False) -> AppTest:
    for widget in app.text_input:
        if widget.label == label:
            widget.input(value)
            return app.run() if run else app
    raise AssertionError(f"Text input {label!r} was not rendered.")


def _markdown_contains(app: AppTest, text: str) -> bool:
    return any(text in str(element.value) for element in app.markdown)


def _state(app: AppTest) -> ConsoleState:
    return cast(ConsoleState, app.session_state["_console_state"])


def test_console_app__full_injected_run_archive_chain() -> None:
    client = AppFakeClient()
    pending = _approval()
    resolved = _approval(status="approved", actor="human", note="Reviewed.")
    client.event_results = [
        _page(RunStatus.EXECUTING, [_event(0), _event(1)]),
        _page(RunStatus.VERIFYING, [_event(1), _event(2)]),
        _page(RunStatus.AWAITING_APPROVAL, [_event(2), _event(3)]),
        _page(RunStatus.AWAITING_APPROVAL, [_event(3)]),
        _page(RunStatus.AWAITING_APPROVAL, [_event(3)]),
        _page(RunStatus.DONE, [_event(3), _event(4)], terminal=True),
    ]
    client.pending_results = [[], [], [pending], [pending], [pending]]
    client.decision_results = [resolved]
    client.detail_results = [_detail()]

    app = _run_app(client)
    assert not app.exception
    assert _state(app).phase is ConsolePhase.EMPTY

    app.selectbox[0].select("fix")
    app.text_area[0].input("Apply the verified correction.")
    _input(app, "Repository path", "C:/work/repo")
    app = _click(app, "Create run")
    assert not app.exception
    assert _state(app).phase is ConsolePhase.RUNNING
    assert [event.seq for event in _state(app).events] == [0, 1]

    app = app.run()
    assert [event.seq for event in _state(app).events] == [0, 1, 2]
    app = app.run()
    assert _state(app).phase is ConsolePhase.AWAITING_APPROVAL
    assert [event.seq for event in _state(app).events] == [0, 1, 2, 3]
    assert [item.value for item in app.code] == [_DIFF.rstrip("\n")]
    assert any(button.label == "Approve" for button in app.button)
    assert any(button.label == "Reject" for button in app.button)

    app = _input(app, "Decision note (optional)", "Reviewed.", run=True)
    app = _click(app, "Approve")
    decided_state = _state(app)
    decision_snapshot = decided_state.decision_records
    assert not app.exception
    assert decided_state.pending == ()
    assert decision_snapshot == (resolved,)
    assert not any(button.label == "Approve" for button in app.button)
    assert ("decide", ("approval-main", "approve", "Reviewed.")) in client.calls

    app = app.run()
    terminal_state = _state(app)
    assert not app.exception
    assert terminal_state.phase is ConsolePhase.TERMINAL
    assert terminal_state.run_detail == _detail()
    assert terminal_state.decision_records == decision_snapshot
    assert [event.seq for event in terminal_state.events] == [0, 1, 2, 3, 4]
    assert _markdown_contains(app, "Verified correction completed.")

    calls_at_terminal = tuple(client.calls)
    app.session_state["_auto_refresh"] = True
    app = app.run(timeout=2)
    assert not app.exception
    assert tuple(client.calls) == calls_at_terminal


def test_console_entry__script_directory_cannot_shadow_app_package() -> None:
    script = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(_APP_PATH.parent)!r}); "
        f"runpy.run_path({str(_APP_PATH)!r}, run_name='console_import_smoke')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_APP_PATH.parents[2],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_console_app__empty_archive_can_select_a_persisted_run() -> None:
    client = AppFakeClient()
    client.archive_result = RunListResponse(runs=[_summary()])
    client.event_results = [_page(RunStatus.DONE, [_event(0)], terminal=True)]
    client.detail_results = [_detail()]
    app = _run_app(client)

    app = _click(app, "DONE · run-main")

    assert not app.exception
    assert _state(app).phase is ConsolePhase.TERMINAL
    assert _state(app).active_run_id == "run-main"
    assert _state(app).run_detail == _detail()


def test_console_app__unreachable_and_not_found_are_safe_page_blocks() -> None:
    request = httpx.Request("GET", "http://api.test/runs")
    unreachable_client = AppFakeClient()
    unreachable_client.archive_result = httpx.ConnectError("offline", request=request)
    unreachable = _run_app(unreachable_client)
    assert not unreachable.exception
    assert _markdown_contains(unreachable, ConsoleErrorCode.API_UNREACHABLE.value)

    missing_client = AppFakeClient()
    missing_client.event_results = [_http_error(404)]
    missing = _run_app(
        missing_client,
        state=ConsoleState(active_run_id="missing", status="PLANNING"),
    )
    assert not missing.exception
    assert _markdown_contains(missing, ConsoleErrorCode.RUN_NOT_FOUND.value)


def test_console_app__bad_repo_is_inline_without_traceback() -> None:
    client = AppFakeClient()
    client.create_result = _http_error(400, method="POST")
    app = _run_app(client)

    app = _click(app, "Create run")

    assert not app.exception
    assert _state(app).phase is ConsolePhase.EMPTY
    assert _markdown_contains(app, ConsoleErrorCode.BAD_REPO.value)


def test_console_app__approval_conflict_refreshes_and_removes_focus() -> None:
    client = AppFakeClient()
    pending = _approval()
    client.event_results = [
        _page(RunStatus.AWAITING_APPROVAL, [_event(0)]),
        _page(RunStatus.AWAITING_APPROVAL, [_event(0)]),
    ]
    client.pending_results = [[pending], [pending], []]
    client.decision_results = [_http_error(409, method="POST")]
    app = _run_app(
        client,
        state=ConsoleState(active_run_id="run-main", status="AWAITING_APPROVAL"),
    )
    assert [item.value for item in app.code] == [_DIFF.rstrip("\n")]

    app = _click(app, "Reject")

    assert not app.exception
    assert _state(app).pending == ()
    assert _markdown_contains(app, ConsoleErrorCode.APPROVAL_CONFLICT.value)
    assert not any(button.label == "Reject" for button in app.button)
    assert [call[0] for call in client.calls].count("list_pending") == 3


def test_console_app__terminal_without_report_is_explicit_and_stops() -> None:
    client = AppFakeClient()
    client.event_results = [_page(RunStatus.FAILED, [], terminal=True, cursor=4)]
    client.detail_results = [_detail(summary=None)]
    app = _run_app(
        client,
        state=ConsoleState(active_run_id="run-main", status="REPORTING", cursor=4),
    )

    assert not app.exception
    assert _state(app).phase is ConsolePhase.TERMINAL
    assert _markdown_contains(app, ConsoleErrorCode.NO_REPORT.value)
    assert _markdown_contains(app, "no report is available")
    calls_at_terminal = tuple(client.calls)

    app = app.run()
    assert not app.exception
    assert tuple(client.calls) == calls_at_terminal
