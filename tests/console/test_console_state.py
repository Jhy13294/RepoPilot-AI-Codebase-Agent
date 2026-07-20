from datetime import UTC, datetime

import httpx

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
from app.console.state import (
    ConsoleErrorCode,
    ConsolePhase,
    ConsoleState,
    advance,
    decide_approval,
    start_run,
)

_NOW = datetime(2026, 7, 20, 8, 0, tzinfo=UTC)


class FakeConsoleClient:
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


def _summary(run_id: str = "run-old") -> RunSummaryView:
    return RunSummaryView(
        run_id=run_id,
        task_type="issue",
        prompt="Locate the defect.",
        repo="C:/work/repo",
        status=RunStatus.DONE,
        step_count=2,
        steps_used=2,
        replans_used=0,
        fix_cycles_used=0,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _event(seq: int, *, kind: str = "plan") -> TraceEventView:
    return TraceEventView(
        seq=seq,
        ts=_NOW,
        kind=kind,
        summary=f"Event {seq}",
    )


def _page(
    status: RunStatus,
    events: list[TraceEventView],
    *,
    terminal: bool = False,
    next_cursor: int | None = None,
) -> RunEventsPage:
    cursor = (
        next_cursor if next_cursor is not None else max((item.seq for item in events), default=-1)
    )
    return RunEventsPage(
        run_id="run-main",
        status=status,
        terminal=terminal,
        next_cursor=cursor,
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
        args={
            "rationale": "Correct the verified defect.",
            "diff": "--- a/value.py\n+++ b/value.py\n-old\n+new\n",
        },
        status=status,
        actor=actor,
        note=note,
        created_at=_NOW,
        decided_at=_NOW if status != "pending" else None,
    )


def _detail(*, summary: str | None = "Completed report.") -> RunView:
    return RunView(
        **_summary("run-main").model_dump(),
        summary=summary,
        tool_calls=[],
    )


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://api.test/resource")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


def test_console_state__four_phase_chain_deduplicates_and_stops_at_terminal() -> None:
    client = FakeConsoleClient()
    client.archive_result = RunListResponse(runs=[_summary()])
    pending = _approval()
    resolved = _approval(status="approved", actor="human", note="Reviewed.")
    client.event_results = [
        _page(RunStatus.EXECUTING, [_event(0), _event(1)]),
        _page(RunStatus.VERIFYING, [_event(1), _event(2)]),
        _page(RunStatus.AWAITING_APPROVAL, [_event(2), _event(3)]),
        _page(RunStatus.DONE, [_event(3), _event(4)], terminal=True),
    ]
    client.pending_results = [[], [], [pending]]
    client.decision_results = [resolved]
    client.detail_results = [_detail()]

    state = advance(ConsoleState(), client)
    assert state.phase is ConsolePhase.EMPTY
    assert [item.run_id for item in state.history] == ["run-old"]

    state = start_run(
        state,
        client,
        task_type="issue",
        prompt="Fix the defect.",
        repo="C:/work/repo",
    )
    assert state.phase is ConsolePhase.RUNNING
    assert state.status == "PLANNING"

    state = advance(state, client)
    state = advance(state, client)
    state = advance(state, client)
    assert state.phase is ConsolePhase.AWAITING_APPROVAL
    assert state.pending == (pending,)
    assert [event.seq for event in state.events] == [0, 1, 2, 3]

    state = decide_approval(
        state,
        client,
        request_id=pending.request_id,
        decision="approve",
        note=" Reviewed. ",
    )
    decision_snapshot = state.decision_records
    assert state.pending == ()
    assert state.decision_records == (resolved,)

    state = advance(state, client)
    assert state.phase is ConsolePhase.TERMINAL
    assert state.status == "DONE"
    assert state.run_detail == _detail()
    assert state.decision_records == decision_snapshot
    assert [event.seq for event in state.events] == [0, 1, 2, 3, 4]
    assert [value[1][1] for value in client.calls if value[0] == "get_events"] == [
        -1,
        1,
        2,
        3,
    ]

    calls_at_terminal = tuple(client.calls)
    terminal_state = advance(state, client)
    assert terminal_state is state
    assert tuple(client.calls) == calls_at_terminal


def test_advance__overlapping_pages_keep_strictly_increasing_unique_events() -> None:
    client = FakeConsoleClient()
    client.event_results = [
        _page(RunStatus.EXECUTING, [_event(0), _event(1)]),
        _page(RunStatus.EXECUTING, [_event(0), _event(1), _event(2)]),
        _page(RunStatus.VERIFYING, [_event(1), _event(2), _event(3)]),
    ]
    client.pending_results = [[], [], []]
    state = ConsoleState(active_run_id="run-main", status="PLANNING")

    for _ in range(3):
        state = advance(state, client)

    assert [event.seq for event in state.events] == [0, 1, 2, 3]
    assert state.cursor == 3


def test_advance__maps_transport_and_not_found_without_losing_known_state() -> None:
    request = httpx.Request("GET", "http://api.test/runs/run-main/events")
    transport_client = FakeConsoleClient()
    transport_client.event_results = [httpx.ConnectError("offline", request=request)]
    known = ConsoleState(active_run_id="run-main", status="EXECUTING", events=(_event(0),))

    unreachable = advance(known, transport_client)
    assert unreachable.page_error is not None
    assert unreachable.page_error.code is ConsoleErrorCode.API_UNREACHABLE
    assert unreachable.events == known.events

    missing_client = FakeConsoleClient()
    missing_client.event_results = [_http_error(404)]
    missing = advance(known, missing_client)
    assert missing.page_error is not None
    assert missing.page_error.code is ConsoleErrorCode.RUN_NOT_FOUND
    assert missing.events == known.events


def test_start_run__bad_repository_is_an_inline_finite_error() -> None:
    client = FakeConsoleClient()
    client.create_result = _http_error(400)

    state = start_run(
        ConsoleState(),
        client,
        task_type="question",
        prompt="Where is the parser?",
        repo="C:/missing",
    )

    assert state.phase is ConsolePhase.EMPTY
    assert state.form_error is not None
    assert state.form_error.code is ConsoleErrorCode.BAD_REPO


def test_decide_approval__conflict_refreshes_pending_and_collapses_request() -> None:
    client = FakeConsoleClient()
    client.decision_results = [_http_error(409)]
    client.pending_results = [[]]
    state = ConsoleState(
        active_run_id="run-main",
        status="AWAITING_APPROVAL",
        pending=(_approval(),),
    )

    updated = decide_approval(
        state,
        client,
        request_id="approval-main",
        decision="deny",
        note=None,
    )

    assert updated.pending == ()
    assert updated.page_error is not None
    assert updated.page_error.code is ConsoleErrorCode.APPROVAL_CONFLICT
    assert ("list_pending", "run-main") in client.calls


def test_advance__terminal_without_summary_exposes_no_report_state() -> None:
    client = FakeConsoleClient()
    client.event_results = [_page(RunStatus.FAILED, [], terminal=True, next_cursor=7)]
    client.detail_results = [_detail(summary=None)]
    state = ConsoleState(active_run_id="run-main", status="REPORTING", cursor=7)

    terminal = advance(state, client)

    assert terminal.phase is ConsolePhase.TERMINAL
    assert terminal.run_detail is not None
    assert terminal.run_detail.summary is None
    assert terminal.page_error is not None
    assert terminal.page_error.code is ConsoleErrorCode.NO_REPORT
    assert [call[0] for call in client.calls] == ["get_events", "get_run"]
