"""Immutable console state and one-poll state transitions."""

from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum

import httpx

from app.api.schemas import (
    ApprovalRequestView,
    RunSummaryView,
    RunView,
    TraceEventView,
)
from app.console.client import ApprovalDecision, ConsoleClientLike, TaskType


class ConsolePhase(StrEnum):
    """Four presentation phases derived from explicit HTTP state."""

    EMPTY = "EMPTY"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    TERMINAL = "TERMINAL"


class ConsoleErrorCode(StrEnum):
    """Finite page-level error vocabulary exposed by the console."""

    API_UNREACHABLE = "API_UNREACHABLE"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    APPROVAL_CONFLICT = "APPROVAL_CONFLICT"
    NO_REPORT = "NO_REPORT"
    BAD_REPO = "BAD_REPO"
    API_ERROR = "API_ERROR"


@dataclass(frozen=True, slots=True)
class ConsoleError:
    """Safe user-facing error state without exception internals."""

    code: ConsoleErrorCode
    message: str
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class ConsoleState:
    """Complete session projection used to redraw the console."""

    active_run_id: str | None = None
    cursor: int = -1
    events: tuple[TraceEventView, ...] = ()
    pending: tuple[ApprovalRequestView, ...] = ()
    decision_records: tuple[ApprovalRequestView, ...] = ()
    history: tuple[RunSummaryView, ...] = ()
    status: str | None = None
    terminal: bool = False
    run_detail: RunView | None = None
    task_type: str | None = None
    prompt: str | None = None
    repo: str | None = None
    page_error: ConsoleError | None = None
    form_error: ConsoleError | None = None

    @property
    def phase(self) -> ConsolePhase:
        """Return the UI phase without guessing lifecycle from trace events."""
        if self.active_run_id is None:
            return ConsolePhase.EMPTY
        if self.terminal:
            return ConsolePhase.TERMINAL
        if self.pending:
            return ConsolePhase.AWAITING_APPROVAL
        return ConsolePhase.RUNNING


def advance(state: ConsoleState, client: ConsoleClientLike) -> ConsoleState:
    """Poll the HTTP API once and return a new session projection."""
    if state.terminal:
        return state

    working = replace(state, page_error=None)
    run_id = working.active_run_id
    if run_id is None:
        try:
            archive = client.list_runs()
        except Exception as exc:
            return replace(working, page_error=console_error_from(exc))
        return replace(working, history=tuple(archive.runs))

    try:
        page = client.get_events(run_id, working.cursor)
    except Exception as exc:
        return replace(working, page_error=console_error_from(exc))

    events, cursor = _merge_events(working, page.events, page.next_cursor)
    working = replace(
        working,
        cursor=cursor,
        events=events,
        status=page.status.value,
        terminal=page.terminal,
    )

    if page.terminal:
        working = replace(working, pending=())
        try:
            detail = client.get_run(run_id)
        except Exception as exc:
            return replace(working, page_error=console_error_from(exc))
        if detail.summary is None:
            return replace(
                working,
                run_detail=detail,
                page_error=ConsoleError(
                    code=ConsoleErrorCode.NO_REPORT,
                    message=f"Run is {working.status}, but no report was persisted.",
                ),
            )
        return replace(working, run_detail=detail)

    try:
        pending = client.list_pending(run_id)
    except Exception as exc:
        return replace(working, page_error=console_error_from(exc))
    return replace(working, pending=tuple(pending))


def start_run(
    state: ConsoleState,
    client: ConsoleClientLike,
    *,
    task_type: TaskType,
    prompt: str,
    repo: str,
) -> ConsoleState:
    """Create one run while converting every client failure into form state."""
    try:
        created = client.create_run(task_type, prompt, repo)
    except Exception as exc:
        error = console_error_from(exc)
        return replace(state, form_error=error, page_error=None)
    return ConsoleState(
        active_run_id=created.run_id,
        history=state.history,
        status=created.status.value,
        task_type=task_type,
        prompt=prompt,
        repo=repo,
    )


def select_run(state: ConsoleState, run_id: str) -> ConsoleState:
    """Select one archive entry and reset all run-local projections."""
    summary = next((item for item in state.history if item.run_id == run_id), None)
    return ConsoleState(
        active_run_id=run_id,
        history=state.history,
        status=summary.status.value if summary is not None else None,
        task_type=summary.task_type if summary is not None else None,
        prompt=summary.prompt if summary is not None else None,
        repo=summary.repo if summary is not None else None,
    )


def reset_to_empty(state: ConsoleState) -> ConsoleState:
    """Return to the creation form while retaining the current archive index."""
    return ConsoleState(history=state.history)


def decide_approval(
    state: ConsoleState,
    client: ConsoleClientLike,
    *,
    request_id: str,
    decision: ApprovalDecision,
    note: str | None,
) -> ConsoleState:
    """Resolve one approval or reconcile a concurrent decision conflict."""
    normalized_note = note.strip() if note and note.strip() else None
    try:
        resolved = client.decide(request_id, decision, normalized_note)
    except Exception as exc:
        error = console_error_from(exc)
        if error.code is not ConsoleErrorCode.APPROVAL_CONFLICT:
            return replace(state, page_error=error)
        remaining = tuple(item for item in state.pending if item.request_id != request_id)
        if state.active_run_id is not None:
            with suppress(Exception):
                remaining = tuple(client.list_pending(state.active_run_id))
        return replace(state, pending=remaining, page_error=error)

    remaining = tuple(item for item in state.pending if item.request_id != request_id)
    records = (
        *(item for item in state.decision_records if item.request_id != resolved.request_id),
        resolved,
    )
    return replace(
        state,
        pending=remaining,
        decision_records=records,
        page_error=None,
    )


def console_error_from(exc: Exception) -> ConsoleError:
    """Map an arbitrary client failure into the finite safe error vocabulary."""
    if isinstance(exc, (httpx.TransportError, httpx.RequestError)):
        return ConsoleError(
            code=ConsoleErrorCode.API_UNREACHABLE,
            message="The RepoPilot API is unreachable. The last known archive state is retained.",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 404:
            return ConsoleError(
                code=ConsoleErrorCode.RUN_NOT_FOUND,
                message="The requested run or approval record was not found.",
                status_code=status_code,
            )
        if status_code == 409:
            return ConsoleError(
                code=ConsoleErrorCode.APPROVAL_CONFLICT,
                message="This approval was decided elsewhere; pending requests were refreshed.",
                status_code=status_code,
            )
        if status_code == 400:
            return ConsoleError(
                code=ConsoleErrorCode.BAD_REPO,
                message="The repository path was rejected by the API.",
                status_code=status_code,
            )
        return ConsoleError(
            code=ConsoleErrorCode.API_ERROR,
            message=f"The RepoPilot API returned HTTP {status_code}.",
            status_code=status_code,
        )
    return ConsoleError(
        code=ConsoleErrorCode.API_ERROR,
        message="The console could not validate the API response.",
    )


def _merge_events(
    state: ConsoleState,
    page_events: list[TraceEventView],
    page_cursor: int,
) -> tuple[tuple[TraceEventView, ...], int]:
    fresh = sorted(
        (event for event in page_events if event.seq > state.cursor),
        key=lambda event: event.seq,
    )
    unique: list[TraceEventView] = []
    last_seq = state.cursor
    for event in fresh:
        if event.seq > last_seq:
            unique.append(event)
            last_seq = event.seq
    cursor = max(state.cursor, page_cursor, last_seq)
    return state.events + tuple(unique), cursor
