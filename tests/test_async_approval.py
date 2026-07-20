from collections.abc import Callable
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

from pydantic import BaseModel, ConfigDict

from app.agent.state import AgentState, PlanStep, RunStatus, TaskSpec
from app.safety.async_approval import ApprovalCoordinator, AsyncApprovalGate
from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.schemas.trace import TraceEvent, TraceEventKind
from app.storage.db import ApprovalRequestView, Database
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec

_THREAD_TIMEOUT_S = 5.0
_POLL_INTERVAL_S = 0.01


class _DangerousArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diff: str
    rationale: str
    mode: str = "guarded"


class _DangerousPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    written: bool


def _seed_executing_run(database: Database, run_id: str, repo: Path) -> None:
    database.save_state(
        AgentState(
            run_id=run_id,
            task=TaskSpec(task_type="fix", prompt="Apply the reviewed change.", repo=str(repo)),
            plan=[
                PlanStep(
                    index=0,
                    intent="Apply the proposed patch.",
                    suggested_tools=["dangerous_write"],
                    success_check="The reviewed change is applied.",
                )
            ],
            cursor=0,
            tool_history=[],
            status=RunStatus.EXECUTING,
        )
    )


def _context(run_id: str, repo: Path) -> ToolContext:
    return ToolContext(run_id=run_id, jail=PathJail(repo))


def _spec() -> ToolSpec:
    return ToolSpec(
        name="dangerous_write",
        description="Write a sentinel after approval.",
        args_schema=_DangerousArgs,
        returns_schema=_DangerousPayload,
        risk_level="high",
    )


def _args() -> _DangerousArgs:
    return _DangerousArgs(
        diff="--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-old\n+new\n",
        rationale="Correct the stale value.",
    )


def _start_daemon(
    action: Callable[[], object],
) -> tuple[Thread, list[object], list[Exception]]:
    results: list[object] = []
    errors: list[Exception] = []

    def target() -> None:
        try:
            results.append(action())
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=target, name="approval-test-worker", daemon=True)
    thread.start()
    return thread, results, errors


def _wait_until_parked(
    coordinator: ApprovalCoordinator,
    database: Database,
    store: TraceStore,
    run_id: str,
) -> ApprovalRequestView:
    deadline = monotonic() + _THREAD_TIMEOUT_S
    while monotonic() < deadline:
        pending = coordinator.list_pending(run_id)
        run = database.get_run(run_id)
        events = store.read(run_id)
        if (
            pending
            and run is not None
            and run.status is RunStatus.AWAITING_APPROVAL
            and any(event.kind is TraceEventKind.approval_request for event in events)
        ):
            assert len(pending) == 1
            return pending[0]
        sleep(_POLL_INTERVAL_S)
    raise AssertionError(f"Approval worker for run {run_id!r} did not park in time.")


def _assert_thread_finished(thread: Thread, errors: list[Exception]) -> None:
    thread.join(timeout=_THREAD_TIMEOUT_S)
    assert not thread.is_alive(), "Approval worker remained parked after a decision."
    assert errors == []


def _assert_request_event(event: TraceEvent, request: ApprovalRequestView) -> None:
    assert event.kind is TraceEventKind.approval_request
    assert event.payload == {
        "request_id": request.request_id,
        "tool_name": "dangerous_write",
        "risk_level": "high",
    }


def test_approval_coordinator__approve_round_trip_parks_and_wakes_worker(
    tmp_path: Path,
) -> None:
    run_id = "run-coordinator-approve"
    database = Database(tmp_path / "runs.sqlite")
    store = TraceStore(tmp_path / "traces")
    coordinator = ApprovalCoordinator(database, store)
    _seed_executing_run(database, run_id, tmp_path)
    args = _args()

    thread, results, errors = _start_daemon(
        lambda: coordinator.request(_spec(), args, _context(run_id, tmp_path))
    )
    pending = _wait_until_parked(coordinator, database, store, run_id)

    assert thread.is_alive()
    assert pending.run_id == run_id
    assert pending.tool_name == "dangerous_write"
    assert pending.risk_level == "high"
    assert pending.args == args.model_dump(mode="json")
    assert pending.status == "pending"
    assert pending.actor is None
    assert pending.note is None
    assert pending.decided_at is None
    assert coordinator.list_pending("another-run") == []
    run = database.get_run(run_id)
    assert run is not None
    assert run.status is RunStatus.AWAITING_APPROVAL
    [request_event] = store.read(run_id)
    _assert_request_event(request_event, pending)

    decided = coordinator.decide(
        pending.request_id,
        approved=True,
        actor="reviewer:api",
        note="Reviewed and approved.",
    )
    _assert_thread_finished(thread, errors)

    assert decided.status == "approved"
    assert decided.actor == "reviewer:api"
    assert decided.note == "Reviewed and approved."
    assert decided.decided_at is not None
    assert coordinator.list_pending(run_id) == []
    assert results == [
        ApprovalOutcome(
            approved=True,
            reason="Reviewed and approved.",
            actor="reviewer:api",
        )
    ]
    run_after_decision = database.get_run(run_id)
    assert run_after_decision is not None
    assert run_after_decision.status is RunStatus.AWAITING_APPROVAL
    assert store.read(run_id) == [request_event]


def test_approval_coordinator__deny_round_trip_preserves_note(
    tmp_path: Path,
) -> None:
    run_id = "run-coordinator-deny"
    database = Database(tmp_path / "runs.sqlite")
    store = TraceStore(tmp_path / "traces")
    coordinator = ApprovalCoordinator(database, store)
    _seed_executing_run(database, run_id, tmp_path)

    thread, results, errors = _start_daemon(
        lambda: coordinator.request(_spec(), _args(), _context(run_id, tmp_path))
    )
    pending = _wait_until_parked(coordinator, database, store, run_id)
    run = database.get_run(run_id)
    assert run is not None
    assert run.status is RunStatus.AWAITING_APPROVAL
    [request_event] = store.read(run_id)
    _assert_request_event(request_event, pending)

    decided = coordinator.decide(
        pending.request_id,
        approved=False,
        actor=None,
        note="Add a regression test first.",
    )
    _assert_thread_finished(thread, errors)

    assert decided.status == "denied"
    assert decided.actor is None
    assert decided.note == "Add a regression test first."
    assert results == [
        ApprovalOutcome(
            approved=False,
            reason="Add a regression test first.",
            actor="human",
        )
    ]
    run_after_decision = database.get_run(run_id)
    assert run_after_decision is not None
    assert run_after_decision.status is RunStatus.AWAITING_APPROVAL
    assert store.read(run_id) == [request_event]


def test_async_approval_gate__registry_drop_in_approve_executes_and_traces(
    tmp_path: Path,
) -> None:
    run_id = "run-registry-approve"
    database = Database(tmp_path / "runs.sqlite")
    store = TraceStore(tmp_path / "traces")
    coordinator = ApprovalCoordinator(database, store)
    _seed_executing_run(database, run_id, tmp_path)
    sentinel: list[str] = []
    registry = ToolRegistry(
        approval_gate=AsyncApprovalGate(coordinator),
        trace_sink=RegistryTraceSink(store),
    )

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _DangerousArgs.model_validate(args)
        sentinel.append(parsed.diff)
        return _DangerousPayload(written=True)

    registry.register(_spec(), handler)
    raw_args = _args().model_dump(mode="json", exclude_defaults=True)
    thread, results, errors = _start_daemon(
        lambda: registry.dispatch("dangerous_write", raw_args, _context(run_id, tmp_path))
    )
    pending = _wait_until_parked(coordinator, database, store, run_id)

    assert thread.is_alive()
    assert sentinel == []
    assert pending.args == _args().model_dump(mode="json")
    [request_event] = store.read(run_id)
    _assert_request_event(request_event, pending)

    coordinator.decide(
        pending.request_id,
        approved=True,
        actor="reviewer:api",
        note="Safe to apply.",
    )
    _assert_thread_finished(thread, errors)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data == _DangerousPayload(written=True)
    assert result.error is None
    assert sentinel == [_args().diff]
    events = store.read(run_id)
    assert [event.kind for event in events] == [
        TraceEventKind.approval_request,
        TraceEventKind.approval_decision,
        TraceEventKind.tool_call,
    ]
    _assert_request_event(events[0], pending)
    assert events[1].payload == {
        "tool_name": "dangerous_write",
        "risk_level": "high",
        "decision": "approved",
        "actor": "reviewer:api",
        "reason": "Safe to apply.",
    }
    assert events[2].payload["tool_name"] == "dangerous_write"
    assert events[2].payload["ok"] is True
    assert events[2].payload["error_type"] is None


def test_async_approval_gate__registry_drop_in_deny_blocks_and_traces(
    tmp_path: Path,
) -> None:
    run_id = "run-registry-deny"
    database = Database(tmp_path / "runs.sqlite")
    store = TraceStore(tmp_path / "traces")
    coordinator = ApprovalCoordinator(database, store)
    _seed_executing_run(database, run_id, tmp_path)
    sentinel: list[str] = []
    registry = ToolRegistry(
        approval_gate=AsyncApprovalGate(coordinator),
        trace_sink=RegistryTraceSink(store),
    )

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _DangerousArgs.model_validate(args)
        sentinel.append(parsed.diff)
        return _DangerousPayload(written=True)

    registry.register(_spec(), handler)
    raw_args = _args().model_dump(mode="json", exclude_defaults=True)
    thread, results, errors = _start_daemon(
        lambda: registry.dispatch("dangerous_write", raw_args, _context(run_id, tmp_path))
    )
    pending = _wait_until_parked(coordinator, database, store, run_id)
    [request_event] = store.read(run_id)
    _assert_request_event(request_event, pending)

    coordinator.decide(
        pending.request_id,
        approved=False,
        actor="reviewer:api",
        note="The diff needs another test.",
    )
    _assert_thread_finished(thread, errors)

    assert sentinel == []
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.type is ErrorType.ApprovalDeniedError
    assert result.error.message == "The diff needs another test."
    assert result.error.details == {
        "risk_level": "high",
        "reason": "The diff needs another test.",
    }
    events = store.read(run_id)
    assert [event.kind for event in events] == [
        TraceEventKind.approval_request,
        TraceEventKind.approval_decision,
        TraceEventKind.tool_call,
    ]
    _assert_request_event(events[0], pending)
    assert events[1].payload == {
        "tool_name": "dangerous_write",
        "risk_level": "high",
        "decision": "denied",
        "actor": "reviewer:api",
        "reason": "The diff needs another test.",
    }
    assert events[2].payload["tool_name"] == "dangerous_write"
    assert events[2].payload["ok"] is False
    assert events[2].payload["error_type"] == ErrorType.ApprovalDeniedError.value
