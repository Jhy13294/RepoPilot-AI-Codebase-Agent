from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent.state import AgentState, PlanStep, RunStatus, TaskSpec
from app.schemas.tool_io import ErrorType
from app.schemas.trace import TraceEvent, TraceEventKind
from app.storage.db import Database, RunSummary, ToolCallView
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.registry import ToolTraceRecord


def _task(repo: str = ".") -> TaskSpec:
    return TaskSpec(task_type="question", prompt="Where is parse_date defined?", repo=repo)


def _step(index: int, *, intent: str | None = None) -> PlanStep:
    return PlanStep(
        index=index,
        intent=intent or f"Find evidence part {index}.",
        suggested_tools=["search_code", "read_file"],
        success_check="A file:line citation identifies the answer.",
    )


def _trace_record(run_id: str = "run-1") -> ToolTraceRecord:
    return ToolTraceRecord(
        run_id=run_id,
        tool_name="search_code",
        args={"query": "def parse_date"},
        ok=True,
        error_type=None,
        latency_ms=7,
        truncated=False,
        ts=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )


def _agent_state(run_id: str = "run-1", **overrides: object) -> AgentState:
    values = {
        "run_id": run_id,
        "task": _task(),
        "plan": [_step(0), _step(1)],
        "cursor": 1,
        "tool_history": [_trace_record(run_id)],
        "scratchpad": "parse_date likely lives in dates.py",
        "status": RunStatus.EXECUTING,
        "steps_used": 1,
        "replans_used": 0,
        "fix_cycles_used": 0,
    }
    values.update(overrides)
    return AgentState.model_validate(values)


def _tool_event(
    run_id: str,
    seq: int,
    *,
    tool_name: str = "read_file",
    ok: bool = True,
    error_type: str | None = None,
    truncated: bool = False,
    ts: datetime | None = None,
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        seq=seq,
        ts=ts or datetime(2026, 2, 3, 4, 5, seq, tzinfo=UTC),
        kind=TraceEventKind.tool_call,
        payload={
            "tool_name": tool_name,
            "args": {"path": "src/sample_pkg/dates.py", "start_line": seq + 1},
            "ok": ok,
            "error_type": error_type,
            "truncated": truncated,
        },
        latency_ms=10 + seq,
    )


def _plan_event(run_id: str, seq: int) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        seq=seq,
        ts=datetime(2026, 2, 3, 4, 5, seq, tzinfo=UTC),
        kind=TraceEventKind.plan,
        payload={"summary": "plan ready"},
    )


def test_database__save_and_load_agent_state_round_trips(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "runs.sqlite"
    database = Database(db_path)
    state = _agent_state()

    database.save_state(state)

    assert db_path.exists()
    assert database.load_state(state.run_id) == state
    assert database.load_state("missing-run") is None

    summaries = database.list_runs()
    assert summaries == [
        RunSummary(
            run_id="run-1",
            task_type="question",
            prompt=state.task.prompt,
            repo=state.task.repo,
            status=RunStatus.EXECUTING,
            cursor=1,
            step_count=2,
            steps_used=1,
            replans_used=0,
            fix_cycles_used=0,
            created_at=summaries[0].created_at,
            updated_at=summaries[0].updated_at,
        )
    ]
    assert summaries[0].created_at.tzinfo is UTC
    assert summaries[0].updated_at.tzinfo is UTC


def test_database__save_state_rebuilds_step_projection(tmp_path: Path) -> None:
    database = Database(tmp_path / "runs.sqlite")
    state = _agent_state()
    updated = _agent_state(plan=[_step(0, intent="Replacement plan.")], status=RunStatus.VERIFYING)

    database.save_state(state)
    database.save_state(updated)

    assert database.load_state("run-1") == updated
    [summary] = database.list_runs()
    assert summary.status is RunStatus.VERIFYING
    assert summary.step_count == 1


def test_database__dtos_are_frozen(tmp_path: Path) -> None:
    database = Database(tmp_path / "runs.sqlite")
    database.save_state(_agent_state())
    database.index_events("run-1", [_tool_event("run-1", 0)])

    run_summary = database.list_runs()[0]
    tool_call = database.tool_calls("run-1")[0]

    with pytest.raises(ValidationError):
        run_summary.status = RunStatus.DONE

    with pytest.raises(ValidationError):
        tool_call.seq = 2


def test_database__tool_call_index_is_rebuildable_from_same_events(tmp_path: Path) -> None:
    db_path = tmp_path / "trace-index.sqlite"
    events = [
        _plan_event("run-1", 0),
        _tool_event("run-1", 1, tool_name="search_code"),
        _tool_event(
            "run-1",
            2,
            ok=False,
            error_type=ErrorType.NotFoundError.value,
            truncated=True,
        ),
    ]
    first_database = Database(db_path)
    first_database.index_events("run-1", events)
    first_rows = first_database.tool_calls("run-1")

    restarted_database = Database(db_path)
    restarted_database.index_events("run-1", events)

    assert restarted_database.tool_calls("run-1") == first_rows
    assert first_rows == [
        ToolCallView(
            run_id="run-1",
            seq=1,
            ts=datetime(2026, 2, 3, 4, 5, 1, tzinfo=UTC),
            tool_name="search_code",
            args={"path": "src/sample_pkg/dates.py", "start_line": 2},
            ok=True,
            error_type=None,
            latency_ms=11,
            truncated=False,
            payload={
                "tool_name": "search_code",
                "args": {"path": "src/sample_pkg/dates.py", "start_line": 2},
                "ok": True,
                "error_type": None,
                "truncated": False,
            },
        ),
        ToolCallView(
            run_id="run-1",
            seq=2,
            ts=datetime(2026, 2, 3, 4, 5, 2, tzinfo=UTC),
            tool_name="read_file",
            args={"path": "src/sample_pkg/dates.py", "start_line": 3},
            ok=False,
            error_type="NotFoundError",
            latency_ms=12,
            truncated=True,
            payload={
                "tool_name": "read_file",
                "args": {"path": "src/sample_pkg/dates.py", "start_line": 3},
                "ok": False,
                "error_type": "NotFoundError",
                "truncated": True,
            },
        ),
    ]


def test_database__index_events_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "trace-index.sqlite")
    events = [_tool_event("run-idempotent", 0), _tool_event("run-idempotent", 1)]

    database.index_events("run-idempotent", events)
    database.index_events("run-idempotent", events)

    rows = database.tool_calls("run-idempotent")
    assert [row.seq for row in rows] == [0, 1]


def test_database__runs_and_tool_calls_are_isolated_by_run_id(tmp_path: Path) -> None:
    database = Database(tmp_path / "trace-index.sqlite")
    database.save_state(_agent_state("run-a", task=_task(repo="repo-a")))
    database.save_state(_agent_state("run-b", task=_task(repo="repo-b")))
    mixed_events = [
        _tool_event("run-a", 0, tool_name="search_code"),
        _tool_event("run-b", 0, tool_name="read_file"),
        _tool_event("run-a", 1, tool_name="get_file_tree"),
    ]

    database.index_events("run-a", mixed_events)
    database.index_events("run-b", mixed_events)

    assert [summary.run_id for summary in database.list_runs()] == ["run-a", "run-b"]
    assert [row.tool_name for row in database.tool_calls("run-a")] == [
        "search_code",
        "get_file_tree",
    ]
    assert [row.tool_name for row in database.tool_calls("run-b")] == ["read_file"]

    database.index_events("run-a", [_tool_event("run-a", 2, tool_name="read_file")])

    assert [row.seq for row in database.tool_calls("run-a")] == [2]
    assert [row.seq for row in database.tool_calls("run-b")] == [0]
    assert database.tool_calls("missing-run") == []


def test_database__tool_call_view_normalizes_naive_sqlite_timestamp_to_utc(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "trace-index.sqlite")
    naive_ts = datetime(2026, 2, 3, 4, 5, 6)

    database.index_events("run-time", [_tool_event("run-time", 0, ts=naive_ts)])

    [tool_call] = database.tool_calls("run-time")
    assert tool_call.ts == datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    assert tool_call.ts.tzinfo is UTC


def test_database__indexes_tool_calls_written_through_the_trace_store(tmp_path: Path) -> None:
    # End-to-end seam: ToolTraceRecord -> RegistryTraceSink -> JSONL -> read -> index_events.
    store = TraceStore(tmp_path / "traces")
    RegistryTraceSink(store).append(
        ToolTraceRecord(
            run_id="run-store",
            tool_name="read_file",
            args={"path": "src/sample_pkg/dates.py"},
            ok=False,
            error_type=ErrorType.NotFoundError,
            latency_ms=9,
            truncated=True,
            ts=datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC),
        )
    )

    events = store.read("run-store")
    database = Database(tmp_path / "runs.sqlite")
    database.index_events("run-store", events)

    [tool_call] = database.tool_calls("run-store")
    assert tool_call.tool_name == "read_file"
    assert tool_call.ok is False
    assert tool_call.error_type == "NotFoundError"
    assert tool_call.truncated is True
    assert tool_call.latency_ms == 9
    assert tool_call.args == {"path": "src/sample_pkg/dates.py"}
    assert tool_call.ts == datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)


def test_database__structured_outcome_round_trips_through_rebuildable_index(
    tmp_path: Path,
) -> None:
    outcome = {
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "skipped": 0,
        "total": 1,
        "failing_test_ids": ["tests.test_tracked::test_value"],
    }
    store = TraceStore(tmp_path / "traces")
    RegistryTraceSink(store).append(
        ToolTraceRecord(
            run_id="run-outcome",
            tool_name="run_tests",
            args={"rationale": "Verify the correction."},
            ok=True,
            error_type=None,
            latency_ms=13,
            truncated=False,
            ts=datetime(2026, 3, 4, 5, 6, 8, tzinfo=UTC),
            outcome=outcome,
        )
    )
    events = store.read("run-outcome")
    db_path = tmp_path / "runs.sqlite"
    database = Database(db_path)

    database.index_events("run-outcome", events)

    first_rows = database.tool_calls("run-outcome")
    assert len(first_rows) == 1
    assert first_rows[0].payload["outcome"] == outcome

    restarted_database = Database(db_path)
    restarted_database.index_events("run-outcome", events)
    assert restarted_database.tool_calls("run-outcome") == first_rows
