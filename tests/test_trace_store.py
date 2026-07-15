import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType
from app.schemas.trace import TraceEvent, TraceEventKind
from app.storage.trace_store import RegistryTraceSink, TraceStore, render_timeline
from app.tools.base import ToolContext
from app.tools.registry import ApprovalTraceRecord, ToolRegistry, ToolSpec, ToolTraceRecord


class _EchoArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


class _EchoPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


class _ExplodingSink:
    def append(self, _record: ToolTraceRecord) -> None:
        raise RuntimeError("trace sink failed")


def _tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="run-1", jail=PathJail(tmp_path))


def _echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo a value.",
        args_schema=_EchoArgs,
        returns_schema=_EchoPayload,
        risk_level="low",
    )


def test_trace_event_kind__matches_architecture_taxonomy() -> None:
    assert tuple(kind.value for kind in TraceEventKind) == (
        "plan",
        "tool_call",
        "tool_result",
        "approval_request",
        "approval_decision",
        "critic_verdict",
        "replan",
        "report",
        "error",
    )


def test_trace_event__is_frozen_and_forbids_extra_fields() -> None:
    event = TraceEvent(
        run_id="run-1",
        seq=0,
        ts=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        kind=TraceEventKind.plan,
        payload={"summary": "start"},
    )

    with pytest.raises(ValidationError):
        event.seq = 1

    with pytest.raises(ValidationError):
        TraceEvent.model_validate(
            {
                "run_id": "run-1",
                "seq": 0,
                "ts": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
                "kind": TraceEventKind.plan,
                "payload": {},
                "extra_field": "nope",
            }
        )


def test_trace_store__appends_and_reads_events_with_monotonic_seq(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)

    first = store.append("run-a", TraceEventKind.plan, {"summary": "plan ready"})
    second = store.append("run-a", TraceEventKind.tool_call, {"tool_name": "read_file"})
    third = store.append("run-a", TraceEventKind.report, {"summary": "done"})

    events = store.read("run-a")

    assert events == [first, second, third]
    assert [event.seq for event in events] == [0, 1, 2]


def test_trace_store__writes_one_valid_json_object_per_line(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)

    store.append("run-jsonl", TraceEventKind.plan, {"summary": "first"})
    store.append("run-jsonl", TraceEventKind.report, {"summary": "second"})

    lines = (tmp_path / "run-jsonl.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    for line in lines:
        decoded = json.loads(line)
        assert isinstance(decoded, dict)
        assert decoded["run_id"] == "run-jsonl"


def test_trace_store__uses_current_utc_timestamp_and_preserves_explicit_ts(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    before = datetime.now(UTC)

    default_ts_event = store.append("run-time", TraceEventKind.plan, {})

    after = datetime.now(UTC)
    explicit_ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    explicit_ts_event = store.append(
        "run-time",
        TraceEventKind.report,
        {},
        ts=explicit_ts,
    )

    assert before <= default_ts_event.ts <= after
    assert default_ts_event.ts.tzinfo is UTC
    assert explicit_ts_event.ts == explicit_ts


def test_trace_store__continues_seq_across_store_instances(tmp_path: Path) -> None:
    first_store = TraceStore(tmp_path)
    first_store.append("run-restart", TraceEventKind.plan, {})
    first_store.append("run-restart", TraceEventKind.tool_call, {"tool_name": "search_code"})

    restarted_store = TraceStore(tmp_path)
    event = restarted_store.append("run-restart", TraceEventKind.report, {"summary": "done"})

    assert event.seq == 2
    assert [event.seq for event in restarted_store.read("run-restart")] == [0, 1, 2]


def test_trace_store__read_missing_run_returns_empty_list(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)

    assert store.read("missing-run") == []


def test_registry_trace_sink__converts_tool_records_to_tool_call_events(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    sink = RegistryTraceSink(store)
    ts = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    record = ToolTraceRecord(
        run_id="run-sink",
        tool_name="read_file",
        args={"path": "README.md"},
        ok=False,
        error_type=ErrorType.NotFoundError,
        latency_ms=17,
        truncated=True,
        ts=ts,
    )

    sink.append(record)

    events = store.read("run-sink")
    assert len(events) == 1
    event = events[0]
    assert event.kind is TraceEventKind.tool_call
    assert event.payload == {
        "tool_name": "read_file",
        "args": {"path": "README.md"},
        "ok": False,
        "error_type": "NotFoundError",
        "truncated": True,
    }
    assert event.latency_ms == 17
    assert event.ts == ts


def test_registry_trace_sink__converts_approval_records_to_decision_events(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    sink = RegistryTraceSink(store)
    ts = datetime(2026, 2, 3, 4, 5, 5, tzinfo=UTC)
    record = ApprovalTraceRecord(
        run_id="run-approval",
        tool_name="apply_patch",
        risk_level="high",
        decision="denied",
        actor="human",
        reason="needs review",
        ts=ts,
    )

    sink.append_approval(record)

    events = store.read("run-approval")
    assert len(events) == 1
    event = events[0]
    assert event.kind is TraceEventKind.approval_decision
    assert event.payload == {
        "tool_name": "apply_patch",
        "risk_level": "high",
        "decision": "denied",
        "actor": "human",
        "reason": "needs review",
    }
    assert event.ts == ts


def test_registry_trace_sink__includes_non_none_outcome_in_tool_call_payload(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    outcome = {
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "skipped": 0,
        "total": 1,
        "failing_test_ids": ["tests.test_sample::test_failure"],
    }
    record = ToolTraceRecord(
        run_id="run-outcome",
        tool_name="generic_test_runner",
        args={"suite": "unit"},
        ok=True,
        error_type=None,
        latency_ms=11,
        truncated=False,
        ts=datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC),
        outcome=outcome,
    )

    RegistryTraceSink(store).append(record)

    [event] = store.read("run-outcome")
    assert event.payload == {
        "tool_name": "generic_test_runner",
        "args": {"suite": "unit"},
        "ok": True,
        "error_type": None,
        "truncated": False,
        "outcome": outcome,
    }


def test_render_timeline__includes_seq_kind_and_readable_summaries(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(
        "run-timeline",
        TraceEventKind.tool_call,
        {
            "tool_name": "read_file",
            "args": {"path": "README.md"},
            "ok": True,
            "error_type": None,
            "truncated": False,
        },
        ts=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    store.append(
        "run-timeline",
        TraceEventKind.error,
        {"message": "budget exhausted"},
        ts=datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC),
    )

    timeline = render_timeline(store.read("run-timeline"))

    lines = timeline.splitlines()
    assert len(lines) == 2
    assert "#0 03:04:05 tool_call - read_file ok" in lines[0]
    assert "#1 03:04:06 error - budget exhausted" in lines[1]


def test_tool_registry__propagates_trace_sink_exceptions(tmp_path: Path) -> None:
    registry = ToolRegistry(trace_sink=_ExplodingSink())
    registry.register(_echo_spec(), lambda args, _context: _EchoArgs.model_validate(args))

    with pytest.raises(RuntimeError, match="trace sink failed"):
        registry.dispatch("echo", {"value": "x"}, _tool_context(tmp_path))
