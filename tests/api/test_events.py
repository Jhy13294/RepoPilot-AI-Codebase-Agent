from datetime import UTC, datetime

import pytest
from pydantic import JsonValue, ValidationError

from app.api.events import format_sse_frame, to_trace_event_view
from app.api.schemas import TraceEventView
from app.schemas.trace import TraceEvent, TraceEventKind

_TS = datetime(2026, 7, 20, 8, 30, tzinfo=UTC)
_SENTINEL = "PRIVATE_DIFF_SENTINEL_7f3b"
_FORBIDDEN_FIELDS = {
    "args",
    "diff",
    "steps",
    "findings",
    "evidence",
    "analysis",
    "final_findings",
    "message",
    "outcome",
}


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (TraceEventKind.plan, {"summary": "Plan ready."}),
        (TraceEventKind.replan, {"summary": "Plan revised."}),
        (
            TraceEventKind.tool_call,
            {
                "tool_name": "apply_patch",
                "ok": False,
                "error_type": "PatchApplyError",
                "latency_ms": 17,
            },
        ),
        (TraceEventKind.tool_result, {"summary": "Tool step completed."}),
        (
            TraceEventKind.approval_request,
            {
                "request_id": "request-1",
                "tool_name": "apply_patch",
                "risk_level": "high",
            },
        ),
        (
            TraceEventKind.approval_decision,
            {
                "tool_name": "apply_patch",
                "risk_level": "high",
                "decision": "approved",
                "actor": "human",
            },
        ),
        (
            TraceEventKind.critic_verdict,
            {"summary": "Evidence is sufficient.", "decision": "proceed"},
        ),
        (TraceEventKind.report, {"summary": "Run completed."}),
        (TraceEventKind.error, {"summary": "Run failed safely."}),
    ],
)
def test_to_trace_event_view__projects_only_kind_specific_safe_scalars(
    kind: TraceEventKind,
    expected: dict[str, object],
) -> None:
    summaries = {
        TraceEventKind.plan: "Plan ready.",
        TraceEventKind.replan: "Plan revised.",
        TraceEventKind.tool_result: "Tool step completed.",
        TraceEventKind.critic_verdict: "Evidence is sufficient.",
        TraceEventKind.report: "Run completed.",
        TraceEventKind.error: "Run failed safely.",
    }
    payload: dict[str, JsonValue] = {
        "summary": summaries.get(kind, "must not project"),
        "tool_name": "apply_patch",
        "ok": False,
        "error_type": "PatchApplyError",
        "risk_level": "high",
        "request_id": "request-1",
        "decision": "approved" if kind is TraceEventKind.approval_decision else "proceed",
        "actor": "human",
        "args": {"diff": _SENTINEL},
        "steps": [_SENTINEL],
        "findings": _SENTINEL,
        "evidence": [_SENTINEL],
        "analysis": _SENTINEL,
        "final_findings": _SENTINEL,
        "message": _SENTINEL,
        "outcome": {"private": _SENTINEL},
    }
    event = TraceEvent(
        run_id="run-1",
        seq=4,
        ts=_TS,
        kind=kind,
        payload=payload,
        latency_ms=17,
    )

    view = to_trace_event_view(event)

    assert view.model_dump(exclude_none=True) == {
        "seq": 4,
        "ts": _TS,
        "kind": kind.value,
        **expected,
    }
    assert _SENTINEL not in view.model_dump_json()
    assert _FORBIDDEN_FIELDS.isdisjoint(TraceEventView.model_fields)


def test_format_sse_frame__uses_sequence_kind_and_safe_view_json() -> None:
    view = TraceEventView(
        seq=7,
        ts=_TS,
        kind="approval_request",
        request_id="request-7",
        tool_name="apply_patch",
        risk_level="high",
    )

    assert format_sse_frame(view) == (
        f"id: 7\nevent: approval_request\ndata: {view.model_dump_json()}\n\n"
    )


def test_trace_event_view__is_frozen_and_forbids_extra_fields() -> None:
    view = TraceEventView(seq=0, ts=_TS, kind="plan", summary="Plan ready.")

    with pytest.raises(ValidationError):
        view.summary = "Changed."
    with pytest.raises(ValidationError):
        TraceEventView.model_validate(
            {
                "seq": 0,
                "ts": _TS,
                "kind": "plan",
                "summary": "Plan ready.",
                "args": {"diff": _SENTINEL},
            }
        )
