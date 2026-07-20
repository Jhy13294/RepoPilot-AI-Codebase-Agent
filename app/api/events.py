"""Pure safe projections and SSE formatting for persisted trace events."""

from pydantic import JsonValue

from app.api.schemas import TraceEventView
from app.schemas.trace import TraceEvent, TraceEventKind


def to_trace_event_view(event: TraceEvent) -> TraceEventView:
    """Project one trace event through the public metadata allowlist."""
    summary: str | None = None
    tool_name: str | None = None
    ok: bool | None = None
    error_type: str | None = None
    risk_level: str | None = None
    request_id: str | None = None
    decision: str | None = None
    actor: str | None = None
    latency_ms: int | None = None

    if event.kind in {
        TraceEventKind.plan,
        TraceEventKind.replan,
        TraceEventKind.tool_result,
        TraceEventKind.report,
        TraceEventKind.error,
    }:
        summary = _payload_string(event.payload, "summary")
    elif event.kind is TraceEventKind.tool_call:
        tool_name = _payload_string(event.payload, "tool_name")
        ok = _payload_bool(event.payload, "ok")
        error_type = _payload_string(event.payload, "error_type")
        latency_ms = event.latency_ms
    elif event.kind is TraceEventKind.approval_request:
        request_id = _payload_string(event.payload, "request_id")
        tool_name = _payload_string(event.payload, "tool_name")
        risk_level = _payload_string(event.payload, "risk_level")
    elif event.kind is TraceEventKind.approval_decision:
        tool_name = _payload_string(event.payload, "tool_name")
        risk_level = _payload_string(event.payload, "risk_level")
        decision = _payload_string(event.payload, "decision")
        actor = _payload_string(event.payload, "actor")
    elif event.kind is TraceEventKind.critic_verdict:
        summary = _payload_string(event.payload, "summary")
        decision = _payload_string(event.payload, "decision")

    return TraceEventView(
        seq=event.seq,
        ts=event.ts,
        kind=event.kind.value,
        summary=summary,
        tool_name=tool_name,
        ok=ok,
        error_type=error_type,
        risk_level=risk_level,
        request_id=request_id,
        decision=decision,
        actor=actor,
        latency_ms=latency_ms,
    )


def format_sse_frame(view: TraceEventView) -> str:
    """Format one safe trace projection as an SSE event frame."""
    return f"id: {view.seq}\nevent: {view.kind}\ndata: {view.model_dump_json()}\n\n"


def _payload_bool(payload: dict[str, JsonValue], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _payload_string(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None
