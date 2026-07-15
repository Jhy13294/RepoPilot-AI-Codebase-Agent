"""Append-only JSONL trace storage."""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from app.schemas.trace import TraceEvent, TraceEventKind
from app.tools.registry import ToolTraceRecord


class TraceStore:
    """Persist one JSONL trace file per run."""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._next_seq: dict[str, int] = {}

    def append(
        self,
        run_id: str,
        kind: TraceEventKind,
        payload: dict[str, JsonValue],
        *,
        ts: datetime | None = None,
        latency_ms: int | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
    ) -> TraceEvent:
        """Append one event to a run trace and return the persisted event."""
        seq = self._next_seq_for(run_id)
        event = TraceEvent(
            run_id=run_id,
            seq=seq,
            ts=ts or datetime.now(UTC),
            kind=kind,
            payload=payload,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        self._append_jsonl(self._path_for(run_id), event.model_dump_json())
        self._next_seq[run_id] = seq + 1
        return event

    def read(self, run_id: str) -> list[TraceEvent]:
        """Read a run trace ordered by sequence number."""
        path = self._path_for(run_id)
        if not path.exists():
            return []

        events: list[TraceEvent] = []
        with path.open("r", encoding="utf-8") as trace_file:
            for line in trace_file:
                events.append(TraceEvent.model_validate_json(line))
        return sorted(events, key=lambda event: event.seq)

    def _next_seq_for(self, run_id: str) -> int:
        if run_id not in self._next_seq:
            self._next_seq[run_id] = self._count_existing_events(run_id)
        return self._next_seq[run_id]

    def _count_existing_events(self, run_id: str) -> int:
        path = self._path_for(run_id)
        if not path.exists():
            return 0

        with path.open("r", encoding="utf-8") as trace_file:
            return sum(1 for _line in trace_file)

    def _path_for(self, run_id: str) -> Path:
        return self.trace_dir / f"{run_id}.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, json_line: str) -> None:
        data = f"{json_line}\n".encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o666)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


class RegistryTraceSink:
    """Trace sink that writes registry records to a TraceStore."""

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def append(self, record: ToolTraceRecord) -> None:
        """Persist one registry trace record as a tool_call event."""
        payload: dict[str, JsonValue] = {
            "tool_name": record.tool_name,
            "args": cast(JsonValue, record.args),
            "ok": record.ok,
            "error_type": record.error_type.value if record.error_type is not None else None,
            "truncated": record.truncated,
        }
        if record.outcome is not None:
            payload["outcome"] = cast(JsonValue, record.outcome)
        self._store.append(
            record.run_id,
            TraceEventKind.tool_call,
            payload,
            ts=record.ts,
            latency_ms=record.latency_ms,
        )


def render_timeline(events: Sequence[TraceEvent]) -> str:
    """Render trace events as a compact human-readable timeline."""
    lines = []
    for event in events:
        lines.append(
            f"#{event.seq} {event.ts:%H:%M:%S} {event.kind.value} - {_summarize_event(event)}"
        )
    return "\n".join(lines)


def _summarize_event(event: TraceEvent) -> str:
    if event.kind is TraceEventKind.tool_call:
        return _summarize_tool_call(event.payload)

    for key in ("summary", "message", "title"):
        value = _payload_string(event.payload, key)
        if value:
            return value

    if not event.payload:
        return "no payload"

    keys = ", ".join(sorted(event.payload))
    return f"payload: {keys}"


def _summarize_tool_call(payload: dict[str, JsonValue]) -> str:
    tool_name = _payload_string(payload, "tool_name") or "unknown_tool"
    ok = payload.get("ok")
    if ok is True:
        status = "ok"
    else:
        error_type = _payload_string(payload, "error_type")
        status = f"error {error_type}" if error_type else "error"

    suffix = ", truncated" if payload.get("truncated") is True else ""
    return f"{tool_name} {status}{suffix}"


def _payload_string(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None
