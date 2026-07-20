"""SQLite projections for agent run state and trace tool-call indexes."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
    update,
)
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.agent.state import AgentState, PlanStep, RunStatus
from app.schemas.trace import TraceEvent, TraceEventKind


class Base(DeclarativeBase):
    """Base class for RepoPilot storage models."""


class _RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    repo: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    cursor: Mapped[int] = mapped_column(Integer, nullable=False)
    steps_used: Mapped[int] = mapped_column(Integer, nullable=False)
    replans_used: Mapped[int] = mapped_column(Integer, nullable=False)
    fix_cycles_used: Mapped[int] = mapped_column(Integer, nullable=False)
    state_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _StepRow(Base):
    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("run_id", "idx", name="uq_steps_run_id_idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    intent: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_tools_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    success_check: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class _ToolCallRow(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_tool_calls_run_id_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    args_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    payload_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class _ApprovalRequestRow(Base):
    __tablename__ = "approval_requests"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    args_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RunSummary(BaseModel):
    """Frozen read model for a persisted run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_type: str
    prompt: str
    repo: str
    status: RunStatus
    cursor: int
    step_count: int
    steps_used: int
    replans_used: int
    fix_cycles_used: int
    created_at: datetime
    updated_at: datetime


class ToolCallView(BaseModel):
    """Frozen read model for one indexed tool-call trace event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seq: int
    ts: datetime
    tool_name: str
    args: dict[str, JsonValue]
    ok: bool
    error_type: str | None
    latency_ms: int | None
    truncated: bool
    payload: dict[str, JsonValue]


class ApprovalRequestView(BaseModel):
    """Frozen read model for one durable approval request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    run_id: str
    tool_name: str
    risk_level: Literal["low", "medium", "high"]
    args: dict[str, JsonValue]
    status: Literal["pending", "approved", "denied"]
    actor: str | None
    note: str | None
    created_at: datetime
    decided_at: datetime | None


class ApprovalRequestNotFoundError(LookupError):
    """Raised when an approval decision targets an unknown request."""


class ApprovalRequestAlreadyDecidedError(RuntimeError):
    """Raised when an approval decision would overwrite an existing decision."""


class Database:
    """Small SQLite persistence facade that keeps ORM rows inside the module."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path.as_posix()}")
        self._session_factory: sessionmaker[Session] = sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        Base.metadata.create_all(self.engine)

    def save_state(self, state: AgentState) -> None:
        """Persist an AgentState snapshot and rebuild its relational step projection."""
        now = datetime.now(UTC)
        state_json = state.model_dump_json().encode("utf-8")

        with self._session_factory.begin() as session:
            row = session.get(_RunRow, state.run_id)
            if row is None:
                row = _RunRow(run_id=state.run_id, created_at=now, updated_at=now)
                session.add(row)

            row.task_type = state.task.task_type
            row.prompt = state.task.prompt
            row.repo = state.task.repo
            row.status = state.status.value
            row.cursor = state.cursor
            row.steps_used = state.steps_used
            row.replans_used = state.replans_used
            row.fix_cycles_used = state.fix_cycles_used
            row.state_json = state_json
            row.updated_at = now

            session.execute(delete(_StepRow).where(_StepRow.run_id == state.run_id))
            session.add_all(_step_row(state.run_id, step) for step in state.plan)

    def load_state(self, run_id: str) -> AgentState | None:
        """Load a persisted AgentState snapshot, or None when the run is unknown."""
        with self._session_factory() as session:
            row = session.get(_RunRow, run_id)
            if row is None:
                return None

            return AgentState.model_validate_json(row.state_json.decode("utf-8"))

    def index_events(self, run_id: str, events: Sequence[TraceEvent]) -> None:
        """Rebuild the per-run tool-call index from JSONL trace events."""
        tool_call_events = [
            event
            for event in sorted(events, key=lambda item: item.seq)
            if event.run_id == run_id and event.kind is TraceEventKind.tool_call
        ]

        with self._session_factory.begin() as session:
            session.execute(delete(_ToolCallRow).where(_ToolCallRow.run_id == run_id))
            session.add_all(_tool_call_row(event) for event in tool_call_events)

    def get_run(self, run_id: str) -> RunSummary | None:
        """Return one persisted run summary, or None when the run is unknown."""
        with self._session_factory() as session:
            row = session.get(_RunRow, run_id)
            if row is None:
                return None
            return _run_summary(row, _step_count(session, run_id))

    def list_runs(self) -> list[RunSummary]:
        """Return persisted runs as frozen DTOs ordered by run_id."""
        with self._session_factory() as session:
            rows = session.scalars(select(_RunRow).order_by(_RunRow.run_id)).all()
            return [_run_summary(row, _step_count(session, row.run_id)) for row in rows]

    def tool_calls(self, run_id: str) -> list[ToolCallView]:
        """Return indexed tool calls for one run as frozen DTOs ordered by trace sequence."""
        statement = (
            select(_ToolCallRow).where(_ToolCallRow.run_id == run_id).order_by(_ToolCallRow.seq)
        )
        with self._session_factory() as session:
            rows = session.scalars(statement).all()
            return [_tool_call_view(row) for row in rows]

    def create_approval_request(
        self,
        request_id: str,
        *,
        run_id: str,
        tool_name: str,
        risk_level: Literal["low", "medium", "high"],
        args: dict[str, JsonValue],
        created_at: datetime | None = None,
    ) -> None:
        """Create one pending approval request without changing any run projection."""
        with self._session_factory.begin() as session:
            session.add(
                _ApprovalRequestRow(
                    request_id=request_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    risk_level=risk_level,
                    args_json=_json_blob(cast(JsonValue, args)),
                    status="pending",
                    actor=None,
                    note=None,
                    created_at=_as_utc(created_at or datetime.now(UTC)),
                    decided_at=None,
                )
            )

    def get_approval_request(self, request_id: str) -> ApprovalRequestView | None:
        """Return one durable approval request, or None when it is unknown."""
        with self._session_factory() as session:
            row = session.get(_ApprovalRequestRow, request_id)
            return _approval_request_view(row) if row is not None else None

    def list_pending_approvals(self, run_id: str | None = None) -> list[ApprovalRequestView]:
        """Return pending approvals, optionally filtered by run, in stable creation order."""
        statement = select(_ApprovalRequestRow).where(_ApprovalRequestRow.status == "pending")
        if run_id is not None:
            statement = statement.where(_ApprovalRequestRow.run_id == run_id)
        statement = statement.order_by(
            _ApprovalRequestRow.created_at,
            _ApprovalRequestRow.request_id,
        )
        with self._session_factory() as session:
            rows = session.scalars(statement).all()
            return [_approval_request_view(row) for row in rows]

    def record_approval_decision(
        self,
        request_id: str,
        *,
        approved: bool,
        actor: str | None,
        note: str | None,
        decided_at: datetime,
    ) -> None:
        """Record the first decision for a pending approval request."""
        with self._session_factory.begin() as session:
            result = cast(
                CursorResult[tuple[object, ...]],
                session.execute(
                    update(_ApprovalRequestRow)
                    .where(
                        _ApprovalRequestRow.request_id == request_id,
                        _ApprovalRequestRow.status == "pending",
                    )
                    .values(
                        status="approved" if approved else "denied",
                        actor=actor,
                        note=note,
                        decided_at=_as_utc(decided_at),
                    )
                ),
            )
            if result.rowcount == 1:
                return

            row = session.get(_ApprovalRequestRow, request_id)
            if row is None:
                raise ApprovalRequestNotFoundError(request_id)
            raise ApprovalRequestAlreadyDecidedError(request_id)


def _step_row(run_id: str, step: PlanStep) -> _StepRow:
    return _StepRow(
        run_id=run_id,
        idx=step.index,
        intent=step.intent,
        suggested_tools_json=_json_blob(cast(JsonValue, step.suggested_tools)),
        success_check=step.success_check,
        status=step.status.value,
    )


def _tool_call_row(event: TraceEvent) -> _ToolCallRow:
    args = _payload_args(event.payload)
    return _ToolCallRow(
        run_id=event.run_id,
        seq=event.seq,
        ts=_as_utc(event.ts),
        tool_name=_payload_string(event.payload, "tool_name") or "unknown_tool",
        args_json=_json_blob(cast(JsonValue, args)),
        ok=_payload_bool(event.payload, "ok"),
        error_type=_payload_string(event.payload, "error_type"),
        latency_ms=event.latency_ms,
        truncated=_payload_bool(event.payload, "truncated"),
        payload_json=_json_blob(cast(JsonValue, event.payload)),
    )


def _run_summary(row: _RunRow, step_count: int) -> RunSummary:
    return RunSummary(
        run_id=row.run_id,
        task_type=row.task_type,
        prompt=row.prompt,
        repo=row.repo,
        status=RunStatus(row.status),
        cursor=row.cursor,
        step_count=step_count,
        steps_used=row.steps_used,
        replans_used=row.replans_used,
        fix_cycles_used=row.fix_cycles_used,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
    )


def _tool_call_view(row: _ToolCallRow) -> ToolCallView:
    return ToolCallView(
        run_id=row.run_id,
        seq=row.seq,
        ts=_as_utc(row.ts),
        tool_name=row.tool_name,
        args=_json_dict(row.args_json),
        ok=row.ok,
        error_type=row.error_type,
        latency_ms=row.latency_ms,
        truncated=row.truncated,
        payload=_json_dict(row.payload_json),
    )


def _approval_request_view(row: _ApprovalRequestRow) -> ApprovalRequestView:
    return ApprovalRequestView(
        request_id=row.request_id,
        run_id=row.run_id,
        tool_name=row.tool_name,
        risk_level=cast(Literal["low", "medium", "high"], row.risk_level),
        args=_json_dict(row.args_json),
        status=cast(Literal["pending", "approved", "denied"], row.status),
        actor=row.actor,
        note=row.note,
        created_at=_as_utc(row.created_at),
        decided_at=_as_utc(row.decided_at) if row.decided_at is not None else None,
    )


def _step_count(session: Session, run_id: str) -> int:
    step_ids = session.scalars(select(_StepRow.id).where(_StepRow.run_id == run_id)).all()
    return len(step_ids)


def _json_blob(value: JsonValue) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_dict(blob: bytes) -> dict[str, JsonValue]:
    value = json.loads(blob.decode("utf-8"))
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, JsonValue], value)


def _payload_args(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = payload.get("args")
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, JsonValue], value)


def _payload_bool(payload: dict[str, JsonValue], key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False


def _payload_string(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
