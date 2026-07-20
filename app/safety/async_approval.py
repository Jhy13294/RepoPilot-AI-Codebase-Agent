"""Durable approval coordination for worker-thread tool dispatch."""

from datetime import UTC, datetime
from threading import Event, Lock
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue

from app.agent.state import RunStatus, Trigger, next_status
from app.schemas.trace import TraceEventKind
from app.storage.db import ApprovalRequestView, Database
from app.storage.trace_store import TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ApprovalGate, ApprovalOutcome, ToolSpec


class ApprovalCoordinator:
    """Persist approval requests and park active worker threads until a decision arrives."""

    def __init__(self, database: Database, store: TraceStore) -> None:
        self._database = database
        self._store = store
        self._waiters: dict[str, Event] = {}
        self._waiters_lock = Lock()

    def request(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        """Create a durable request and block until its in-process waiter is decided."""
        request_id = str(uuid4())
        created_at = datetime.now(UTC)
        validated_args = cast(dict[str, JsonValue], args.model_dump(mode="json"))
        waiter = Event()

        # Keep a just-created pending row from being decided before its waiter is registered.
        with self._waiters_lock:
            self._database.create_approval_request(
                request_id,
                run_id=context.run_id,
                tool_name=spec.name,
                risk_level=spec.risk_level,
                args=validated_args,
                created_at=created_at,
            )
            self._store.append(
                context.run_id,
                TraceEventKind.approval_request,
                {
                    "request_id": request_id,
                    "tool_name": spec.name,
                    "risk_level": spec.risk_level,
                },
                ts=created_at,
            )
            self._overlay_awaiting_approval(context.run_id)
            self._waiters[request_id] = waiter

        waiter.wait()
        try:
            request = self._database.get_approval_request(request_id)
            if request is None or request.status == "pending":
                raise RuntimeError(f"Approval request '{request_id}' woke without a decision.")
            return ApprovalOutcome(
                approved=request.status == "approved",
                reason=request.note,
                actor=request.actor or "human",
            )
        finally:
            with self._waiters_lock:
                if self._waiters.get(request_id) is waiter:
                    self._waiters.pop(request_id)

    def decide(
        self,
        request_id: str,
        *,
        approved: bool,
        actor: str | None,
        note: str | None,
    ) -> ApprovalRequestView:
        """Persist one decision before waking its active in-process waiter, if present."""
        with self._waiters_lock:
            self._database.record_approval_decision(
                request_id,
                approved=approved,
                actor=actor,
                note=note,
                decided_at=datetime.now(UTC),
            )
            waiter = self._waiters.get(request_id)
            if waiter is not None:
                waiter.set()
            updated = self._database.get_approval_request(request_id)

        if updated is None:
            raise RuntimeError(f"Approval request '{request_id}' disappeared after its decision.")
        return updated

    def list_pending(self, run_id: str | None = None) -> list[ApprovalRequestView]:
        """Return durable pending approval requests, optionally filtered by run."""
        return self._database.list_pending_approvals(run_id)

    def _overlay_awaiting_approval(self, run_id: str) -> None:
        summary = self._database.get_run(run_id)
        if summary is None or summary.status is not RunStatus.EXECUTING:
            return

        state = self._database.load_state(run_id)
        if state is None or state.status is not RunStatus.EXECUTING:
            return
        awaiting_status = next_status(state.status, Trigger.request_approval)
        self._database.save_state(state.model_copy(update={"status": awaiting_status}))


class AsyncApprovalGate(ApprovalGate):
    """Approval gate that delegates decisions to a shared durable coordinator."""

    def __init__(self, coordinator: ApprovalCoordinator) -> None:
        self._coordinator = coordinator

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        """Block the calling worker until the coordinator returns a durable decision."""
        return self._coordinator.request(spec, args, context)
