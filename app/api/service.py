"""Application service for durable, background read-only agent runs."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue

from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import run_agent_loop
from app.agent.planner import Planner
from app.agent.reporter import Reporter
from app.agent.state import (
    TERMINAL_STATUSES,
    AgentState,
    Budgets,
    RunStatus,
    TaskSpec,
)
from app.api.schemas import (
    ApprovalRequestView,
    CreateRunRequest,
    CreateRunResponse,
    RunListResponse,
    RunSummaryView,
    RunView,
    ToolCallEventView,
)
from app.cli import _build_read_only_registry, _tools_doc
from app.config import Settings
from app.safety.async_approval import ApprovalCoordinator
from app.safety.path_jail import PathJail
from app.schemas.trace import TraceEvent, TraceEventKind
from app.services.llm_client import LLMClient, build_llm_client
from app.storage.db import ApprovalRequestView as StoredApprovalRequestView
from app.storage.db import Database, RunSummary
from app.storage.trace_store import RegistryTraceSink, TraceStore

SpawnStrategy = Callable[[Callable[[], None]], None]
ClientFactory = Callable[[Settings], LLMClient]


class InvalidRepositoryError(ValueError):
    """Raised when an API run targets a missing or non-directory repository."""


class RunService:
    """Create, execute, and read durable agent runs."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        store: TraceStore | None = None,
        coordinator: ApprovalCoordinator | None = None,
        spawn: SpawnStrategy | None = None,
        client_factory: ClientFactory = build_llm_client,
    ) -> None:
        self._settings = settings
        self._database = database or Database(settings.db_path)
        self._store = store or TraceStore(settings.trace_dir)
        self._coordinator = coordinator or ApprovalCoordinator(self._database, self._store)
        self._client_factory = client_factory
        self._executor: ThreadPoolExecutor | None = None
        self._spawn: SpawnStrategy
        if spawn is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="repopilot-run",
            )
            self._spawn = self._submit
        else:
            self._spawn = spawn

    def create_run(self, request: CreateRunRequest) -> CreateRunResponse:
        """Persist a PLANNING run before scheduling its background execution."""
        try:
            jail = PathJail(Path(request.repo))
        except ValueError as exc:
            raise InvalidRepositoryError(str(exc)) from exc

        run_id = str(uuid4())
        task = TaskSpec(
            task_type=request.task_type,
            prompt=request.prompt,
            repo=str(jail.root),
        )
        budgets = Budgets(
            max_steps=(
                request.max_steps if request.max_steps is not None else self._settings.max_steps
            ),
            max_replans=self._settings.max_replans,
            max_fix_cycles=self._settings.max_fix_cycles,
        )
        state = AgentState(
            run_id=run_id,
            task=task,
            plan=[],
            cursor=0,
            tool_history=[],
            budgets=budgets,
            status=RunStatus.PLANNING,
        )
        self._database.save_state(state)
        self._spawn(lambda: self._execute(run_id, task, budgets, jail))
        return CreateRunResponse(run_id=run_id, status=state.status)

    def get_run(self, run_id: str) -> RunView | None:
        """Read one run from SQLite plus its live JSONL tool-call timeline."""
        summary = self._database.get_run(run_id)
        if summary is None:
            return None

        events = self._store.read(run_id)
        summary_view = _run_summary_view(summary)
        return RunView.model_validate(
            {
                **summary_view.model_dump(),
                "summary": _terminal_summary(summary.status, events),
                "tool_calls": [
                    _tool_call_view(event)
                    for event in events
                    if event.kind is TraceEventKind.tool_call
                ],
            }
        )

    def list_runs(self) -> RunListResponse:
        """Read the lightweight SQLite projection for all persisted runs."""
        return RunListResponse(
            runs=[_run_summary_view(summary) for summary in self._database.list_runs()]
        )

    def get_approval_request(self, request_id: str) -> ApprovalRequestView | None:
        """Read one durable approval request for endpoint precondition checks."""
        request = self._database.get_approval_request(request_id)
        return _approval_request_view(request) if request is not None else None

    def list_pending_approvals(self, run_id: str | None = None) -> list[ApprovalRequestView]:
        """Forward a pending-approval query to the shared coordinator."""
        return [_approval_request_view(item) for item in self._coordinator.list_pending(run_id)]

    def decide_approval(
        self,
        request_id: str,
        *,
        approved: bool,
        actor: str | None,
        note: str | None,
    ) -> ApprovalRequestView:
        """Forward one human approval decision to the shared coordinator."""
        return _approval_request_view(
            self._coordinator.decide(
                request_id,
                approved=approved,
                actor=actor,
                note=note,
            )
        )

    def close(self) -> None:
        """Release the service-owned background executor, if any."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _submit(self, job: Callable[[], None]) -> None:
        assert self._executor is not None
        self._executor.submit(job)

    def _execute(
        self,
        run_id: str,
        task: TaskSpec,
        budgets: Budgets,
        jail: PathJail,
    ) -> None:
        trace_sink = RegistryTraceSink(self._store)
        registry = _build_read_only_registry(trace_sink=trace_sink)
        client = self._client_factory(self._settings)
        run_agent_loop(
            task,
            planner=Planner(client, self._store, tools_doc=_tools_doc(registry)),
            executor=Executor(client, registry, jail, self._store),
            critic=Critic(client, self._store),
            store=self._store,
            database=self._database,
            run_id=run_id,
            budgets=budgets,
            reporter=Reporter(client),
            jail=jail,
        )


def _run_summary_view(summary: RunSummary) -> RunSummaryView:
    return RunSummaryView(
        run_id=summary.run_id,
        task_type=summary.task_type,
        prompt=summary.prompt,
        repo=summary.repo,
        status=summary.status,
        step_count=summary.step_count,
        steps_used=summary.steps_used,
        replans_used=summary.replans_used,
        fix_cycles_used=summary.fix_cycles_used,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _approval_request_view(request: StoredApprovalRequestView) -> ApprovalRequestView:
    return ApprovalRequestView.model_validate(request.model_dump())


def _terminal_summary(status: RunStatus, events: Sequence[TraceEvent]) -> str | None:
    if status not in TERMINAL_STATUSES:
        return None
    for event in reversed(events):
        if event.kind is TraceEventKind.report:
            return _payload_string(event.payload, "summary")
    return None


def _tool_call_view(event: TraceEvent) -> ToolCallEventView:
    return ToolCallEventView(
        seq=event.seq,
        ts=event.ts,
        tool_name=_payload_string(event.payload, "tool_name") or "unknown_tool",
        ok=_payload_bool(event.payload, "ok"),
        error_type=_payload_string(event.payload, "error_type"),
        latency_ms=event.latency_ms,
    )


def _payload_bool(payload: dict[str, JsonValue], key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False


def _payload_string(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None
