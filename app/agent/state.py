"""Agent state schemas and pure lifecycle transitions."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolTraceRecord


class RunStatus(StrEnum):
    """Lifecycle status for a full agent run."""

    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    VERIFYING = "VERIFYING"
    REPLANNING = "REPLANNING"
    REPORTING = "REPORTING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskSpec(BaseModel):
    """Minimal task description accepted by the P3 agent state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: Literal["question", "issue", "fix"]
    prompt: str
    repo: str


class PlanStepStatus(StrEnum):
    """Execution status for one planned step."""

    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class PlanStep(BaseModel):
    """One planner-produced step with a critic-checkable success criterion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    intent: str
    suggested_tools: list[str]
    success_check: str
    status: PlanStepStatus = PlanStepStatus.pending


class Budgets(BaseModel):
    """Run-level limits copied from the current configuration defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=20, ge=1)
    max_replans: int = Field(default=3, ge=1)
    max_fix_cycles: int = Field(default=2, ge=1)
    token_cap: int | None = None
    cost_cap: float | None = None


class AgentState(BaseModel):
    """Flat single source of truth for one agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task: TaskSpec
    plan: list[PlanStep]
    cursor: int = Field(ge=0)
    tool_history: list[ToolTraceRecord]
    scratchpad: str = ""
    budgets: Budgets = Field(default_factory=Budgets)
    status: RunStatus
    steps_used: int = Field(default=0, ge=0)
    replans_used: int = Field(default=0, ge=0)
    fix_cycles_used: int = Field(default=0, ge=0)


class Trigger(StrEnum):
    """Events that drive the pure run status transition table."""

    plan_produced = "plan_produced"
    request_approval = "request_approval"
    approval_granted = "approval_granted"
    approval_denied = "approval_denied"
    step_finished = "step_finished"
    verdict_proceed = "verdict_proceed"
    verdict_retry = "verdict_retry"
    verdict_replan = "verdict_replan"
    replan_ok = "replan_ok"
    replan_exhausted = "replan_exhausted"
    all_steps_done = "all_steps_done"
    fatal_or_budget = "fatal_or_budget"
    cancel = "cancel"
    report_done = "report_done"
    report_failed = "report_failed"
    report_cancelled = "report_cancelled"


TERMINAL_STATUSES = frozenset({RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED})

_EXPLICIT_TRANSITIONS: dict[tuple[RunStatus, Trigger], RunStatus] = {
    (RunStatus.PLANNING, Trigger.plan_produced): RunStatus.EXECUTING,
    (RunStatus.EXECUTING, Trigger.request_approval): RunStatus.AWAITING_APPROVAL,
    (RunStatus.AWAITING_APPROVAL, Trigger.approval_granted): RunStatus.EXECUTING,
    (RunStatus.AWAITING_APPROVAL, Trigger.approval_denied): RunStatus.REPLANNING,
    (RunStatus.EXECUTING, Trigger.step_finished): RunStatus.VERIFYING,
    (RunStatus.VERIFYING, Trigger.verdict_proceed): RunStatus.EXECUTING,
    (RunStatus.VERIFYING, Trigger.verdict_retry): RunStatus.EXECUTING,
    (RunStatus.VERIFYING, Trigger.verdict_replan): RunStatus.REPLANNING,
    (RunStatus.REPLANNING, Trigger.replan_ok): RunStatus.EXECUTING,
    (RunStatus.REPLANNING, Trigger.replan_exhausted): RunStatus.REPORTING,
    (RunStatus.VERIFYING, Trigger.all_steps_done): RunStatus.REPORTING,
    (RunStatus.PLANNING, Trigger.fatal_or_budget): RunStatus.REPORTING,
    (RunStatus.EXECUTING, Trigger.fatal_or_budget): RunStatus.REPORTING,
    (RunStatus.VERIFYING, Trigger.fatal_or_budget): RunStatus.REPORTING,
    (RunStatus.REPLANNING, Trigger.fatal_or_budget): RunStatus.REPORTING,
    (RunStatus.REPORTING, Trigger.report_done): RunStatus.DONE,
    (RunStatus.REPORTING, Trigger.report_failed): RunStatus.FAILED,
    (RunStatus.REPORTING, Trigger.report_cancelled): RunStatus.CANCELLED,
}

_CANCEL_TRANSITIONS: dict[tuple[RunStatus, Trigger], RunStatus] = {
    (status, Trigger.cancel): RunStatus.REPORTING
    for status in RunStatus
    if status not in TERMINAL_STATUSES
}

_TRANSITIONS = _EXPLICIT_TRANSITIONS | _CANCEL_TRANSITIONS


def next_status(current: RunStatus, trigger: Trigger) -> RunStatus:
    """Return the next lifecycle status or raise for an illegal transition."""
    if current in TERMINAL_STATUSES:
        raise ValueError(f"Terminal status {current.value} has no outgoing transitions.")

    try:
        return _TRANSITIONS[(current, trigger)]
    except KeyError as exc:
        raise ValueError(f"Invalid transition: {current.value} --{trigger.value}--> ?") from exc
