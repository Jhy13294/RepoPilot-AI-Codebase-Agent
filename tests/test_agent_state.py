from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.agent.state import (
    TERMINAL_STATUSES,
    AgentState,
    Budgets,
    PlanStep,
    PlanStepStatus,
    RunStatus,
    TaskSpec,
    Trigger,
    next_status,
)
from app.schemas.agent_io import AskStatus
from app.tools.registry import ToolTraceRecord

VALID_TRANSITIONS = [
    (RunStatus.PLANNING, Trigger.plan_produced, RunStatus.EXECUTING),
    (RunStatus.EXECUTING, Trigger.request_approval, RunStatus.AWAITING_APPROVAL),
    (RunStatus.AWAITING_APPROVAL, Trigger.approval_granted, RunStatus.EXECUTING),
    (RunStatus.AWAITING_APPROVAL, Trigger.approval_denied, RunStatus.REPLANNING),
    (RunStatus.EXECUTING, Trigger.step_finished, RunStatus.VERIFYING),
    (RunStatus.VERIFYING, Trigger.verdict_proceed, RunStatus.EXECUTING),
    (RunStatus.VERIFYING, Trigger.verdict_retry, RunStatus.EXECUTING),
    (RunStatus.VERIFYING, Trigger.verdict_replan, RunStatus.REPLANNING),
    (RunStatus.REPLANNING, Trigger.replan_ok, RunStatus.EXECUTING),
    (RunStatus.REPLANNING, Trigger.replan_exhausted, RunStatus.REPORTING),
    (RunStatus.VERIFYING, Trigger.all_steps_done, RunStatus.REPORTING),
    (RunStatus.EXECUTING, Trigger.fatal_or_budget, RunStatus.REPORTING),
    (RunStatus.REPORTING, Trigger.report_done, RunStatus.DONE),
    (RunStatus.REPORTING, Trigger.report_failed, RunStatus.FAILED),
    (RunStatus.REPORTING, Trigger.report_cancelled, RunStatus.CANCELLED),
]
NON_REPORTING_STATUSES = [status for status in RunStatus if status is not RunStatus.REPORTING]
NON_TERMINAL_STATUSES = [status for status in RunStatus if status not in TERMINAL_STATUSES]


def _task() -> TaskSpec:
    return TaskSpec(task_type="question", prompt="Where is parse_date defined?", repo=".")


def _step(index: int = 0) -> PlanStep:
    return PlanStep(
        index=index,
        intent="Find the relevant definition.",
        suggested_tools=["search_code", "read_file"],
        success_check="A file:line citation identifies the definition.",
    )


def _trace_record() -> ToolTraceRecord:
    return ToolTraceRecord(
        run_id="run-1",
        tool_name="search_code",
        args={"query": "def parse_date"},
        ok=True,
        error_type=None,
        latency_ms=3,
        truncated=False,
        ts=datetime.now(UTC),
    )


def _agent_state(**overrides: object) -> AgentState:
    values = {
        "run_id": "run-1",
        "task": _task(),
        "plan": [_step()],
        "cursor": 0,
        "tool_history": [_trace_record()],
        "status": RunStatus.PLANNING,
    }
    values.update(overrides)
    return AgentState.model_validate(values)


@pytest.mark.parametrize(("current", "trigger", "expected"), VALID_TRANSITIONS)
def test_next_status__valid_agent_design_edges(
    current: RunStatus,
    trigger: Trigger,
    expected: RunStatus,
) -> None:
    assert next_status(current, trigger) is expected


@pytest.mark.parametrize(
    ("current", "trigger"),
    [
        (RunStatus.PLANNING, Trigger.step_finished),
        (RunStatus.PLANNING, Trigger.report_done),
        (RunStatus.EXECUTING, Trigger.plan_produced),
        (RunStatus.VERIFYING, Trigger.approval_granted),
        (RunStatus.REPORTING, Trigger.step_finished),
    ],
)
def test_next_status__illegal_combinations_raise(
    current: RunStatus,
    trigger: Trigger,
) -> None:
    with pytest.raises(ValueError):
        next_status(current, trigger)


@pytest.mark.parametrize("terminal_status", sorted(TERMINAL_STATUSES))
@pytest.mark.parametrize("trigger", list(Trigger))
def test_next_status__terminal_statuses_have_no_outgoing_edges(
    terminal_status: RunStatus,
    trigger: Trigger,
) -> None:
    with pytest.raises(ValueError):
        next_status(terminal_status, trigger)


@pytest.mark.parametrize("current", NON_REPORTING_STATUSES)
@pytest.mark.parametrize("trigger", list(Trigger))
def test_next_status__terminal_statuses_are_only_reachable_from_reporting(
    current: RunStatus,
    trigger: Trigger,
) -> None:
    try:
        target = next_status(current, trigger)
    except ValueError:
        return

    assert target not in TERMINAL_STATUSES


@pytest.mark.parametrize("current", NON_TERMINAL_STATUSES)
def test_next_status__cancel_from_any_non_terminal_status_goes_to_reporting(
    current: RunStatus,
) -> None:
    assert next_status(current, Trigger.cancel) is RunStatus.REPORTING


def test_agent_state__run_status_contains_full_lifecycle() -> None:
    assert [status.value for status in RunStatus] == [
        "PLANNING",
        "EXECUTING",
        "AWAITING_APPROVAL",
        "VERIFYING",
        "REPLANNING",
        "REPORTING",
        "DONE",
        "FAILED",
        "CANCELLED",
    ]


def test_agent_state__budgets_defaults_align_config_budget_table() -> None:
    budgets = Budgets()

    assert budgets.max_steps == 20
    assert budgets.max_replans == 3
    assert budgets.max_fix_cycles == 2
    assert budgets.token_cap is None
    assert budgets.cost_cap is None


def test_agent_state__flat_schema_defaults_and_tool_history_reuse() -> None:
    state = _agent_state()

    assert set(AgentState.model_fields) == {
        "run_id",
        "task",
        "plan",
        "cursor",
        "tool_history",
        "scratchpad",
        "budgets",
        "status",
        "steps_used",
        "replans_used",
        "fix_cycles_used",
    }
    assert state.scratchpad == ""
    assert state.budgets == Budgets()
    assert state.steps_used == 0
    assert state.replans_used == 0
    assert state.fix_cycles_used == 0
    assert isinstance(state.tool_history[0], ToolTraceRecord)


def test_agent_state__plan_step_defaults_to_pending() -> None:
    step = _step()

    assert step.status is PlanStepStatus.pending


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"plan": [{"index": -1, "intent": "x", "suggested_tools": [], "success_check": "x"}]},
        {"cursor": -1},
        {"steps_used": -1},
        {"replans_used": -1},
        {"fix_cycles_used": -1},
    ],
)
def test_agent_state__agent_state_numeric_boundaries(invalid_payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _agent_state(**invalid_payload)


@pytest.mark.parametrize(
    "invalid_budget",
    [
        {"max_steps": 0},
        {"max_replans": 0},
        {"max_fix_cycles": 0},
    ],
)
def test_agent_state__budget_boundaries(invalid_budget: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Budgets.model_validate(invalid_budget)


def test_agent_state__schemas_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(task_type="question", prompt="p", repo=".", extra_field=True)
    with pytest.raises(ValidationError):
        PlanStep(index=0, intent="i", suggested_tools=[], success_check="s", extra_field=True)
    with pytest.raises(ValidationError):
        Budgets(extra_field=True)
    with pytest.raises(ValidationError):
        _agent_state(extra_field=True)


def test_agent_state__schemas_are_frozen() -> None:
    state = _agent_state()

    with pytest.raises(ValidationError):
        state.status = RunStatus.EXECUTING


def test_agent_io__ask_status_is_separate_from_lifecycle_run_status() -> None:
    assert AskStatus.answered.value == "answered"
    assert not hasattr(AskStatus, "PLANNING")
