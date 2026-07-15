"""Orchestrate Planner, Executor, and Critic into one bounded agent run."""

import json
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

from pydantic import JsonValue

from app.agent.citations import CitationStatus, validate_citations
from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.planner import Planner
from app.agent.reporter import Reporter, ReporterError
from app.agent.state import (
    AgentState,
    Budgets,
    PlanStep,
    PlanStepStatus,
    RunStatus,
    TaskSpec,
    Trigger,
    next_status,
)
from app.agent.usage import UsageAccumulator
from app.safety.path_jail import PathJail
from app.schemas.agent_io import (
    AnalysisReport,
    CitationGrounding,
    GroundingReport,
    RunResult,
    StepResult,
    Verdict,
    VerdictDecision,
)
from app.schemas.llm_io import Usage
from app.schemas.tool_io import ErrorType
from app.schemas.trace import TraceEvent, TraceEventKind
from app.storage.db import Database
from app.storage.trace_store import TraceStore, render_timeline
from app.tools.registry import ToolTraceRecord

_DEFAULT_BUDGETS = Budgets()
_MAX_RENDERED_FAILING_TEST_IDS = 10


def run_agent_loop(
    task: TaskSpec,
    *,
    planner: Planner,
    executor: Executor,
    critic: Critic,
    store: TraceStore,
    database: Database,
    budgets: Budgets = _DEFAULT_BUDGETS,
    reporter: Reporter | None = None,
    jail: PathJail | None = None,
) -> RunResult:
    """Run one Planner-Executor-Critic lifecycle to DONE or FAILED."""
    run_id = str(uuid4())
    state = AgentState(
        run_id=run_id,
        task=task,
        plan=[],
        cursor=0,
        tool_history=[],
        budgets=budgets,
        status=RunStatus.PLANNING,
    )
    usage = UsageAccumulator()
    step_result: StepResult | None = None
    evidence_window_start = 0
    latest_verdict: Verdict | None = None
    failure_summary: str | None = None
    report_success = False
    denials_used = 0
    pending_replan_summary: str | None = None

    while state.status is not RunStatus.REPORTING:
        if state.status is RunStatus.PLANNING:
            before = len(store.read(run_id))
            try:
                plan_result = planner.plan(run_id, task, repo_overview=None)
            except Exception as exc:
                _add_usage_from_events(usage, store.read(run_id)[before:])
                failure_summary = f"Planning failed: {exc}"
                state = _transition(state, Trigger.fatal_or_budget, database)
                continue

            usage.add(plan_result.usage)
            state = _transition(
                state,
                Trigger.plan_produced,
                database,
                plan=plan_result.plan,
                cursor=0,
                scratchpad="",
                tool_history=[],
            )
            continue

        if state.status is RunStatus.EXECUTING:
            if state.steps_used >= state.budgets.max_steps:
                failure_summary = f"Step budget exhausted after {state.steps_used} step(s)."
                state = _transition(
                    _mark_current_step(state, PlanStepStatus.failed),
                    Trigger.fatal_or_budget,
                    database,
                )
                continue

            try:
                step = _current_step(state)
            except RuntimeError as exc:
                failure_summary = str(exc)
                state = _transition(state, Trigger.fatal_or_budget, database)
                continue

            attempt_state = state.model_copy(update={"steps_used": state.steps_used + 1})
            evidence_window_start = len(store.read(run_id))
            try:
                step_result = executor.execute_step(
                    run_id,
                    step,
                    scratchpad=attempt_state.scratchpad,
                    repo_overview=None,
                )
            except Exception as exc:
                window = store.read(run_id)[evidence_window_start:]
                _add_usage_from_events(usage, window)
                failure_summary = f"Execution failed at step {step.index}: {exc}"
                attempt_state = attempt_state.model_copy(
                    update={"tool_history": attempt_state.tool_history + _tool_records(window)}
                )
                state = _transition(
                    _mark_current_step(attempt_state, PlanStepStatus.failed),
                    Trigger.fatal_or_budget,
                    database,
                )
                continue

            usage.add(step_result.usage)
            window = store.read(run_id)[evidence_window_start:]
            attempt_state = attempt_state.model_copy(
                update={"tool_history": attempt_state.tool_history + _tool_records(window)}
            )
            denial = _detect_denial(window)
            if denial is not None:
                denials_used += 1
                if denials_used >= state.budgets.max_denials:
                    failure_summary = (
                        "Approval denial budget exhausted after "
                        f"{denials_used} denial(s); the human reviewer denied the requested "
                        "high-risk action."
                    )
                    state = _transition(
                        _mark_current_step(attempt_state, PlanStepStatus.failed),
                        Trigger.fatal_or_budget,
                        database,
                    )
                    continue

                pending_replan_summary = _denial_replan_summary(denial)
                state = _transition(
                    attempt_state,
                    Trigger.approval_denied,
                    database,
                )
                continue

            state = _transition(attempt_state, Trigger.step_finished, database)
            continue

        if state.status is RunStatus.VERIFYING:
            if step_result is None:
                failure_summary = "Verifier reached without an executor step result."
                state = _transition(
                    _mark_current_step(state, PlanStepStatus.failed),
                    Trigger.fatal_or_budget,
                    database,
                )
                continue

            step = _current_step(state)
            raw_evidence = _raw_evidence(store.read(run_id)[evidence_window_start:])
            before = len(store.read(run_id))
            try:
                verdict = critic.critique(
                    run_id,
                    step,
                    step_result,
                    raw_evidence=raw_evidence,
                    repo_overview=None,
                )
            except Exception as exc:
                _add_usage_from_events(usage, store.read(run_id)[before:])
                failure_summary = f"Critic failed at step {step.index}: {exc}"
                state = _transition(
                    _mark_current_step(state, PlanStepStatus.failed),
                    Trigger.fatal_or_budget,
                    database,
                )
                continue

            usage.add(verdict.usage)
            latest_verdict = verdict

            if verdict.decision is VerdictDecision.proceed:
                scratchpad = _append_step_findings(state.scratchpad, step, step_result, verdict)
                marked = _mark_current_step(state, PlanStepStatus.done).model_copy(
                    update={"scratchpad": scratchpad}
                )
                if state.cursor == len(state.plan) - 1:
                    report_success = True
                    state = _transition(marked, Trigger.all_steps_done, database)
                    continue

                state = _transition(
                    marked,
                    Trigger.verdict_proceed,
                    database,
                    cursor=state.cursor + 1,
                )
                continue

            if verdict.decision is VerdictDecision.retry:
                scratchpad = _append_retry_hint(state.scratchpad, step, verdict)
                if state.fix_cycles_used >= state.budgets.max_fix_cycles:
                    state = _transition(
                        state.model_copy(update={"scratchpad": scratchpad}),
                        Trigger.verdict_replan,
                        database,
                    )
                    continue

                state = _transition(
                    state,
                    Trigger.verdict_retry,
                    database,
                    fix_cycles_used=state.fix_cycles_used + 1,
                    scratchpad=scratchpad,
                )
                continue

            if verdict.decision is VerdictDecision.replan:
                scratchpad = _append_replan_hint(state.scratchpad, step, verdict)
                state = _transition(
                    state,
                    Trigger.verdict_replan,
                    database,
                    scratchpad=scratchpad,
                )
                continue

        if state.status is RunStatus.REPLANNING:
            if state.replans_used >= state.budgets.max_replans:
                failure_summary = f"Replan budget exhausted after {state.replans_used} replan(s)."
                state = _transition(
                    _mark_current_step(state, PlanStepStatus.failed),
                    Trigger.replan_exhausted,
                    database,
                )
                continue

            failure_summary_for_planner = pending_replan_summary or _replan_summary(latest_verdict)
            pending_replan_summary = None
            attempt_state = state.model_copy(update={"replans_used": state.replans_used + 1})
            before = len(store.read(run_id))
            try:
                plan_result = planner.plan(
                    run_id,
                    task,
                    repo_overview=None,
                    failure_summary=failure_summary_for_planner,
                )
            except Exception as exc:
                _add_usage_from_events(usage, store.read(run_id)[before:])
                failure_summary = f"Replanning failed: {exc}"
                state = _transition(attempt_state, Trigger.fatal_or_budget, database)
                continue

            usage.add(plan_result.usage)
            state = _transition(
                attempt_state,
                Trigger.replan_ok,
                database,
                plan=plan_result.plan,
                cursor=0,
            )
            continue

    return _finalize_run(
        state,
        success=report_success,
        failure_summary=failure_summary,
        store=store,
        database=database,
        usage=usage,
        task=task,
        reporter=reporter,
        jail=jail,
    )


def _transition(
    state: AgentState,
    trigger: Trigger,
    database: Database,
    **updates: object,
) -> AgentState:
    next_state = state.model_copy(update={"status": next_status(state.status, trigger), **updates})
    database.save_state(next_state)
    return next_state


def _finalize_run(
    state: AgentState,
    *,
    success: bool,
    failure_summary: str | None,
    store: TraceStore,
    database: Database,
    usage: UsageAccumulator,
    task: TaskSpec,
    reporter: Reporter | None,
    jail: PathJail | None,
) -> RunResult:
    final_findings = _final_findings(state)
    timeline_digest = render_timeline(store.read(state.run_id))
    report: AnalysisReport | None = None
    report_error: str | None = None
    if reporter is not None:
        try:
            report = reporter.report(
                task,
                outcome="succeeded" if success else "failed",
                final_findings=final_findings,
                timeline_digest=timeline_digest,
                failure_summary=failure_summary,
            )
            usage.add(report.usage)
        except ReporterError as exc:
            report_error = f"{exc.reason.value}: {exc}"
            usage.add(exc.usage)
        except Exception as exc:
            report_error = str(exc)

    if report is not None:
        summary = f"{report.headline}\n\n{report.analysis}"
    else:
        summary = _report_summary(state, success=success, failure_summary=failure_summary)

    grounding = _ground_report(report, jail)
    terminal_status = RunStatus.DONE if success else RunStatus.FAILED
    payload: dict[str, JsonValue] = {
        "summary": summary,
        "status": state.status.value,
        "terminal_status": terminal_status.value,
        "steps_used": state.steps_used,
        "replans_used": state.replans_used,
        "fix_cycles_used": state.fix_cycles_used,
        "final_findings": final_findings,
    }
    if failure_summary is not None:
        payload["failure_summary"] = failure_summary
    if report is not None:
        payload["headline"] = report.headline
        payload["analysis"] = report.analysis
        payload["confidence"] = report.confidence.value
        payload["open_questions"] = cast(JsonValue, report.open_questions)
        payload["suspects"] = cast(
            JsonValue,
            [suspect.model_dump() for suspect in report.suspects],
        )
        payload["citations"] = cast(JsonValue, report.citations)
        if grounding is not None:
            payload["grounding"] = cast(
                JsonValue,
                [check.model_dump() for check in grounding.checks],
            )
    elif report_error is not None:
        payload["report_generation_error"] = report_error

    store.append(state.run_id, TraceEventKind.report, payload)
    trigger = Trigger.report_done if success else Trigger.report_failed
    terminal_state = _transition(state, trigger, database)
    database.index_events(state.run_id, store.read(state.run_id))
    return RunResult(
        run_id=state.run_id,
        status=terminal_state.status,
        summary=summary,
        steps_used=terminal_state.steps_used,
        replans_used=terminal_state.replans_used,
        fix_cycles_used=terminal_state.fix_cycles_used,
        usage=usage.snapshot(),
        report=report,
        grounding=grounding,
    )


def _ground_report(report: AnalysisReport | None, jail: PathJail | None) -> GroundingReport | None:
    if report is None or jail is None:
        return None

    citations = _ordered_unique_citations(report)
    if not citations:
        return None

    try:
        citation_report = validate_citations(citations, jail)
        return GroundingReport(
            checks=[
                CitationGrounding(
                    citation=check.raw,
                    status=check.status.value,
                    grounded=check.status is CitationStatus.valid,
                    detail=check.detail,
                )
                for check in citation_report.checks
            ]
        )
    except Exception:
        return None


def _ordered_unique_citations(report: AnalysisReport) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for citation in [*report.citations, *(suspect.path for suspect in report.suspects)]:
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


def _current_step(state: AgentState) -> PlanStep:
    try:
        return state.plan[state.cursor]
    except IndexError as exc:
        raise RuntimeError(
            f"Agent cursor {state.cursor} is outside a plan with {len(state.plan)} step(s)."
        ) from exc


def _mark_current_step(state: AgentState, status: PlanStepStatus) -> AgentState:
    if not state.plan or state.cursor >= len(state.plan):
        return state

    updated_plan = [
        step.model_copy(update={"status": status}) if index == state.cursor else step
        for index, step in enumerate(state.plan)
    ]
    return state.model_copy(update={"plan": updated_plan})


def _raw_evidence(events: Sequence[TraceEvent]) -> list[str]:
    return [_render_tool_call(event) for event in events if event.kind is TraceEventKind.tool_call]


def _render_tool_call(event: TraceEvent) -> str:
    tool_name = _payload_string(event.payload, "tool_name") or "unknown_tool"
    args = json.dumps(_payload_args(event.payload), sort_keys=True, separators=(",", ":"))
    ok = event.payload.get("ok")
    if ok is True:
        status = "ok"
    else:
        error_type = _payload_string(event.payload, "error_type") or "unknown"
        status = f"error:{error_type}"
    truncated = " truncated" if event.payload.get("truncated") is True else ""
    rendered = f"seq={event.seq} tool={tool_name} status={status}{truncated} args={args}"
    outcome = _payload_outcome(event.payload)
    if outcome is None:
        return rendered

    failing_test_ids = _outcome_failing_test_ids(outcome)
    visible_ids = failing_test_ids[:_MAX_RENDERED_FAILING_TEST_IDS]
    failing = json.dumps(visible_ids, separators=(",", ":"))
    omitted = len(failing_test_ids) - len(visible_ids)
    more = f"(+{omitted} more)" if omitted else ""
    outcome_text = (
        f"failed:{_outcome_count(outcome, 'failed')},"
        f"passed:{_outcome_count(outcome, 'passed')},"
        f"failing:{failing}{more}"
    )
    errors = _outcome_count(outcome, "errors")
    if errors:
        outcome_text = f"{outcome_text},errors:{errors}"
    return f"{rendered} outcome={outcome_text}"


def _tool_records(events: Sequence[TraceEvent]) -> list[ToolTraceRecord]:
    records = []
    for event in events:
        if event.kind is not TraceEventKind.tool_call:
            continue
        records.append(
            ToolTraceRecord(
                run_id=event.run_id,
                tool_name=_payload_string(event.payload, "tool_name") or "unknown_tool",
                args=_payload_args(event.payload),
                ok=event.payload.get("ok") is True,
                error_type=_payload_error_type(event.payload),
                latency_ms=event.latency_ms or 0,
                truncated=event.payload.get("truncated") is True,
                ts=event.ts,
            )
        )
    return records


def _detect_denial(events: Sequence[TraceEvent]) -> TraceEvent | None:
    for event in events:
        if (
            event.kind is TraceEventKind.tool_call
            and _payload_error_type(event.payload) is ErrorType.ApprovalDeniedError
        ):
            return event
    return None


def _denial_replan_summary(event: TraceEvent) -> str:
    tool_name = _payload_string(event.payload, "tool_name") or "unknown_tool"
    args = _payload_args(event.payload)
    diff = args.get("diff")
    if isinstance(diff, str):
        denied_context = f"Denied diff:\n{diff}"
    else:
        denied_context = (
            f"Denied call args: {json.dumps(args, sort_keys=True, separators=(',', ':'))}"
        )
    return (
        f"The human reviewer denied the previous {tool_name} call. {denied_context}\n"
        "Propose a different approach; do not resubmit the identical call or diff."
    )


def _payload_args(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = payload.get("args")
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, JsonValue], value)


def _payload_outcome(payload: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    value = payload.get("outcome")
    if not isinstance(value, dict):
        return None
    return cast(dict[str, JsonValue], value)


def _outcome_count(outcome: dict[str, JsonValue], key: str) -> int:
    value = outcome.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _outcome_failing_test_ids(outcome: dict[str, JsonValue]) -> list[str]:
    value = outcome.get("failing_test_ids")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _payload_string(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _payload_error_type(payload: dict[str, JsonValue]) -> ErrorType | None:
    value = _payload_string(payload, "error_type")
    if value is None:
        return None
    try:
        return ErrorType(value)
    except ValueError:
        return None


def _add_usage_from_events(usage: UsageAccumulator, events: Sequence[TraceEvent]) -> None:
    for event in events:
        if event.tokens_in is None and event.tokens_out is None and event.cost_usd is None:
            continue
        usage.add(
            Usage(
                tokens_in=event.tokens_in or 0,
                tokens_out=event.tokens_out or 0,
                cost_usd=event.cost_usd,
            )
        )


def _append_step_findings(
    scratchpad: str,
    step: PlanStep,
    result: StepResult,
    verdict: Verdict,
) -> str:
    entry = f"Step {step.index} findings: {result.findings}\nCritic reason: {verdict.reason}"
    return _append_scratchpad(scratchpad, entry)


def _append_retry_hint(scratchpad: str, step: PlanStep, verdict: Verdict) -> str:
    hint = verdict.hint or verdict.reason
    return _append_scratchpad(scratchpad, f"Retry step {step.index}: {hint}")


def _append_replan_hint(scratchpad: str, step: PlanStep, verdict: Verdict) -> str:
    hint = verdict.hint or verdict.reason
    return _append_scratchpad(scratchpad, f"Replan after step {step.index}: {hint}")


def _append_scratchpad(scratchpad: str, entry: str) -> str:
    if not scratchpad:
        return entry
    return f"{scratchpad}\n\n{entry}"


def _replan_summary(verdict: Verdict | None) -> str:
    if verdict is None:
        return "The previous execution branch requested replanning."
    return verdict.hint or verdict.reason


def _report_summary(
    state: AgentState,
    *,
    success: bool,
    failure_summary: str | None,
) -> str:
    outcome = "succeeded" if success else "failed"
    parts = [
        f"Run {state.run_id} {outcome}.",
        (
            f"status={state.status.value}; steps={state.steps_used}; "
            f"replans={state.replans_used}; fix_cycles={state.fix_cycles_used}."
        ),
    ]
    if failure_summary is not None:
        parts.append(f"Failure: {failure_summary}")
    findings = _final_findings(state)
    if findings:
        parts.append(f"Final findings: {findings}")
    return " ".join(parts)


def _final_findings(state: AgentState) -> str:
    return state.scratchpad.strip()
